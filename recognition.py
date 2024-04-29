import torch
import torch.nn as nn
import math
import numpy as np
import tensorflow as tf
from copy import deepcopy
from Tokenizer import GlossTokenizer_S2G
from Visualhead import VisualHead
from itertools import groupby


def ctc_decode_func(tf_gloss_logits, input_lengths, beam_size):
    ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
        inputs=tf_gloss_logits,
        sequence_length=input_lengths.cpu().detach().numpy(),
        beam_width=beam_size,
        top_paths=1,
    )
    ctc_decode = ctc_decode[0]
    tmp_gloss_sequences = [[] for i in range(input_lengths.shape[0])]
    for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
        tmp_gloss_sequences[dense_idx[0]].append(
            ctc_decode.values[value_idx].numpy() + 1
        )
    decoded_gloss_sequences = []
    for seq_idx in range(0, len(tmp_gloss_sequences)):
        decoded_gloss_sequences.append(
            [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
        )
    return decoded_gloss_sequences


class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x


class STAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=2, num_node=27, num_frame=400,
                 kernel_size=1, stride=1, t_kernel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
                 use_temporal_att=False, use_spatial_att=True, attentiondrop=0., use_pes=True, use_pet=False):
        super(STAttentionBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet

        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            atts = torch.zeros((1, num_subset, num_node, num_node))
            self.register_buffer('atts', atts)
            self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_s:
                self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node,
                                                requires_grad=True)

            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
                nn.BatchNorm2d(out_channels),
            )
        padd = int(t_kernel / 2)
        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (t_kernel, 1), padding=(padd, 0), bias=True, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if use_temporal_att:
                self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            if use_spatial_att:
                self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            self.downt2 = lambda x: x
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):
        N, C, T, V = x.size()
        if self.use_spatial_att:
            attention = self.atts
            if self.use_pes:
                y = self.pes(x)
            else:
                y = x
            if self.att_s:
                q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv [4,16,t,v]
                attention = attention + self.tan(torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
            if self.glo_reg_s:
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous() \
                .view(N, self.num_subset * self.in_channels, T, V)
            y = self.out_nets(y)  # nctv
            y = self.relu(self.downs1(x) + y)
            y = self.ff_nets(y)
            y = self.relu(self.downs2(x) + y)
        else:
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)
        z = self.out_nett(y)
        z = self.relu(self.downt2(y) + z)

        return z


class DSTA(nn.Module):
    def __init__(self, num_frame=400,
                 num_subset=6, dropout=0.1,
                 cfg=None,
                 args=None,
                 num_channel=2, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
                 use_temporal_att=False, use_spatial_att=True, attentiondrop=0.1, use_pet=False,
                 use_pes=True, mode='SLR'):
        super(DSTA, self).__init__()
        self.mode = mode
        self.cfg = cfg
        self.args = args
        config = self.cfg['net']
        self.out_channels = config[-1][1]
        in_channels = config[0][0]
        self.num_frame = num_frame
        param = {
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop
        }
        self.left_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.right_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.body_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.face_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.face_graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
            self.face_graph_layers.append(
                STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, t_kernel=t_kernel,num_node=26,
                                 num_frame=num_frame,
                                 **param))
            num_frame = int(num_frame / stride + 0.5)
        num_frame = self.num_frame
        self.left_graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
            self.left_graph_layers.append(
                STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_node=27,
                                 t_kernel=t_kernel, num_frame=num_frame,
                                 **param))
            num_frame = int(num_frame / stride + 0.5)
        num_frame = self.num_frame
        self.right_graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
            self.right_graph_layers.append(
                STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, t_kernel=t_kernel,
                                 num_frame=num_frame,
                                 **param))
            num_frame = int(num_frame / stride + 0.5)
        num_frame = self.num_frame
        self.body_graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
            self.body_graph_layers.append(
                STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_node=79,
                                 t_kernel=t_kernel, num_frame=num_frame,
                                 **param))
            num_frame = int(num_frame / stride + 0.5)
        self.drop_out = nn.Dropout(dropout)


    def forward(self,src_input):
        x = src_input['keypoint'].cuda()
        N, C, T, V = x.shape
        x = x.permute(0, 1, 2, 3).contiguous().view(N, C, T, V)
        left = self.left_input_map(x[:, :, :, self.cfg['left']])
        right = self.right_input_map(x[:, :, :, self.cfg['right']])
        face = self.face_input_map(x[:, :, :, self.cfg['face']])
        body = self.body_input_map(x[:, :, :, self.cfg['body']])
        for i, m in enumerate(self.face_graph_layers):
            face = m(face)
        for i, m in enumerate(self.left_graph_layers):
            left = m(left)
        for i, m in enumerate(self.right_graph_layers):
            right = m(right)
        for i, m in enumerate(self.body_graph_layers):
            body = m(body)  # [B,256,T/4,N] -> [B,256]
        left = left.permute(0, 2, 1, 3).contiguous()
        right = right.permute(0, 2, 1, 3).contiguous()
        face = face.permute(0, 2, 1, 3).contiguous()
        body = body.permute(0, 2, 1, 3).contiguous()
        body = body.mean(3)
        face = face.mean(3)
        left = left.mean(3)
        right = right.mean(3)
        output = torch.cat([left, face, right, body], dim=-1)
        left_output = torch.cat([left, face], dim=-1)
        right_output = torch.cat([right,  face], dim=-1)
        return output, left_output, right_output, body


class Recognition(nn.Module):
    def __init__(self, cfg, args, input_streams=None):
        super(Recognition, self).__init__()
        self.cfg = cfg
        self.args = args
        self.input_type = cfg['input_type']
        self.gloss_tokenizer = GlossTokenizer_S2G(cfg['GlossTokenizer'])
        self.input_streams = input_streams
        self.fuse_method = cfg.get('fuse_method', 'empty')
        self.heatmap_cfg = cfg.get('heatmap_cfg', {})
        if self.input_type == 'keypoint':
            self.visual_backbone = None
            self.rgb_visual_head = None
            self.visual_backbone_keypoint = DSTA(cfg=self.cfg['DSTA-Net'],num_channel=3,args=args)
            self.fuse_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['fuse_visual_head'])
            self.body_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['body_visual_head'])
            self.left_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['left_visual_head'])
            self.right_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['right_visual_head'])
        else:
            raise ValueError
        if 'pretrained_path' in self.cfg:
            load_dict = torch.load(cfg['pretrained_path'],map_location='cpu')['model']
            backbone_dict = {}
            for k, v in load_dict.items():
                backbone_dict[k.replace('recognition_network.','')] = v
            self.load_state_dict(backbone_dict)
        self.recognition_loss_func = torch.nn.CTCLoss(
            blank=0,
            zero_infinity=True,
            reduction='sum'
        )

    def compute_recognition_loss(self, gloss_labels, gloss_lengths, gloss_probabilities_log, input_lengths):
        loss = self.recognition_loss_func(
            log_probs = gloss_probabilities_log.permute(1,0,2), #T,N,C
            targets = gloss_labels,
            input_lengths = input_lengths,
            target_lengths = gloss_lengths
        )
        loss = loss/gloss_probabilities_log.shape[0]
        return loss

    def decode(self, gloss_logits, beam_size, input_lengths):
        gloss_logits = gloss_logits.permute(1, 0, 2) #T,B,V  [10,1,1124]
        gloss_logits = gloss_logits.cpu().detach().numpy()
        tf_gloss_logits = np.concatenate(
            (gloss_logits[:, :, 1:], gloss_logits[:, :, 0, None]),
            axis=-1,
        )
        decoded_gloss_sequences = ctc_decode_func(
            tf_gloss_logits=tf_gloss_logits,
            input_lengths=input_lengths,
            beam_size=beam_size
        )
        return decoded_gloss_sequences

    def forward(self, src_input):
        head_outputs = {}
        if self.input_type == 'keypoint':
            fuse, left_output, right_output, body = self.visual_backbone_keypoint(src_input)
            body_head = self.body_visual_head(
                x=body,
                mask=src_input['mask'].cuda(),
                valid_len_in=src_input['new_src_lengths'].cuda())
            fuse_head = self.fuse_visual_head(
                x=fuse,
                mask=src_input['mask'].cuda(),
                valid_len_in=src_input['new_src_lengths'].cuda())
            left_head = self.left_visual_head(
                x=left_output,
                mask=src_input['mask'].cuda(),
                valid_len_in=src_input['new_src_lengths'].cuda())
            right_head = self.right_visual_head(
                x=right_output,
                mask=src_input['mask'].cuda(),
                valid_len_in=src_input['new_src_lengths'].cuda())
            head_outputs = {'ensemble_last_gloss_logits': (left_head['gloss_probabilities'] + right_head['gloss_probabilities'] +
                                                           body_head['gloss_probabilities']+fuse_head['gloss_probabilities']).log(),
                            'fuse': fuse,
                            'fuse_gloss_logits': fuse_head['gloss_logits'],
                            'fuse_gloss_probabilities_log': fuse_head['gloss_probabilities_log'],
                            'body_gloss_logits': body_head['gloss_logits'],
                            'body_gloss_probabilities_log': body_head['gloss_probabilities_log'],
                            'left_gloss_logits': left_head['gloss_logits'],
                            'left_gloss_probabilities_log': left_head['gloss_probabilities_log'],
                            'right_gloss_logits': right_head['gloss_logits'],
                            'right_gloss_probabilities_log': right_head['gloss_probabilities_log'],
                            }

            head_outputs['ensemble_last_gloss_probabilities_log'] = head_outputs['ensemble_last_gloss_logits'].log_softmax(2)
            head_outputs['ensemble_last_gloss_probabilities'] = head_outputs['ensemble_last_gloss_logits'].softmax(2)
            self.cfg['gloss_feature_ensemble'] = self.cfg.get('gloss_feature_ensemble', 'gloss_feature')
            head_outputs['gloss_feature'] = fuse_head[self.cfg['gloss_feature_ensemble']]
        else:
            raise ValueError
        outputs = {**head_outputs,
                   'input_lengths':src_input['new_src_lengths']}

        for k in ['left', 'right', 'fuse', 'body']:
            outputs[f'recognition_loss_{k}'] = self.compute_recognition_loss(
                gloss_labels=src_input['gloss_input']['gloss_labels'].cuda(),
                gloss_lengths=src_input['gloss_input']['gls_lengths'].cuda(),
                gloss_probabilities_log=head_outputs[f'{k}_gloss_probabilities_log'],
                input_lengths=src_input['new_src_lengths'].cuda())
        outputs['recognition_loss'] = outputs['recognition_loss_left'] + outputs['recognition_loss_right'] + \
                                      outputs['recognition_loss_fuse'] + outputs['recognition_loss_body']
        if 'cross_distillation' in self.cfg:
            loss_func = torch.nn.KLDivLoss(reduction="batchmean")
            for student in ['left', 'right', 'fuse', 'body']:
                teacher_prob = outputs['ensemble_last_gloss_probabilities']
                teacher_prob = teacher_prob.detach()
                student_log_prob = outputs[f'{student}_gloss_probabilities_log']
                outputs[f'{student}_distill_loss'] = loss_func(input=student_log_prob, target=teacher_prob)
                outputs['recognition_loss'] += outputs[f'{student}_distill_loss']
        return outputs