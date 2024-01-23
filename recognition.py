import torch
import torch.nn as nn
import math
import numpy as np
import tensorflow as tf
import torchvision
from copy import deepcopy

from einops import rearrange

from S3D import S3D_backbone
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

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


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
        # pe = position/position.max()*2 -1
        # pe = pe.view(time_len, joint_num).unsqueeze(0).unsqueeze(0)
        # Compute the positional encodings once in log space.
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

class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # MBart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)

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

        self.embed_positions = LearnedPositionalEmbedding(
            num_frame,
            num_node,
        )
        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            atts = torch.zeros((1, num_subset, num_node, num_node))
            self.register_buffer('atts', atts)
            # self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
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
        self.use_temporal_att = use_temporal_att
        if use_temporal_att:
            attt = torch.zeros((1, num_subset, num_frame, num_frame))
            self.register_buffer('attt', attt)
            self.pet = PositionalEncoding(out_channels, num_node, num_frame, 'temporal')
            self.ff_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_t:
                self.in_nett = nn.Conv2d(out_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphat = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_t:
                self.attention0t = nn.Parameter(torch.zeros(1, num_subset, num_frame, num_frame) + torch.eye(num_frame),
                                                requires_grad=True)
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
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
            if use_temporal_att:
                self.downt1 = lambda x: x
            self.downt2 = lambda x: x

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        N, C, T, V = x.size()
        if self.use_spatial_att:
            attention = self.atts
            if self.use_pes:
                y = rearrange(x, 'b c t l -> (b c) t l')
                y = self.embed_positions(y)
                y = rearrange(y, '(b c) t l -> b c t l',b=N)
                # y = self.pes(x)
            else:
                y = x
            if self.att_s:
                q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                attention = attention + (torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
            if self.glo_reg_s:
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.softmax(attention)
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

        if self.use_temporal_att:
            attention = self.attt
            attention = attention[:,:,:y.size(2),:y.size(2)]
            if self.use_pet:
                z = self.pet(y)
            else:
                z = y
            if self.att_t:
                q, k = torch.chunk(self.in_nett(z).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                attention = attention + self.tan(
                    torch.einsum('nsctv,nscqv->nstq', [q, k]) / (self.inter_channels * V)) * self.alphat
            if self.glo_reg_t:
                attention = attention + self.attention0t.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            z = torch.einsum('nctv,nstq->nscqv', [y, attention]).contiguous() \
                .view(N, self.num_subset * self.out_channels, T, V)
            z = self.out_nett(z)  # nctv

            z = self.relu(self.downt1(y) + z)

            z = self.ff_nett(z)

            z = self.relu(self.downt2(y) + z)
        else:
            z = self.out_nett(y)
            z = self.relu(self.downt2(y) + z)

        return z


class DSTA(nn.Module):
    def __init__(self, num_class=1094, num_point=52, num_frame=400,
                 num_subset=2, dropout=0.1,
                 cfg=None,
                 args=None,
                 num_person=1,
                 num_channel=2, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
                 use_temporal_att=False, use_spatial_att=True, attentiondrop=0.1, dropout2d=0.1, use_pet=False,
                 use_pes=True, mode='SLR', pretrain=True):
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
        # self.body_input_map = nn.Sequential(
        #     nn.Conv2d(num_channel, in_channels, 1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.LeakyReLU(0.1),
        # )
        if 'face' in self.cfg['body']:
            self.face_input_map = nn.Sequential(
                nn.Conv2d(num_channel, 64, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1),
            )
            self.face_graph_layers = nn.ModuleList()
            for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
                self.face_graph_layers.append(
                    STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, t_kernel=t_kernel,num_node=26,
                                     num_frame=num_frame,
                                     **param))
                num_frame = int(num_frame / stride + 0.5)
        # if 'mouth' in self.cfg['body']:
        #     num_frame = self.num_frame
        #     self.mouth_input_map = nn.Sequential(
        #         nn.Conv2d(num_channel, 64, 1),
        #         nn.BatchNorm2d(64),
        #         nn.LeakyReLU(0.1),
        #     )
        #     self.mouth_graph_layers = nn.ModuleList()
        #     for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
        #         self.mouth_graph_layers.append(
        #             STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_node=10,
        #                              t_kernel=t_kernel, num_frame=num_frame,
        #                              **param))
        #         num_frame = int(num_frame / stride + 0.5)
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
        # self.body_graph_layers = nn.ModuleList()
        # for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
        #     self.body_graph_layers.append(
        #         STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_node=10,
        #                          t_kernel=t_kernel, num_frame=num_frame,
        #                          **param))
        #     num_frame = int(num_frame / stride + 0.5)

        # self.fc = nn.Linear(self.out_channels, num_class)
        self.drop_out = nn.Dropout(dropout)


    def forward(self,src_input):
        x = src_input['keypoint'].cuda()
        N, C, T, V = x.shape
        x = x.permute(0, 1, 2, 3).contiguous().view(N, C, T, V)
        left = self.left_input_map(x[:, :, :, self.cfg['left']])
        right = self.right_input_map(x[:, :, :, self.cfg['right']])
        face = self.face_input_map(x[:, :, :, self.cfg['face']])
        for i, m in enumerate(self.face_graph_layers):
            face = m(face)
        for i, m in enumerate(self.left_graph_layers):
            left = m(left)
        for i, m in enumerate(self.right_graph_layers):
            right = m(right)
        left = left.permute(0, 2, 1, 3).contiguous()
        right = right.permute(0, 2, 1, 3).contiguous()
        face = face.permute(0, 2, 1, 3).contiguous()
        face = face.mean(3)
        left = left.mean(3)
        right = right.mean(3)
        output = torch.cat([left, right], dim=-1)
        # TODO add face mouth keypoint index
        left_output = torch.cat([left, face], dim=-1)
        right_output = torch.cat([right,  face], dim=-1)
        output = torch.cat([output, face], dim=-1)
        return output, left_output, right_output

class Recognition(nn.Module):
    def __init__(self, cfg, args, transform_cfg, input_streams=None):
        super(Recognition, self).__init__()
        self.cfg = cfg
        self.args = args
        self.input_type = cfg['input_type']
        self.gloss_tokenizer = GlossTokenizer_S2G(cfg['GlossTokenizer'])
        self.input_streams = input_streams
        self.fuse_method = cfg.get('fuse_method', 'empty')
        self.heatmap_cfg = cfg.get('heatmap_cfg', {})
        self.transform_cfg = transform_cfg
        if self.input_type == 'keypoint':
            self.visual_backbone = None
            self.rgb_visual_head = None
            self.visual_backbone_keypoint = DSTA(cfg=self.cfg['DSTA-Net'],num_channel=3,args=args)
            self.fuse_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['fuse_visual_head'])
            self.left_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['left_visual_head'])
            self.right_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['right_visual_head'])
        elif self.input_type == 'video':
            cfg['pyramid'] = cfg.get('pyramid', {'version': None, 'rgb': None, 'pose': None})
            self.visual_backbone = S3D_backbone(in_channel=3, **cfg['s3d'], cfg_pyramid=cfg['pyramid'])
            self.visual_backbone_keypoint = DSTA(cfg=self.cfg['DSTA-Net'])
            self.rgb_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['rgb_visual_head'])
            self.keypoint_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['fuse_visual_head'])
            self.left_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['left_visual_head'])
            self.right_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['right_visual_head'])
            new_cfg = deepcopy(cfg['fuse_visual_head'])
            new_cfg['input_size'] = 832 + 768
            self.visual_head_fuse = VisualHead(
                cls_num=len(self.gloss_tokenizer), **new_cfg)
        elif self.input_type == 'feature':
            self.visual_backbone_keypoint = DSTA(cfg=self.cfg['DSTA-Net'])
            self.rgb_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['rgb_visual_head'])
            self.keypoint_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['fuse_visual_head'])
            self.left_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['left_visual_head'])
            self.right_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['right_visual_head'])
            new_cfg = deepcopy(cfg['fuse_visual_head'])
            new_cfg['input_size'] = 832 + 768
            self.visual_head_fuse = VisualHead(
                cls_num=len(self.gloss_tokenizer), **new_cfg)
        else:
            raise ValueError
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
            fuse, left_output, right_output = self.visual_backbone_keypoint(src_input)
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
            head_outputs = {'ensemble_last_gloss_logits': (left_head['gloss_probabilities'] + right_head['gloss_probabilities'] + fuse_head['gloss_probabilities']).log(),
                            # 'gloss_logits': None,
                            'fuse_gloss_logits': fuse_head['gloss_logits'],
                            'fuse_gloss_probabilities_log': fuse_head['gloss_probabilities_log'],
                            'left_gloss_logits': left_head['gloss_logits'],
                            'left_gloss_probabilities_log': left_head['gloss_probabilities_log'],
                            'right_gloss_logits': right_head['gloss_logits'],
                            'right_gloss_probabilities_log': right_head['gloss_probabilities_log'],
                            }
            head_outputs['ensemble_last_gloss_probabilities_log'] = head_outputs['ensemble_last_gloss_logits'].log_softmax(2)
            head_outputs['ensemble_last_gloss_probabilities'] = head_outputs['ensemble_last_gloss_logits'].softmax(2)

        elif self.input_type == 'video':
            # TODO add src_input['feature'] and feature_mask
            # for name, param in self.visual_backbone.backbone.named_parameters():
            #     print(name, ':', param.requires_grad)
            s3d_outputs = self.visual_backbone(sgn_videos=src_input['videos'].cuda(), sgn_lengths=src_input['src_length'].cuda())
            head_outputs_rgb = self.rgb_visual_head(
                                x=s3d_outputs['sgn'],
                                mask=s3d_outputs['sgn_mask'][-1],
                                valid_len_in=s3d_outputs['valid_len_out'][-1])
            keypoint_outputs, left_output, right_output = self.visual_backbone_keypoint(src_input)
            head_outputs_keypoint = self.keypoint_visual_head(
                x=keypoint_outputs,
                mask=src_input['mask'].cuda(),
                valid_len_in=src_input['new_src_lengths'].cuda())
            head_outputs_left = self.left_visual_head(
                x=left_output,
                mask=src_input['mask'].cuda(),
                valid_len_in=src_input['new_src_lengths'].cuda())
            head_outputs_right = self.right_visual_head(
                x=right_output,
                mask=src_input['mask'].cuda(),
                valid_len_in=src_input['new_src_lengths'].cuda())
            fuse = torch.cat([keypoint_outputs, s3d_outputs['sgn'].cuda()],dim=-1)
            head_outputs_fuse = self.visual_head_fuse(
                x=fuse,
                mask=src_input['mask'].cuda(),
                valid_len_in=src_input['new_src_lengths'])
            head_outputs['fuse_gloss_probabilities'] = head_outputs_fuse['gloss_probabilities']
            head_outputs['fuse_gloss_probabilities_log'] = head_outputs_fuse['gloss_probabilities_log']
            head_outputs['fuse_gloss_logits'] = head_outputs_fuse['gloss_logits']
            head_outputs['fuse_gloss_feature'] = head_outputs_fuse['gloss_feature']
            head_outputs = {
                            'ensemble_last_gloss_logits': (
                                        head_outputs_rgb['gloss_probabilities'] + head_outputs_keypoint['gloss_probabilities'] +
                                        head_outputs_fuse['gloss_probabilities'] + head_outputs_left['gloss_probabilities']+
                                        head_outputs_right['gloss_probabilities']).log(),
                            'rgb_gloss_logits': head_outputs_rgb['gloss_logits'],
                            'keypoint_gloss_logits': head_outputs_keypoint['gloss_logits'],
                            'fuse_gloss_logits': head_outputs_fuse['gloss_logits'],
                            'left_gloss_logits': head_outputs_left['gloss_logits'],
                            'left_gloss_probabilities_log': head_outputs_left['gloss_probabilities_log'],
                            'right_gloss_logits': head_outputs_right['gloss_logits'],
                            'right_gloss_probabilities_log': head_outputs_right['gloss_probabilities_log'],
                            'rgb_gloss_probabilities_log': head_outputs_rgb['gloss_probabilities_log'],
                            'keypoint_gloss_probabilities_log': head_outputs_keypoint['gloss_probabilities_log'],
                            'fuse_gloss_probabilities_log': head_outputs_fuse['gloss_probabilities_log'],}
            head_outputs['ensemble_last_gloss_probabilities_log'] = head_outputs[
                'ensemble_last_gloss_logits'].log_softmax(2)
            head_outputs['ensemble_last_gloss_probabilities'] = head_outputs['ensemble_last_gloss_logits'].softmax(2)
        elif self.input_type == 'feature':
            head_outputs_rgb = self.rgb_visual_head(x=src_input['feature'].cuda(), mask=src_input['mask'].cuda())
            keypoint_outputs, left_output, right_output = self.visual_backbone_keypoint(src_input)
            head_outputs_keypoint = self.keypoint_visual_head(
                x=keypoint_outputs,
                mask=src_input['mask'].cuda(),
                valid_len_in=src_input['new_src_lengths'])
            head_outputs_left = self.left_visual_head(
                x=left_output,
                mask=src_input['mask'].cuda(),
                valid_len_in=src_input['new_src_lengths'].cuda())
            head_outputs_right = self.right_visual_head(
                x=right_output,
                mask=src_input['mask'].cuda(),
                valid_len_in=src_input['new_src_lengths'].cuda())
            fuse = torch.cat([keypoint_outputs, src_input['feature'].cuda()], dim=-1)
            head_outputs_fuse = self.visual_head_fuse(
                x=fuse,
                mask=src_input['mask'].cuda(),
                valid_len_in=src_input['new_src_lengths'])
            head_outputs['fuse_gloss_probabilities'] = head_outputs_fuse['gloss_probabilities']
            head_outputs['fuse_gloss_probabilities_log'] = head_outputs_fuse['gloss_probabilities_log']
            head_outputs['fuse_gloss_logits'] = head_outputs_fuse['gloss_logits']
            head_outputs['fuse_gloss_feature'] = head_outputs_fuse['gloss_feature']
            head_outputs = {
                'ensemble_last_gloss_logits': (
                        head_outputs_rgb['gloss_probabilities'] + head_outputs_keypoint['gloss_probabilities'] +
                        head_outputs_fuse['gloss_probabilities'] + head_outputs_left['gloss_probabilities'] +
                        head_outputs_right['gloss_probabilities']).log(),
                'rgb_gloss_logits': head_outputs_rgb['gloss_logits'],
                'keypoint_gloss_logits': head_outputs_keypoint['gloss_logits'],
                'fuse_gloss_logits': head_outputs_fuse['gloss_logits'],
                'left_gloss_logits': head_outputs_left['gloss_logits'],
                'left_gloss_probabilities_log': head_outputs_left['gloss_probabilities_log'],
                'right_gloss_logits': head_outputs_right['gloss_logits'],
                'right_gloss_probabilities_log': head_outputs_right['gloss_probabilities_log'],
                'rgb_gloss_probabilities_log': head_outputs_rgb['gloss_probabilities_log'],
                'keypoint_gloss_probabilities_log': head_outputs_keypoint['gloss_probabilities_log'],
                'fuse_gloss_probabilities_log': head_outputs_fuse['gloss_probabilities_log'], }
            head_outputs['ensemble_last_gloss_probabilities_log'] = head_outputs[
                'ensemble_last_gloss_logits'].log_softmax(2)
            head_outputs['ensemble_last_gloss_probabilities'] = head_outputs['ensemble_last_gloss_logits'].softmax(2)
        else:
            raise ValueError
        outputs = {**head_outputs,
                   'input_lengths':src_input['new_src_lengths']}

        # TODO
        if self.input_type == 'video':
            for k in ['keypoint', 'rgb', 'fuse', 'left', 'right']:
                outputs[f'recognition_loss_{k}'] = self.compute_recognition_loss(
                    gloss_labels=src_input['gloss_input']['gloss_labels'].cuda(),
                    gloss_lengths=src_input['gloss_input']['gls_lengths'].cuda(),
                    gloss_probabilities_log=head_outputs[f'{k}_gloss_probabilities_log'],
                    input_lengths=src_input['new_src_lengths'].cuda())
            outputs['recognition_loss'] = outputs['recognition_loss_keypoint'] + outputs['recognition_loss_rgb'] + outputs['recognition_loss_fuse'] + \
                                            outputs['recognition_loss_left'] + outputs['recognition_loss_right']
            outputs['gloss_feature'] = head_outputs_fuse['gloss_feature']
            loss_func = torch.nn.KLDivLoss(reduction="batchmean")
            for student in ['keypoint', 'rgb', 'fuse','left','right']:
                teacher_prob = outputs['ensemble_last_gloss_probabilities']
                teacher_prob = teacher_prob.detach()
                student_log_prob = outputs[f'{student}_gloss_probabilities_log']
                outputs[f'{student}_distill_loss'] = loss_func(input=student_log_prob, target=teacher_prob)
                outputs['recognition_loss'] += outputs[f'{student}_distill_loss']
        elif self.input_type == 'keypoint':
            for k in ['left', 'right', 'fuse']:
                outputs[f'recognition_loss_{k}'] = self.compute_recognition_loss(
                    gloss_labels=src_input['gloss_input']['gloss_labels'].cuda(),
                    gloss_lengths=src_input['gloss_input']['gls_lengths'].cuda(),
                    gloss_probabilities_log=head_outputs[f'{k}_gloss_probabilities_log'],
                    input_lengths=src_input['new_src_lengths'].cuda())
            outputs['recognition_loss'] = outputs['recognition_loss_left'] + outputs['recognition_loss_right'] + outputs['recognition_loss_fuse']
            loss_func = torch.nn.KLDivLoss(reduction="batchmean")
            for student in ['left', 'right', 'fuse']:
                teacher_prob = outputs['ensemble_last_gloss_probabilities']
                teacher_prob = teacher_prob.detach()
                student_log_prob = outputs[f'{student}_gloss_probabilities_log']
                outputs[f'{student}_distill_loss'] = loss_func(input=student_log_prob, target=teacher_prob)
                outputs['recognition_loss'] += outputs[f'{student}_distill_loss']
        elif self.input_type == 'feature':
            for k in ['keypoint', 'rgb', 'fuse', 'left', 'right']:
                outputs[f'recognition_loss_{k}'] = self.compute_recognition_loss(
                    gloss_labels=src_input['gloss_input']['gloss_labels'].cuda(),
                    gloss_lengths=src_input['gloss_input']['gls_lengths'].cuda(),
                    gloss_probabilities_log=head_outputs[f'{k}_gloss_probabilities_log'],
                    input_lengths=src_input['new_src_lengths'].cuda())
            outputs['recognition_loss'] = outputs['recognition_loss_keypoint'] + outputs['recognition_loss_rgb'] + \
                                          outputs['recognition_loss_fuse'] + \
                                          outputs['recognition_loss_left'] + outputs['recognition_loss_right']
            outputs['gloss_feature'] = head_outputs_fuse['gloss_feature']
            loss_func = torch.nn.KLDivLoss(reduction="batchmean")
            for student in ['keypoint', 'rgb', 'fuse', 'left', 'right']:
                teacher_prob = outputs['ensemble_last_gloss_probabilities']
                teacher_prob = teacher_prob.detach()
                student_log_prob = outputs[f'{student}_gloss_probabilities_log']
                outputs[f'{student}_distill_loss'] = loss_func(input=student_log_prob, target=teacher_prob)
                outputs['recognition_loss'] += outputs[f'{student}_distill_loss']

        return outputs


if __name__ == '__main__':
    config = [[64, 64, 16, 1], [64, 64, 16, 1],
              [64, 128, 32, 2], [128, 128, 32, 1],
              [128, 256, 64, 2], [256, 256, 64, 1],
              [256, 256, 64, 1], [256, 256, 64, 1],
              ]
    net = DSTANet(config=config)  # .cuda()
    ske = torch.rand([2, 2, 84, 52])  # .cuda()
    print(net(ske).shape)