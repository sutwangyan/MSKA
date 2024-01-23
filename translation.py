import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig
from utils import XentLoss
from Tokenizer import GlossTokenizer_G2T, TextTokenizer
import pickle, math


class TranslationNetwork(torch.nn.Module):
    def __init__(self, input_type='feature', cfg=None, task='S2T') -> None:
        super().__init__()
        self.task = task  # 'S2T'
        self.input_type = input_type  # 'feature'
        assert self.input_type in ['gloss', 'feature']
        self.text_tokenizer = TextTokenizer({'pretrained_model_name_or_path': 'pretrained_models/mBart_de',
                                             'pruneids_file': 'pretrained_models/mBart_de/map_ids.pkl',
                                             'tgt_lang': 'de_DE'})  # pretrained_model_name_or_path: pretrained_models/mBart_de
        # pruneids_file: pretrained_models/mBart_de/map_ids.pkl
        # tgt_lang de_DE

        # if 'pretrained_model_name_or_path' in cfg:   # True
        self.model = MBartForConditionalGeneration.from_pretrained(
            'pretrained_models/mBart_de',  # pretrained_models/mBart_de
            **{'attention_dropout': 0.1, 'dropout': 0.3}  # TODO check   # attention_dropout 0.1  dropout 0.3
        )

        self.translation_loss_fun = XentLoss(
            pad_index=self.text_tokenizer.pad_index,
            smoothing=0.2)   # smoothing 0.2
        self.input_dim = self.model.config.d_model
        self.input_embed_scale = math.sqrt(self.model.config.d_model)

        # if self.task in ['S2T', 'G2T'] and 'pretrained_model_name_or_path' in cfg:
            # in both S2T or G2T, we need gloss_tokenizer and gloss_embedding
        # TODO data/phoenix-2014t/gloss2ids_old.pkl
        self.gloss_tokenizer = GlossTokenizer_G2T({'gloss2id_file':'pretrained_models/mBart_de/gloss2ids.pkl', 'src_lang':'de_DGS'})  # {gloss2id_file:pretrained_models/mBart_de/gloss2ids.pkl, src_lang:de_DGS}
        self.gloss_embedding = self.build_gloss_embedding(**{'freeze':False,'gloss2embed_file':'E:/KP-VLP/pretrained_models/mBart_de/gloss_embeddings.bin'})  # {freeze:false,gloss2embed_file:pretrained_models/mBart_de/gloss_embedding.bin}
        # debug
        self.gls_eos = 'gls'  # gls or txt
        # if 'load_ckpt' in cfg:
        # TODO experiments/outputs/SingleStream/phoenix-2014t_g2t/ckpts/best.ckpt  same
        self.load_from_pretrained_ckpt('pretrained_models/phoenix-2014t_g2t/ckpts/best.ckpt')   # 'experiments/outputs/SingleStream/phoenix-2014t_g2t/ckpts/best.ckpt'

    def load_from_pretrained_ckpt(self, pretrained_ckpt):

        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
        load_dict = {}
        for k, v in checkpoint.items():
            if 'translation_network' in k:
                load_dict[k.replace('translation_network.', '')] = v
        self.load_state_dict(load_dict)
        # logger.info('Load Translation network from pretrained ckpt {}'.format(pretrained_ckpt))

    def build_gloss_embedding(self, gloss2embed_file, from_scratch=False, freeze=False):
        gloss_embedding = torch.nn.Embedding(
            num_embeddings=len(self.gloss_tokenizer.id2gloss),
            embedding_dim=self.model.config.d_model,
            padding_idx=self.gloss_tokenizer.gloss2id['<pad>'])
        if from_scratch:
            # self.logger.info('Train Gloss Embedding from scratch')
            assert freeze == False
        else:
            gls2embed = torch.load(gloss2embed_file)
            self.gls2embed = gls2embed
            with torch.no_grad():
                for id_, gls in self.gloss_tokenizer.id2gloss.items():
                    if gls in gls2embed:
                        assert gls in gls2embed, gls
                        gloss_embedding.weight[id_, :] = gls2embed[gls]
                    # else:
                    #     self.logger.info('{} not in gls2embed train from scratch'.format(gls))

        return gloss_embedding

    def prepare_gloss_inputs(self, input_ids):
        input_emb = self.gloss_embedding(input_ids) * self.input_embed_scale
        return input_emb

    def prepare_feature_inputs(self, input_feature, input_lengths, gloss_embedding=None, gloss_lengths=None):
        if self.task == 'S2T_glsfree':
            suffix_len = 0
            suffix_embedding = None
        else:
            if self.gls_eos == 'gls':
                suffix_embedding = [self.gloss_embedding.weight[self.gloss_tokenizer.convert_tokens_to_ids('</s>'), :]]
            else:
                suffix_embedding = [self.model.model.shared.weight[self.text_tokenizer.eos_index, :]]
            if self.task in ['S2T', 'G2T'] and self.gloss_embedding:
                if self.gls_eos == 'gls':
                    src_lang_code_embedding = self.gloss_embedding.weight[ \
                                              self.gloss_tokenizer.convert_tokens_to_ids(self.gloss_tokenizer.src_lang),
                                              :]  # to-debug
                else:
                    src_lang_id = self.text_tokenizer.pruneids[
                        30]  # self.text_tokenizer.pruneids[self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)]
                    assert src_lang_id == 31
                    src_lang_code_embedding = self.model.model.shared.weight[src_lang_id, :]
                suffix_embedding.append(src_lang_code_embedding)
            suffix_len = len(suffix_embedding)
            suffix_embedding = torch.stack(suffix_embedding, dim=0)

        max_length = torch.max(input_lengths) + suffix_len
        inputs_embeds = []
        attention_mask = torch.zeros([input_feature.shape[0], max_length], dtype=torch.long,
                                     device=input_feature.device)

        for ii, feature in enumerate(input_feature):
            valid_len = input_lengths[ii]
            if 'gloss+feature' in self.input_type:
                valid_feature = torch.cat(
                    [gloss_embedding[ii, :gloss_lengths[ii], :], feature[:valid_len - gloss_lengths[ii], :]],
                    dim=0)
            else:
                valid_feature = feature[:valid_len, :]  # t,D
            if suffix_embedding != None:
                feature_w_suffix = torch.cat([valid_feature, suffix_embedding], dim=0)  # t+2, D
            else:
                feature_w_suffix = valid_feature
            if feature_w_suffix.shape[0] < max_length:
                pad_len = max_length - feature_w_suffix.shape[0]
                padding = torch.zeros([pad_len, feature_w_suffix.shape[1]],
                                      dtype=feature_w_suffix.dtype, device=feature_w_suffix.device)
                padded_feature_w_suffix = torch.cat([feature_w_suffix, padding], dim=0)  # t+2+pl,D
                inputs_embeds.append(padded_feature_w_suffix)
            else:
                inputs_embeds.append(feature_w_suffix)
            attention_mask[ii, :valid_len + suffix_len] = 1
        transformer_inputs = {
            'inputs_embeds': torch.stack(inputs_embeds, dim=0) * self.input_embed_scale,  # B,T,D
            'attention_mask': attention_mask  # attention_mask
        }
        return transformer_inputs

    def forward(self, **kwargs):
        input_feature = kwargs.pop('input_feature')
        input_lengths = kwargs.pop('input_lengths')
        # quick fix
        kwargs.pop('gloss_ids', None)
        kwargs.pop('gloss_lengths', None)
        new_kwargs = self.prepare_feature_inputs(input_feature, input_lengths)
        kwargs = {**kwargs, **new_kwargs}
        kwargs = {key: value.to('cuda') for key, value in kwargs.items()}
        output_dict = self.model(**kwargs, return_dict=True)
        # print(output_dict.keys()) loss, logits, past_key_values, encoder_last_hidden_state
        log_prob = torch.nn.functional.log_softmax(output_dict['logits'], dim=-1)  # B, T, L
        batch_loss_sum = self.translation_loss_fun(log_probs=log_prob, targets=kwargs['labels'])
        output_dict['translation_loss'] = batch_loss_sum / log_prob.shape[0]

        output_dict['transformer_inputs'] = kwargs  # for later use (decoding)
        return output_dict

    def generate(self,
                 input_ids=None, attention_mask=None,  # decoder_input_ids,
                 inputs_embeds=None, input_lengths=None,
                 num_beams=4, max_length=100, length_penalty=1, **kwargs):
        assert attention_mask != None
        batch_size = attention_mask.shape[0]
        decoder_input_ids = torch.ones([batch_size, 1], dtype=torch.long,
                                       device=attention_mask.device) * self.text_tokenizer.sos_index
        assert inputs_embeds != None and attention_mask != None
        output_dict = self.model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,  # same with forward
            decoder_input_ids=decoder_input_ids,
            num_beams=num_beams, length_penalty=length_penalty, max_length=max_length,
            return_dict_in_generate=True)
        output_dict['decoded_sequences'] = self.text_tokenizer.batch_decode(output_dict['sequences'])
        return output_dict



