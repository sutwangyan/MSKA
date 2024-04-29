import torch

from Tokenizer import GlossTokenizer_S2G
from recognition import Recognition
from translation import TranslationNetwork
from vl_mapper import VLMapper


class SignLanguageModel(torch.nn.Module):
    def __init__(self, cfg, args):
        super().__init__()
        self.args = args
        self.task, self.device = cfg['task'], cfg['device']
        model_cfg = cfg['model']
        self.frozen_modules = []
        if self.task == 'S2G':
            self.text_tokenizer = None
            self.recognition_network = Recognition(cfg=model_cfg['RecognitionNetwork'],args=self.args)
            self.gloss_tokenizer = self.recognition_network.gloss_tokenizer
        elif self.task == 'S2T':
            self.recognition_weight = model_cfg.get('recognition_weight', 1)
            self.translation_weight = model_cfg.get('translation_weight', 1)
            self.recognition_network = Recognition(cfg=model_cfg['RecognitionNetwork'], args=self.args)
            self.translation_network = TranslationNetwork(cfg=model_cfg['TranslationNetwork'])
            self.gloss_tokenizer = self.recognition_network.gloss_tokenizer
            self.text_tokenizer = self.translation_network.text_tokenizer
            if model_cfg['VLMapper'].get('type','projection') == 'projection':
                if 'in_features' in model_cfg['VLMapper']:
                    in_features = model_cfg['VLMapper'].pop('in_features')
                else:
                    in_features = model_cfg['RecognitionNetwork']['visual_head']['hidden_size']
            else:
                in_features = len(self.gloss_tokenizer)
            self.vl_mapper = VLMapper(cfg=model_cfg['VLMapper'],
                                      in_features=in_features,
                                      out_features=self.translation_network.input_dim,
                                      gloss_id2str=self.gloss_tokenizer.id2gloss,
                                      gls2embed=getattr(self.translation_network, 'gls2embed', None))

    def forward(self, src_input, **kwargs):
        if self.task == "S2G":
            recognition_outputs = self.recognition_network(src_input)
            model_outputs = {**recognition_outputs}
            model_outputs['total_loss'] = recognition_outputs['recognition_loss']
        else:
            recognition_outputs = self.recognition_network(src_input)
            mapped_feature = self.vl_mapper(visual_outputs=recognition_outputs)
            translation_inputs = {
                **src_input['translation_inputs'],
                'input_feature': mapped_feature,
                'input_lengths': recognition_outputs['input_lengths']}
            translation_outputs = self.translation_network(**translation_inputs)
            model_outputs = {**translation_outputs, **recognition_outputs}
            model_outputs['transformer_inputs'] = model_outputs['transformer_inputs']  # for latter use of decoding
            model_outputs['total_loss'] = model_outputs['recognition_loss'] + model_outputs['translation_loss']

        return model_outputs


    def generate_txt(self, transformer_inputs=None, generate_cfg={}, **kwargs):
            model_outputs = self.translation_network.generate(**transformer_inputs, **generate_cfg)
            return model_outputs

    def predict_gloss_from_logits(self, gloss_logits, beam_size, input_lengths):
        return self.recognition_network.decode(
            gloss_logits=gloss_logits,
            beam_size=beam_size,
            input_lengths=input_lengths)