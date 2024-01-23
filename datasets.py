from ctypes import util

import torchvision
from cv2 import IMREAD_GRAYSCALE
import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
import math
from torchvision import transforms
from PIL import Image
import cv2
import os
import random
import numpy as np
import lmdb
import io
import time

from Tokenizer import GlossTokenizer_S2G, TextTokenizer
from augmentation import *
from loguru import logger

# global definition
from definition import *

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        if isinstance(Image, PIL.Image.Image):
            Image = np.asarray(Image, dtype=np.uint8)
        new_video_x = (Image - 127.5) / 128
        return new_video_x

class SomeOf(object):
    """
    Selects one augmentation from a list.
    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
    """

    def __init__(self, transforms1, transforms2):
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, clip):
        select = random.choice([0, 1, 2])
        if select == 0:
            return clip
        elif select == 1:
            if random.random() > 0.5:
                return self.transforms1(clip)
            else:
                return self.transforms2(clip)
        else:
            clip = self.transforms1(clip)
            clip = self.transforms2(clip)
            return clip

class S2T_Dataset(Dataset.Dataset):
    def __init__(self, path, tokenizer, config, args, phase, training_refurbish=False):
        self.config = config
        self.args = args
        self.training_refurbish = training_refurbish
        self.clip_len = 400
        if phase == 'train':
            self.tmin, self.tmax = 0.5, 1.5
        else:
            self.tmin, self.tmax = 1, 1
        self.transform_cfg = config['data']['transform_cfg']
        self.raw_data = utils.load_dataset_file(path)
        self.tokenizer = tokenizer
        self.img_path = config['data']['img_path']
        self.phase = phase
        self.max_length = config['data']['max_length']
        self.list = [key for key, value in self.raw_data.items()]

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]
        gloss = sample['gloss']
        length = sample['num_frames']
        text = sample['text']
        keypoint = sample['keypoint'].permute(2, 0, 1)
        name_sample = sample['name']
        image_list = None
        if 'video' in self.config['data']['input_streams']:
            folder_path = self.config['data']['img_path'] + name_sample
            image_list = [Image.open(os.path.join(folder_path, filename)).convert('RGB')
                          for filename in os.listdir(folder_path)]
        feature = None
        if 'feature' in self.config['data']['input_streams']:
            feature = sample['feature']
        return name_sample, keypoint, gloss, text, image_list, length

    def pil_list_to_tensor(self,pil_list, int2float=True):
        func = torchvision.transforms.PILToTensor()
        tensors = [func(pil_img) for pil_img in pil_list]
        tensors = torch.stack(tensors, dim=0)
        if int2float:
            tensors = tensors / 255
        return tensors  # (T,C,H,W)

    def get_seq_frames(self, num_frames):
        """
        Modified from https://github.com/facebookresearch/SlowFast/blob/64abcc90ccfdcbb11cf91d6e525bed60e92a8796/slowfast/datasets/ssv2.py#L159
        Given the video index, return the list of sampled frame indexes.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        seg_size = float(num_frames - 1) / self.clip_len
        seq = []
        for i in range(self.clip_len):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.phase == 'train':
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        return np.array(seq)

    def apply_spatial_ops(self, x, spatial_ops_func):
        B, T, C_, H, W = x.shape
        x = x.view(-1, C_, H, W)
        chunks = torch.split(x, 16, dim=0)
        transformed_x = []
        for chunk in chunks:
            transformed_x.append(spatial_ops_func(chunk))
        _, C_, H_o, W_o = transformed_x[-1].shape
        transformed_x = torch.cat(transformed_x, dim=0)
        transformed_x = transformed_x.view(B, T, C_, H_o, W_o)
        return transformed_x

    def augment_preprocess_inputs(self, is_train, sgn_videos=None, sgn_heatmaps=None):
        rgb_h, rgb_w = self.transform_cfg.get('img_size',224), self.transform_cfg.get('img_size',224)
        if is_train == 'train':
            if sgn_videos!=None:
                rgb_h0, rgb_w0 = sgn_videos.shape[-2],sgn_videos.shape[-1]
                if self.transform_cfg.get('color_jitter', False) and random.random()<0.3:
                    color_jitter_op = torchvision.transforms.ColorJitter(0.4,0.4,0.4,0.1)
                    sgn_videos = color_jitter_op(sgn_videos)
                i,j,h,w = torchvision.transforms.RandomResizedCrop.get_params(
                    img=sgn_videos,
                    scale=(self.transform_cfg.get('bottom_area',0.2), 1.0),
                    ratio=(self.transform_cfg.get('aspect_ratio_min',3./4),
                        self.transform_cfg.get('aspect_ratio_max',4./3)))
                sgn_videos = self.apply_spatial_ops(
                    sgn_videos,
                    spatial_ops_func=lambda x:torchvision.transforms.functional.resized_crop(
                        x, i, j, h, w, [rgb_h, rgb_w]))
            # TODO check x and y right
                sgn_heatmaps[:, 1, :, :] = h + i - sgn_heatmaps[:, 1, :, :]
                sgn_heatmaps[:, 0, :, :] = sgn_heatmaps[:, 0, :, :] - j
                x_ratio, y_ratio = rgb_w/w, rgb_h/h
                sgn_heatmaps[:, 0, :, :] = sgn_heatmaps[:, 0, :, :] * x_ratio
                sgn_heatmaps[:, 1, :, :] = sgn_heatmaps[:, 1, :, :] * y_ratio
                sgn_heatmaps[:, 0, :, :] = sgn_heatmaps[:, 0, :, :] / w
                sgn_heatmaps[:, 1, :, :] = sgn_heatmaps[:, 1, :, :] / h
                sgn_heatmaps = (sgn_heatmaps - 0.5) / 0.5
            else:
                # TODO keypoint augment
                sgn_heatmaps[:, 0, :, :] /= 210
                sgn_heatmaps[:, 1, :, :] = 260 - sgn_heatmaps[:, 1, :, :]
                sgn_heatmaps[:, 1, :, :] /= 260
                sgn_heatmaps[:,:2,:,:] = (sgn_heatmaps[:,:2,:,:] - 0.5) / 0.5
                sgn_heatmaps[:,:2,:,:] = self.random_move(sgn_heatmaps[:,:2,:,:].permute(0,2,3,1).numpy()).permute(0,3,1,2)

        else:
            if sgn_videos != None:
                spatial_ops = []
                if self.transform_cfg.get('center_crop',False)==True:
                    spatial_ops.append(torchvision.transforms.CenterCrop(
                        self.transform_cfg['center_crop_size']))
                spatial_ops.append(torchvision.transforms.Resize([rgb_h, rgb_w]))
                spatial_ops = torchvision.transforms.Compose(spatial_ops)
                sgn_videos = self.apply_spatial_ops(sgn_videos, spatial_ops)
                x_ratio, y_ratio = rgb_w / 210, rgb_h / 260
                sgn_heatmaps[:, 0, :, :] = sgn_heatmaps[:, 0, :, :] * x_ratio
                sgn_heatmaps[:, 1, :, :] = sgn_heatmaps[:, 1, :, :] * y_ratio
                sgn_heatmaps[:, 0, :, :] = sgn_heatmaps[:, 0, :, :] / 224
                sgn_heatmaps[:, 1, :, :] = sgn_heatmaps[:, 1, :, :] / 224
                sgn_heatmaps = (sgn_heatmaps - 0.5) / 0.5
            else:
                sgn_heatmaps[:,0, :, :] /= 210
                sgn_heatmaps[:,1, :, :] = 260 - sgn_heatmaps[:,1, :, :]
                sgn_heatmaps[:,1, :, :] /= 260
                sgn_heatmaps[:,:2,:,:] = (sgn_heatmaps[:,:2,:,:] - 0.5) / 0.5
        # TODO video and keypoint use same normalization
        if sgn_videos != None:
            sgn_videos = sgn_videos[:,:,[2,1,0],:,:]
            sgn_videos = (sgn_videos-0.5) / 0.5
            sgn_videos = sgn_videos.permute(0,2,1,3,4).float()
        return sgn_videos, sgn_heatmaps


    def collate_fn(self, batch):
        tgt_batch,keypoint_batch,src_length_batch,name_batch,text_batch,img_batch = [],[],[],[],[],[]
        for name_sample, keypoint_sample, tgt_sample, text, img, length in batch:
            # TODO add get_selected_index return index and valid len
            index, valid_len = self.get_selected_index(length)
            if img is not None:
                img_batch.append([img[i] for i in index])
            keypoint_batch.append(torch.stack([keypoint_sample[:, i, :] for i in index],dim=1))
            src_length_batch.append(valid_len)
            name_batch.append(name_sample)
            tgt_batch.append(tgt_sample)
            text_batch.append(text)

        max_length = max(src_length_batch)
        padded_sgn_videos, padded_sgn_keypoints = [], []
        for video, len_ in zip(img_batch, src_length_batch):
            video = self.pil_list_to_tensor(video, int2float=True)
            if len_ < max_length:
                padding = video[-1, :, :, :].unsqueeze(0)
                padding = torch.tile(padding, [max_length - len_, 1, 1, 1])
                padded_video = torch.cat([video, padding], dim=0)
                padded_sgn_videos.append(padded_video)
            else:
                padded_sgn_videos.append(video)

        for keypoints, len_ in zip(keypoint_batch, src_length_batch):
            if len_ < max_length:
                padding = keypoints[:,-1,:].unsqueeze(1)
                padding = torch.tile(padding, [1,max_length-len_, 1])
                padded_keypoint = torch.cat([keypoints, padding], dim=1)
                padded_sgn_keypoints.append(padded_keypoint)
            else:
                padded_sgn_keypoints.append(keypoints)
        lengths = torch.tensor(src_length_batch, dtype=torch.long)
        videos = None
        if 'video' in self.config['data']['input_streams']:
            videos = torch.stack(padded_sgn_videos, dim=0)
        keypoints = torch.stack(padded_sgn_keypoints, dim=0)

        # TODO videos and keypoint augment
        videos, keypoints = self.augment_preprocess_inputs(self.phase, videos, keypoints)

        src_length_batch = torch.tensor(src_length_batch)
        new_src_lengths = (((src_length_batch - 1) / 2) + 1).long()
        new_src_lengths = (((new_src_lengths - 1) / 2) + 1).long()
        gloss_input = self.tokenizer(tgt_batch)
        max_len = max(new_src_lengths)
        mask = torch.zeros(new_src_lengths.shape[0], 1, max_len)
        for i in range(new_src_lengths.shape[0]):
            mask[i, :, :new_src_lengths[i]] = 1
        mask = mask.to(torch.bool)
        src_input = {}
        if True:
            self.text_tokenizer = TextTokenizer({'pretrained_model_name_or_path': 'pretrained_models/mBart_de',
                                                 'pruneids_file': 'pretrained_models/mBart_de/map_ids.pkl',
                                                 'tgt_lang': 'de_DE'})
            t = self.text_tokenizer(text_batch)
            src_input['translation_inputs'] = {**t}
            src_input['text'] = text_batch
        # keypoint_batch = torch.cat(keypoint_batch, 0)
        # feature = torch.cat(feature_batch, 0)
        src_input['name'] = name_batch
        src_input['keypoint'] = keypoints.to(torch.float32)
        if videos is not None:
            src_input['videos'] = videos
        src_input['gloss'] = tgt_batch
        src_input['mask'] = mask
        # src_input['feature'] = feature
        src_input['new_src_lengths'] = new_src_lengths
        src_input['gloss_input'] = gloss_input
        src_input['src_length'] = src_length_batch
        src_input['translation_inputs']['gloss_ids'] = gloss_input['gloss_labels']
        src_input['translation_inputs']['gloss_lengths'] = gloss_input['gls_lengths']
        # src_input['new_src_length_batch'] = new_src_lengths
        
        if self.training_refurbish:
            # masked_tgt = utils.NoiseInjecting(tgt_batch, self.args.noise_rate, noise_type=self.args.noise_type, random_shuffle=self.args.random_shuffle, is_train=(self.phase=='train'))
            # with self.tokenizer.as_target_tokenizer():
            #     masked_tgt_input = self.tokenizer(masked_tgt, return_tensors="pt", padding = True,  truncation=True)
            return src_input
        return src_input

    def rotate_points(self, points, angle):
        center = [0, 0]

        # 将坐标平移到原点
        points_centered = points - center

        # 构建旋转矩阵，注意方向
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                    [-np.sin(angle), np.cos(angle)]])

        # 进行坐标点旋转
        points_rotated = np.dot(points_centered, rotation_matrix.T)

        # 将坐标平移到原来的中心位置
        points_transformed = points_rotated + center

        return points_transformed

    def translate_points(self, points, translation):
        # 进行平移操作
        points_translated = points + translation

        return points_translated

    def scale_points(self, points, scale_factor):
        # 缩放坐标
        points_scaled = points * scale_factor

        return points_scaled

    def get_selected_index(self, vlen):
        if self.tmin == 1 and self.tmax == 1:
            if vlen <= self.clip_len:
                frame_index = np.arange(vlen)
                valid_len = vlen
            else:
                sequence = np.arange(vlen)
                an = (vlen - self.clip_len) // 2
                en = vlen - self.clip_len - an
                frame_index = sequence[an: -en]
                valid_len = self.clip_len

            if (valid_len % 4) != 0:
                valid_len -= (valid_len % 4)
                frame_index = frame_index[:valid_len]

            assert len(frame_index) == valid_len, (frame_index, valid_len)
            return frame_index, valid_len
        min_len = min(int(self.tmin * vlen),self.clip_len)
        max_len = min(self.clip_len, int(self.tmax * vlen))
        selected_len = np.random.randint(min_len, max_len + 1)
        if (selected_len % 4) != 0:
            selected_len += (4 - (selected_len % 4))
        if selected_len <= vlen:
            selected_index = sorted(np.random.permutation(np.arange(vlen))[:selected_len])
        else:
            copied_index = np.random.randint(0, vlen, selected_len - vlen)
            selected_index = sorted(np.concatenate([np.arange(vlen), copied_index]))

        if selected_len <= self.clip_len:
            frame_index = selected_index
            valid_len = selected_len
        else:
            assert False, (vlen, selected_len, min_len, max_len)
        assert len(frame_index) == valid_len, (frame_index, valid_len)
        return frame_index, valid_len

    def random_move(self,data_numpy):
        # input: C,T,V,M
        # C, T, V, M = data_numpy.shape
        degrees = np.random.uniform(-15, 15)
        theta = np.radians(degrees)
        p = np.random.uniform(0,1)
        if p >= 0.5:
            data_numpy = self.rotate_points(data_numpy, theta)
        dx = np.random.uniform(-0.21, 0.21)
        dy = np.random.uniform(-0.26, 0.26)
        data_numpy = self.translate_points(data_numpy, [dx, dy])
        scale = np.random.uniform(0.8, 1.2)
        p = np.random.uniform(0, 1)
        if p >= 0.5:
            data_numpy = self.scale_points(data_numpy, scale)
        return torch.from_numpy(data_numpy)

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'






