# -*- coding: utf-8 -*-
# @Time : 2022/6/1 8:21
# @Author : Lingo
# @File : prepro_feats.py
# @Descripition : runs only on single GPU

from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import clip
from clip.clip import _transform
from clip.model import VisionTransformer, ModifiedResNet
import os.path as op
from PIL import Image
import timm
import warnings
import argparse
import os
import numpy as np
from tqdm import tqdm
from timm.models.vision_transformer import resize_pos_embed
from timm.models.swin_transformer import SwinTransformer


class ImageDataset(Dataset):
    def __init__(self, root_path, pre_process):
        self.root_path = root_path
        self.imgs = os.listdir(root_path)
        self.preprocess = pre_process

    def __len__(self):
        self.imgs.sort()
        return len(self.imgs)

    def __getitem__(self, idx):
        file_name = self.imgs[idx]
        if self.preprocess:
            return file_name, self.preprocess(Image.open(op.join(self.root_path, self.imgs[idx])))
        return file_name, Image.open(op.join(self.root_path, self.imgs[idx]))


class ImageFeatureProjection(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if model_name in clip.available_models():
            clip_model, _ = clip.load(model_name)
            clip_model.to(device='cpu', dtype=torch.float)
            self.visual = clip_model.visual
        elif model_name in timm.list_models():
            self.visual = timm.create_model(model_name, pretrained=True)
        else:
            raise NotImplementedError
        self.global_pool = None
        if isinstance(self.visual, ModifiedResNet):
            self.visual.attnpool = nn.Identity()
        elif isinstance(self.visual, VisionTransformer):
            self.visual.ln_post = nn.Identity()
            self.visual.proj = None
        elif isinstance(self.visual, timm.models.ResNet):
            self.visual.global_pool = nn.Identity()
            self.visual.fc = nn.Identity()
        elif isinstance(self.visual, timm.models.efficientnet.EfficientNet):
            self.visual.conv_head = nn.Identity()
            self.visual.global_pool, self.visual.classifier = nn.Identity(), nn.Identity()
        elif isinstance(self.visual, SwinTransformer):
            self.visual.avgpool = nn.Identity()
            self.visual.head = nn.Identity()
        else:
            raise NotImplementedError

    @torch.no_grad()
    def forward(self, x):
        if isinstance(self.visual,
                      (ModifiedResNet, timm.models.ResNet, timm.models.efficientnet.EfficientNet)):
            x = self.visual(x)
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(0, 2, 1)  # NCHW -> N(HW)C

        elif isinstance(self.visual, VisionTransformer):
            x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                       device=x.device), x],
                dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.visual.positional_embedding.to(x.dtype)
            x = self.visual.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

        elif isinstance(self.visual, SwinTransformer):
            x = self.visual.patch_embed(x)
            if self.visual.absolute_pos_embed is not None:
                x = x + self.visual.absolute_pos_embed
            x = self.visual.pos_drop(x)
            x = self.visual.layers(x)  # NLD
            x = x.transpose(1, 2)  # NLD


        else:
            raise NotImplementedError
        return x


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-input_resolution', type=int, default=288, help='')

    parser.add_argument('-model_name', type=str, default='RN50x4', help='')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-dataset', type=str, default='/home/stormai/userfile/szw/flickr/flickr30k/flickr30k-images',
                        help='path of dataset')
    parser.add_argument('-save_path', type=str, default='/home/stormai/userfile/szw/CONICA-flickr/features', help='path of dataset')
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.gpu else "cpu")
    prepocess = _transform(args.input_resolution)
    dataset_path = args.dataset
    save_path = args.save_path
    model_name = args.model_name
    proj = ImageFeatureProjection(model_name).to(device)
    proj.eval()
    dataset = ImageDataset(dataset_path, prepocess)
    if not op.exists(op.join(save_path, model_name)):
        os.makedirs(op.join(save_path, model_name))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for filenames, imgs in tqdm(dataloader):
            features = proj(imgs.to(device))
            for feature, filename in zip(features, filenames):
                id = int(filename.split('.')[0].split('_')[-1])
                np.savez_compressed(op.join(args.save_path, model_name, str(id)),
                                    features=feature.cpu().float().numpy(),
                                    )
