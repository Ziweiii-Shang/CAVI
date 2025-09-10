# -*- coding: utf-8 -*-
# @Time : 2023/2/27 下午3:46
# @Author : Lingo
# @File : dataset.py
from torch.utils.data import Dataset
import numpy as np
import random
import os.path as op
import pandas as pd


class ImageCaptionDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 df: pd.DataFrame,
                 seq_per_img: int = 5,
                 max_features_len=None,
                 training=True,
                 add_mean_cls=True
                 ):
        self.file_path = file_path
        self.df = df
        self.training = training
        self.add_mean_cls = add_mean_cls
        self.seq_per_img = seq_per_img
        self.max_features_len = max_features_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        cocoid = item["cocoid"]
        gts_caps = eval(item['sentences'])
        img_feats = np.load(op.join(self.file_path, str(cocoid) + ".npz"))
        num_caps = len(gts_caps)
        features = img_feats["features"]
        if self.add_mean_cls:
            cls_features = features.mean(0)[np.newaxis, :]
        else:
            cls_features = features[:1]
            features = features[1:]
        sample_caps = ["" for _ in range(self.seq_per_img)]
        if self.training:
            np.random.shuffle(features)
            if num_caps < self.seq_per_img:
                sample_caps[:num_caps] = gts_caps
                for _ in range(self.seq_per_img - num_caps):
                    sample_caps[num_caps + _] = gts_caps[random.randint(0, num_caps - 1)]
            else:
                _ = random.randint(0, num_caps - self.seq_per_img)
                sample_caps = gts_caps[_:_ + self.seq_per_img]

        if self.max_features_len:
            features = features[:self.max_features_len, ]
        features = np.concatenate((cls_features, features), axis=0)

        return features, sample_caps, gts_caps
