# -*- coding: utf-8 -*-
# @Time : 2023/2/27 下午8:09
# @Author : Lingo
# @File : prepro_butd.py
import base64
import csv
import os.path as op
import sys
import argparse
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_path', type=str, default='/dataset/caption/mscoco/features/', help='path of dataset')
    parser.add_argument('-root_path', type=str, default='/home/smart-solution-server-001/Documents/dataset_slow/butd',
                        help='path of dataset')
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    infiles = ['karpathy_test_resnet101_faster_rcnn_genome.tsv',
               'karpathy_val_resnet101_faster_rcnn_genome.tsv',
               'karpathy_train_resnet101_faster_rcnn_genome.tsv.0',
               'karpathy_train_resnet101_faster_rcnn_genome.tsv.1']
    for infile in infiles:
        with open(op.join(args.root_path, infile), "r") as tsv:
            reader = csv.DictReader(tsv, delimiter="\t", fieldnames=FIELDNAMES)

            for item in tqdm(reader):
                item["num_boxes"] = int(item["num_boxes"])
                for field in ["boxes", "features"]:
                    item[field] = np.frombuffer(base64.decodebytes(item[field].encode("ascii")),
                                                dtype=np.float32).reshape(item["num_boxes"], -1)

                features = item["features"]
                np.savez_compressed(op.join(args.save_path, "butd", item["image_id"]),
                                    features=features)
