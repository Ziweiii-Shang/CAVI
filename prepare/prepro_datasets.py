# -*- coding: utf-8 -*-
# @Time : 2022/6/2 上午12:55
# @Author : Lingo
# @File : prepro_datasets.py
import sys,os

root_path = os.path.abspath(__file__)
sys.path.append('/'.join(root_path.split('/')[:-2]))
from tqdm import tqdm
import pandas as pd
import warnings
import argparse
from utils.ptbtokenizer import PTBTokenizer

punctuation = ["!!","??","???","-lrb-","-rrb-","-lsb-","-rsb-","-lcb-","-rcb-"]
def remove_punctuation(sentences):
    for i,sentence in enumerate(sentences):

        for p in punctuation:
            sentence = sentence.replace(p,"")
        sentences[i] = sentence
    return sentences
def json_to_csv(jsonfile, outputfile):
    header = ["cocoid", "imgid", "filepath", "filename", "sentences", "split"]

    df_list =[]
    ptbtokenizer = PTBTokenizer()
    raw_sentences = []

    for image in tqdm(jsonfile["images"]):
        sentences = [sentence["raw"] for sentence in
                              image["sentences"]]
        raw_sentences.append(sentences)
    all_sentences = ptbtokenizer.tokenize(raw_sentences)
    for image,sentences in tqdm(zip(jsonfile["images"],all_sentences)):
        imgid = image["imgid"]
        if "cocoid" in image:
            item = {"cocoid": image["cocoid"], "imgid": imgid, "filepath": image["filepath"], "filename": image["filename"],
                    "sentences": remove_punctuation(sentences) if image["split"] in ["train","restval"] else sentences, "split": image["split"]}
        else:
            id = int(image["filename"].split('.')[0].split('_')[-1])
            item = {"cocoid": id, "imgid": imgid, "filepath": "",
                    "filename": image["filename"],
                    "sentences": remove_punctuation(sentences) if image["split"] in ["train", "restval"] else sentences,
                    "split": image["split"]}

        df_list.append(item)

    df = pd.DataFrame(data=df_list,columns=header)

    df.to_csv(outputfile, index=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('-karpathy_split_json', type=str, default="/home/stormai/userfile/szw/caption_dataset/caption_datasets/dataset_flickr30k.json",
                        help="path of karpathy split json")
    parser.add_argument('-output_file', type=str, default="/home/stormai/userfile/szw/CONICA-flickr/data/flickr30k.csv",
                        help="path of output_file")
    args = parser.parse_args()
    karpathy_split_json = pd.read_json(args.karpathy_split_json)
    json_to_csv(karpathy_split_json, args.output_file)
