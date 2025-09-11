# -*- coding: utf-8 -*-
# @Time : 2023/3/1 下午4:38
# @Author : Lingo
# @File : register_conf_tokenizer.py
# -*- coding: utf-8 -*-
# @Time : 2022/6/2 上午3:06
# @Author : Lingo
# @File : register.py
import os
import sys

root_path = os.path.abspath(__file__)
sys.path.append('/'.join(root_path.split('/')[:-2]))
from tokenizers.models import WordLevel
from conica.configuration_conica import ConicaConfig

import warnings
import argparse

from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast

from tokenizers import Tokenizer
from tokenizers import Regex

from tokenizers import normalizers, pre_tokenizers,decoders
import pandas as pd
from tokenizers.processors import TemplateProcessing
from transformers import AutoModel, AutoTokenizer
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from transformers import CLIPModel

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_path', type=str, default="/home/stormai/userfile/szw/CONICA-flickr/data/flickr30k.csv",
                        help="path of karpathy split json")
    parser.add_argument('-output_file', type=str, default="/home/stormai/userfile/szw/CONICA-flickr/conica-clip",
                        help="path of output_file")
    parser.add_argument('-min_frequency', type=int, default=5,
                        help="The minimum frequency a word should appear in corpus")
    parser.add_argument('-pretrained_tokenizer_name', type=str, default="/home/stormai/userfile/szw/CONICA-flickr/model",
                        help="the name of tokenizer, e.g, 'openai/clip-vit-large-patch14' ")
    parser.add_argument("-d_vision", type=int, default=1024)
    parser.add_argument("-d_model", type=int, default=512)
    parser.add_argument("-d_align", type=int, default=128)
    parser.add_argument("-n_head", type=int, default=8)
    parser.add_argument("-vision_encoder_layers", type=int, default=6)
    parser.add_argument("-text_encoder_layers", type=int, default=3)
    parser.add_argument("-multimodal_decoder_layers", type=int, default=3)
    args = parser.parse_args()

    if args.pretrained_tokenizer_name is not None:

        clip_path='/home/stormai/userfile/szw/CONICA-flickr/model'
        tokenizer = AutoTokenizer.from_pretrained(clip_path)
        if "openai/clip" in args.pretrained_tokenizer_name:
            tokenizer.add_special_tokens({"pad_token": "!"})
        config = ConicaConfig(d_model=args.d_model,
                              d_align=args.d_align,
                              d_vision=args.d_vision,
                              n_head=args.n_head,
                              ffn_ratio=4,
                              vision_encoder_layers=args.vision_encoder_layers,
                              text_encoder_layers=args.text_encoder_layers,
                              multimodal_decoder_layers=args.multimodal_decoder_layers,
                              bos_token_id=tokenizer.bos_token_id,
                              pad_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id,
                              vocab_size=len(tokenizer))
        tokenizer.save_pretrained(args.output_file)
        config.save_pretrained(args.output_file)
    else:

        # train a WordLevel tokenizer from scratch on MSCOCO corpus

        dataset_csv = pd.read_csv(args.dataset_path)
        dataset_csv = dataset_csv[dataset_csv.split.isin(["train", "restval"])]
        with open("temp_sentences.raw", "w") as file:

            for i in range(len(dataset_csv)):
                item = dataset_csv.iloc[i]
                gts_caps = eval(item['sentences'])

                for sent in gts_caps:
                    file.write(sent + "\n")
        file.close()

        # initialize tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        # initialize normalizer
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFC(),
            normalizers.Replace(Regex("\\s+"), " "),
            normalizers.Strip(),
            normalizers.Lowercase()])
        # initialize trainer

        trainer = WordLevelTrainer(min_frequency=args.min_frequency,
                                   special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex("'s|'t|'re|'ve|'m|'ll|'d|\d+|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"),
                                     behavior="removed", invert=True),
                # pre_tokenizers.ByteLevel(add_prefix_space=False),

                pre_tokenizers.Split(Regex("[\-:`\",.;!?\[\]()]"), behavior="removed"),

            ]
        )
        tokenizer.train(["temp_sentences.raw"], trainer)
        tokenizer.add_special_tokens(["<cls>"])
        # tokenizer.decoder = decoders.ByteLevel()
        os.remove("temp_sentences.raw")
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <cls>",
            special_tokens=[
                ("<bos>", tokenizer.token_to_id("<bos>")),
                ("<cls>", tokenizer.token_to_id("<cls>")),
            ],
        )

        config = ConicaConfig(d_model=args.d_model,
                              d_align=args.d_align,
                              d_vision=args.d_vision,
                              n_head=args.n_head,
                              ffn_ratio=4,
                              vision_encoder_layers=args.vision_encoder_layers,
                              text_encoder_layers=args.text_encoder_layers,
                              multimodal_decoder_layers=args.multimodal_decoder_layers,
                              bos_token_id=tokenizer.token_to_id("<bos>"),
                              pad_token_id=tokenizer.token_to_id("<pad>"),
                              eos_token_id=tokenizer.token_to_id("<eos>"),
                              cls_token_id=tokenizer.token_to_id("<cls>"),
                              vocab_size=tokenizer.get_vocab_size())

        config.save_pretrained(args.output_file)
        tokenizer.save('temp_tokenizer.json')

        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="temp_tokenizer.json")

        fast_tokenizer.add_special_tokens(
            {'pad_token': '<pad>', 'unk_token': '<unk>', 'bos_token': "<bos>", "eos_token": "<eos>",
             "cls_token": "<cls>"})
        fast_tokenizer.save_pretrained(args.output_file)
        os.remove("temp_tokenizer.json")

