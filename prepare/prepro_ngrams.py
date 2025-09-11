# -*- coding: utf-8 -*-
# @Time : 2022/6/2 下午1:05
# @Author : Lingo
# @File : prepro_ngrams.py

import os
import sys

root_path = os.path.abspath(__file__)
sys.path.append('/'.join(root_path.split('/')[:-2]))
from utils.cider import CiderScorer
from transformers import AutoTokenizer

from tqdm import tqdm
from utils.ptbtokenizer import PTBTokenizer
from six.moves import cPickle
import json
import argparse
import pandas as pd


def get_doc_freq(refs):
    tmp = CiderScorer()
    for ref in refs:
        tmp.cook_append(None, ref)
    tmp.compute_doc_freq()
    return tmp.document_frequency

def build_dict(dataset, params, tokenizer):
    if params['split'] == 'all':
        pass
    elif params['split'] == 'train':
        dataset = dataset[dataset.split.isin(["train", "restval"])]
    else:
        dataset = dataset[dataset.split.isin([params['split']])]

    count_imgs = 0

    refs_words = []
    refs_idxs = []

    for i in tqdm(range(len(dataset))):
        item = dataset.iloc[i]
        sentences = eval(item['sentences'])
        ref_words = []
        ref_idxs = []
        for j, sentence in enumerate(sentences):
            encoded_results = tokenizer._encode_plus(sentence+" "+tokenizer.eos_token,
                                                     add_special_tokens=False)
            ref_idxs.append(
                ' '.join([str(ids) for ids in encoded_results.input_ids]))
            sentence = ' '.join(tokenizer.batch_decode(encoded_results.input_ids))
            ref_words.append(sentence)
            # ref_words.append(sentence)
        refs_words.append(ref_words)
        refs_idxs.append(ref_idxs)
        count_imgs += 1
    print('total imgs:', count_imgs)
    ngram_words = get_doc_freq(refs_words)
    ngram_idxs = get_doc_freq(refs_idxs)
    print('count_refs:', count_imgs)
    return ngram_words, ngram_idxs, count_imgs


def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    return cPickle.dump(obj, f)


def main(params, tokenizer):
    dataset = pd.read_csv(params['input_csv'])

    ngram_words, ngram_idxs, ref_len = build_dict(dataset, params, tokenizer)

    pickle_dump({'document_frequency': ngram_words, 'ref_len': ref_len}, open(params['output_pkl'] + '-words.p', 'wb'))
    pickle_dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, open(params['output_pkl'] + '-idxs.p', 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('-input_csv', default='/home/stormai/userfile/szw/CONICA-flickr/data/flickr30k.csv',
                        help='input json file to process into hdf5')
    parser.add_argument('-output_pkl', default='flickr30k-train', help='output pickle file')
    parser.add_argument('-split', default='train', help='test, val, train, all')
    parser.add_argument("-tokenizer", default="/home/stormai/userfile/szw/CONICA-flickr/conica-clip")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    params = vars(args)  # convert to ordinary dict
    main(params, tokenizer)
