# -*- coding: utf-8 -*-
# @Time : 2023/4/5 下午6:14
# @Author : Lingo
# @File : train.py

import torch
from transformers.trainer import WEIGHTS_NAME
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from dataset import ImageCaptionDataset
from conica.modeling_conica import ConicaModelWithLMHead
from conica.configuration_conica import ConicaConfig
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed
from utils.trainer_conica import ConicaTrainer
import sys, os, logging
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import transformers
import json
from transformers.trainer_utils import get_last_checkpoint
from transformers.file_utils import add_start_docstrings

logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


@dataclass
class ConfigArguments:
    config_name: str = field(
        default='conica-base', metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


@dataclass
class DataArguments:
    feature_path: str = field(default="/home/stormai/userfile/szw/Conica-coco/data/features/ViT-L/14@336px",
                              metadata={"help": "Path of pre-extracted features"}
                              )

    # root: str = field(default="/dataset/caption/mscoco/images",
    #                   metadata={"help": "Path of dataset"}
    #                   )
    max_features_len: int = field(default=None,
                                  metadata={"help": "number of grid/regional features, 50 for BUTD, none for others"}
                                  )
    dataset_csv: str = field(default="/home/stormai/userfile/szw/Conica-coco/data/mscoco.csv",
                             metadata={"help": "Path of dataset csv"}
                             )
    add_mean_cls: bool = field(default=False,
                               metadata={
                                   "help": "add global pool as cls feature, True for BUTD and SWIN, False for ViT"})
    train_corpus_path: str = field(default="/home/stormai/userfile/szw/Conica-coco/data/coco-train-words.p",
                                   metadata={"help": "path of the training corpus to compute cider"})
    seq_per_img: int = field(default=5,
                             metadata={"help": "sample captions per image for training"})

    coco_eval_json: str = field(default="/home/stormai/userfile/szw/CONICA-flickr/coco-caption/coco-caption/annotations/captions_val2014.json",
                                metadata={"help": "coco json"})


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class SCCLTrainingArguments(TrainingArguments):
    scst: bool = field(default=False, metadata={"help": "if adapt self critical sequence training or not"})
    scst_num_sample_sequences: int = field(default=None, metadata={"help": "number of sample sequences in scst"})
    scst_baseline_type: str = field(default='avg_rest', metadata={"help": "the base line type of scst"}, )
    optim: str = field(default="adamw_torch", metadata={"help": "The optimizer to use."})
    init_tau: bool = field(default=False, metadata={"help": "re-init tau"})

if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = HfArgumentParser((ConfigArguments, DataArguments, SCCLTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        config_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        config_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                , )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = ConicaConfig.from_pretrained(config_args.config_name,
                                          cache_dir=config_args.config_name,
                                          )
    tokenizer = AutoTokenizer.from_pretrained(config_args.tokenizer_name if
                                              config_args.tokenizer_name else config_args.config_name,
                                              cache_dir=config_args.config_name)

    train_dataset, eval_dataset, test_dataset = None, None, None
    dataset_csv = pd.read_csv(data_args.dataset_csv)

    if data_args.feature_path is not None:
        train_dataset = ImageCaptionDataset(data_args.feature_path,
                                            dataset_csv[dataset_csv.split.isin(["train", "restval"])],
                                            training=True,
                                            add_mean_cls=data_args.add_mean_cls,
                                            seq_per_img=data_args.seq_per_img,
                                            max_features_len=data_args.max_features_len)
        eval_dataset = ImageCaptionDataset(data_args.feature_path,
                                           dataset_csv[dataset_csv.split.isin(["val"])],
                                           training=False,
                                           add_mean_cls=data_args.add_mean_cls,
                                           seq_per_img=data_args.seq_per_img)

        test_dataset = ImageCaptionDataset(data_args.feature_path,
                                           dataset_csv[dataset_csv.split.isin(["test"])],
                                           training=False,
                                           add_mean_cls=data_args.add_mean_cls,
                                           seq_per_img=data_args.seq_per_img)

    # if data_args.root is not None:
    #     train_dataset = ImageDataset(data_args.root,
    #                                  dataset_csv[dataset_csv.split.isin(["train", "restval"])],
    #                                  training=True,
    #                                  seq_per_img=data_args.seq_per_img,f
    #                                  transform=train_transform(336))
    #     eval_dataset = ImageDataset(data_args.root,
    #                                 dataset_csv[dataset_csv.split.isin(["val"])],
    #                                 t7974raining=False,
    #                                 seq_per_img=data_args.seq_per_im51359g,
    #                                 transform=valid_transform(336))
    #
    #     test_dataset = ImageDataset(data_args.root,51359
    #                                 dataset_csv[dataset_csv.split.isin(["test"])],
    #                                 training=False,
    #                                 seq_per_img=data_args.seq_per_img,
    #                                 transform=valid_transform(336))
    model = ConicaModelWithLMHead(config=config)
    trainer = ConicaTrainer(model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            tokenizer=tokenizer,
                            train_cached_cider=data_args.train_corpus_path,
                            scst=training_args.scst,
                            scst_num_sample_sequences=training_args.scst_num_sample_sequences,
                            scst_baseline_type=training_args.scst_baseline_type,
                            init_tau=training_args.init_tau
                            )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
    results = {}
    max_length = ()

    if training_args.do_predict:
        coco = COCO(data_args.coco_eval_json)
        logger.info("*** Predict ***")
        if not os.path.isfile(os.path.join(checkpoint, WEIGHTS_NAME)):
            raise ValueError(f"Can't find a valid checkpoint at {checkpoint}")

        logger.info(f"Loading model from {checkpoint}).")
        state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME), map_location="cpu")
        trainer.model.load_state_dict(state_dict, strict=True)

        print(trainer.model.tau)
        # release memory
        predict_results = trainer.predict(
            test_dataset, metric_key_prefix="predict", num_beams=5)

        max_predict_samples = (len(test_dataset))
        metrics = predict_results.metrics
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        sentences = tokenizer.batch_decode(
            predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        cocoids = [cocoid for cocoid in dataset_csv[dataset_csv.split.isin(["test"])].cocoid]

        if trainer.is_world_process_zero():
            results = []
            for cocoid, sentence in zip(cocoids, sentences):
                result = dict()
                result['image_id'] = int(cocoid)
                result['caption'] = sentence.strip()

                results.append(result)
            resultfile = 'result.json'
            with open(resultfile, 'w') as F:
                json.dump(results, F)
            cocoRes = coco.loadRes(resultfile)
            cocoEval = COCOEvalCap(coco, cocoRes)

            cocoEval.params['image_id'] = cocoRes.getImgIds()
            cocoEval.evaluate()
