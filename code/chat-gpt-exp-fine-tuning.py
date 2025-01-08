import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import json
from typing import List, Dict, Any, NewType
import numpy as np
import random 
from random import shuffle
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
from evaluate import load
from feature_conversion_methods import format_instance, formatting, label_mapping
from load_custom_dataset import inverse_label_mapping
from sklearn.metrics import classification_report

from metrics import evaluate
from nli_demo import evaluate_score
from load_custom_dataset import load_raw_data, load_format_data
from utils import sep_label_explanation
from Inference import inference

# from transformers.integrations import WandbCallback
import transformers

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

import torch
import datasets
from datasets import Dataset
from copy import deepcopy 

InputDataClass = NewType("InputDataClass", Any)
text2label = {'entailment':0, 'neutral':1, 'contradiction':2}
from training import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset_name", default='esnli', type=str, required=False, 
                       help="training dataset name")
    parser.add_argument("--target_dataset_name", default=None, type=str, required=True, 
                       help="dataset name")
    parser.add_argument("--data_sub", default= 0, type=int, required=False, 
                       help="subset dataset")
    parser.add_argument("--lr", default= 3e-5, type=float, required=False, 
                       help="learning rate")
    parser.add_argument("--data_split", default='test', type=str, required=False, 
                       help="dataset split: train, val, test")
    parser.add_argument("--model_class", default='t5', type=str, required=False, 
                       help="model base")
    parser.add_argument("--model_name", default='t5-large', type=str, required=False, 
                       help="model base")
    parser.add_argument("--n_shots", default= 8, type=int, required=False, 
                       help="number of shots per class")
    parser.add_argument("--sample_selection", default= 'random', type=str, required=False, 
                       help="sample selection method")
    parser.add_argument("--explanation_source", default= 'na', type=str, required=False, 
                       help="source of explanation") 
    parser.add_argument("--seed", default= 42, type=int, required=False, 
                       help="random seed for each experiment")
    parser.add_argument("--sample_seed", default= 0, type=int, required=False, 
                       help="sample selection seed")
    parser.add_argument("--test_bsz", default= 128, type=int, required=False, 
                       help="test batch size")
    parser.add_argument("--explanation_sep", default= ' "explanation: " ', type=str, required=False, 
                       help="separation string between prediction and explanation")
    parser.add_argument("--model_path", default= '../model/efever/accept-fastvotek/sub0/nt64/', type=str, required=False, 
                       help="path to save model")
    parser.add_argument("--result_path", default= '../result', type=str, required=False, 
                       help="path to save model")
    parser.add_argument("--num_epochs", default= 10, type=int, required=False, 
                       help="number of training epochs")
    
    args = parser.parse_args()

    set_seed(args.seed)
    set_other_seeds(args.seed)
    
    df_train=pd.read_json('../samples/'+args.target_dataset_name+'/gpt-4-1106/train_select.json',lines=True)
    df_train['label']=[text2label[row['label']] for _, row in df_train.iterrows()]
    data_train = datasets.Dataset.from_pandas(df_train)
    
    input_string, answer_string = zip(*list(map(lambda x: formatting(x, args.target_dataset_name), data_train)))
    data_train = data_train.add_column("input_string", input_string)
    data_train = data_train.add_column("answer_string", answer_string)
    
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, config=config)
    
    print('Data formatting...')
    data_splits = {'train': None, 'validation': None, 'test': None}

    data_splits['train'] = deepcopy(format_data(
        dataset=data_train, 
        task=args.target_dataset_name,
        tokenizer = tokenizer, 
        explanation_sep=args.explanation_sep, 
        io_format='standard'
    ))
    
    save_model_path = '/'.join(('../model',args.source_dataset_name,'gpt-4-1106/target-fine-tuning',args.target_dataset_name))

    train(
    save_model_path, 
    model=model, 
    tokenizer = tokenizer, 
    train_bsz=4, 
    random_seed=args.seed, 
    data=data_splits, 
    lr=args.lr, 
    num_epochs=args.num_epochs
        )
    
    ## Load test data
    data_test = load_format_data(args.target_dataset_name, 'test')
    result_path = '/'.join((args.result_path,args.target_dataset_name,args.source_dataset_name,'gpt-4-1106/target_fine-tuning'))
    
    labels, explanations, _ = inference(
        model=model, 
        tokenizer=tokenizer,
        seed=args.seed, 
        data=data_test, 
        test_bsz=args.test_bsz, 
        result_path=result_path,
        explanation_sep=args.explanation_sep
    )
    
    results, cm = evaluate(
        result_path,
        data_test,
        tokenizer,
        "test",
        task=args.target_dataset_name,
        labels=labels,
        explanations=explanations
    )
    
    df_data = data_test.to_pandas()
    exp_score = evaluate_score(result_path, df_data, args.target_dataset_name, 64, only_correct=True)
    results['accept_score'] = exp_score
    print(results)
    
if __name__ == "__main__":
    main()