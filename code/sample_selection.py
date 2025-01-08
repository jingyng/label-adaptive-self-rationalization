import argparse
import os
from random import shuffle
import pandas as pd
import pyarrow as pa
import numpy as np
import datasets
from tqdm import tqdm
import torch
from utils import format_example, calculate_sentence_transformer_embedding, fast_votek
from Inference import MyDataset
from feature_conversion_methods import formatting
from load_custom_dataset import load_raw_data
from torch.utils.data import Dataset, DataLoader
from nli_demo import get_scores
import transformers
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
)

label2text = {0:'entailment', 1:'neutral', 2:'contradiction'}

def random_select(data, seed, sample_size):
    data = data.shuffle(seed=seed)
    labels = data.features['label'].names
    label_subsets = []
    for label in labels:
        label_int = data.features['label'].str2int(label)
        train_examples = [sample for sample in data if sample['label'] == label_int ]
        label_subsets = label_subsets + train_examples[:sample_size]
    tmp_df = pd.DataFrame(label_subsets)
    pa_tab= pa.Table.from_pandas(tmp_df)
    data = datasets.Dataset(pa_tab)
    # data.to_csv(save_path + '/train_selet.csv')
    return data

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", default='averitec', type=str, required=False, 
                       help="source dataset")
    parser.add_argument("--sample_selection", default='random', type=str, required=False, 
                       help="sample selection method")
    parser.add_argument("--n_shots", default=16, type=int, required=False, 
                       help="number of samples per class")  
    parser.add_argument("--sample_seed", default=22, type=int, required=False, 
                       help="random seed for sampling")
    parser.add_argument("--gen_path", default='random', type=str, required=False, 
                       help="random seed for sampling")
    parser.add_argument("--model_path", default='../model', type=str, required=False, 
                       help="model path is required for least_confidence method")
    parser.add_argument("--output_dir", default='../samples', type=str, required=False, 
                       help="output directory")                   
    args = parser.parse_args()
    
    print("Loading data...")

    if args.source_dataset == 'anli':
        df_data = pd.read_json('../data/anli_train_r3_rationle.json', lines=True)
        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)
        data_sample = data.shuffle(seed=42) 
        data_sample = data_sample.rename_column("rationale", "explanation_1")

    if args.source_dataset == 'fm2':
        df_data = pd.read_json('../data/gpt-3.5-turbo-0125/fm2/train_select.json', lines=True)
        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", 'none', "contradiction"])
        data_sample = data.cast_column("label", labelNames)

    if args.source_dataset == 'snopes_stance':
        df_data = pd.read_json('../data/gpt-3.5-turbo-0125/snopes_stance/train_select.json', lines=True)
        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", 'neutral', "contradiction"])
        data_sample = data.cast_column("label", labelNames)
    
    if args.source_dataset == 'vitaminc':
        df_data = pd.read_json('../data/gpt-3.5-turbo-0125/vitaminc/train_select.json', lines=True)
        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", 'neutral', "contradiction"])
        data_sample = data.cast_column("label", labelNames)

    if args.source_dataset == 'averitec':
        if args.gen_path!='random':
            df_data = pd.read_json('../data/'+args.gen_path+'/averitec/train_select.json', lines=True)
            df_data['explanation_gpt4']=[str(e) for e in df_data['explanation_gpt4']]
            data = datasets.Dataset.from_pandas(df_data)
            labelNames = datasets.ClassLabel(names=["entailment", 'neutral', "contradiction", "mixture"])
            data_sample = data.cast_column("label", labelNames)
        else:
            data_sample=load_raw_data(args.source_dataset,'train')
    if args.source_dataset == 'pubhealth':
        if args.gen_path!='random':
            df_data = pd.read_json('../data/'+args.gen_path+'/pubhealth/train_select.json', lines=True)
            df_data['explanation_gpt4']=[str(e) for e in df_data['explanation_gpt4']]
            data = datasets.Dataset.from_pandas(df_data)
            labelNames = datasets.ClassLabel(names=["entailment", 'neutral', "contradiction", "mixture"])
            data_sample = data.cast_column("label", labelNames)
        else:
            data_sample=load_raw_data(args.source_dataset,'train')
    
    save_dir = '/'.join((args.output_dir, args.source_dataset, args.gen_path))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.sample_selection == 'random':
        #generate 10 random seeds for sample selection
#         seeds = [x for x in range(0, 10)]
#         for s in tqdm(seeds):
        tmp_dir = save_dir
        sample_seed = args.sample_seed
        for ns in [4,8,16]: #4,8,16,32,64,128,256,512
            sample = random_select(data_sample, sample_seed, ns)
            save_dir = tmp_dir + '/' + str(ns) + '/seed_'+str(sample_seed)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            sample.to_json(save_dir + '/train_select.json', orient='records', lines=True)
            print("succefully saved!")

    if args.sample_selection == 'fastvotek':
        fastvotek(data_sample, save_dir)

    if args.sample_selection == 'accept-fastvotek':
        data_sample = acceptability(data_sample, save_dir, thres = 0.3)
        fastvotek(data_sample, save_dir)

    if args.sample_selection == 'least_confidence':
        least_confidence(data_sample, args.model_path, save_dir)

    if args.sample_selection == 'accept-least_confidence':
        data_sample = acceptability(data_sample, save_dir, thres = 0.3)
        least_confidence(data_sample, args.model_path, save_dir)   
        
    if args.sample_selection == 'ambiguous':
        ambiguous(data_sample, args.model_path, save_dir)

    if args.sample_selection == 'accept-ambiguous':
        data_sample = acceptability(data_sample, save_dir, thres = 0.3)
        ambiguous(data_sample, args.model_path, save_dir)    

if __name__ == "__main__":
    main()

