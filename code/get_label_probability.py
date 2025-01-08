import pandas as pd
import numpy as np
import argparse
import os

from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    set_seed
)

import torch
from load_custom_dataset import load_format_data
from Inference import inference

CONFIG_MAPPING = {"t5": T5Config}
MODEL_MAPPING = {"t5": T5ForConditionalGeneration}
TOKENIZER_MAPPING = {"t5": T5Tokenizer}

def set_other_seeds(seed):
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", default='few-shot-inference', type=str, required=False, 
                       help="wandb project name for the experiment")
    parser.add_argument("--source_dataset_name", default='esnli', type=str, required=False, 
                       help="training dataset name")
    parser.add_argument("--target_dataset_name", default=None, type=str, required=True, 
                       help="dataset name")
    parser.add_argument("--data_sub", default= 0, type=int, required=False, 
                       help="subset dataset")
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
    parser.add_argument("--test_bsz", default= 64, type=int, required=False, 
                       help="test batch size")
    parser.add_argument("--explanation_sep", default= ' "explanation: " ', type=str, required=False, 
                       help="separation string between prediction and explanation")
    parser.add_argument("--model_path", default= '../model', type=str, required=False, 
                       help="path to save model")
    parser.add_argument("--result_path", default= '../result', type=str, required=False, 
                       help="path to save model")
    
    args = parser.parse_args()

    relative_path = "/".join((args.source_dataset_name, args.sample_selection, 'sub'+str(args.data_sub), 'nt'+str(args.n_shots)))

    model_path = '/'.join((args.model_path, relative_path))
    result_path = '/'.join((args.result_path, args.target_dataset_name, relative_path))

    data = load_format_data(args.target_dataset_name, 'test')

    set_seed(args.seed)
    set_other_seeds(args.seed)

    tokenizer_name = TOKENIZER_MAPPING[args.model_class]
    tokenizer = tokenizer_name.from_pretrained(model_path, local_files_only=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=False)

    labels, explanations, label_probabilities = inference(
        model=model, 
        tokenizer=tokenizer,
        seed=args.seed, 
        data=data, 
        test_bsz=args.test_bsz, 
        result_path=result_path,
        explanation_sep=args.explanation_sep
    )

    label_probabilities=np.asarray(label_probabilities)

    df_result=pd.DataFrame()
    df_result['label']=labels
#     df_result['explanation']=explanations
    df_result['prob_en']=label_probabilities[:,0]
    df_result['prob_neutral']=label_probabilities[:,1]
    df_result['prob_contradiction']=label_probabilities[:,2]
    df_result.to_json(result_path+'/l_e_prob.json', orient='records', lines=True)
    
if __name__ == "__main__":
    main() 

