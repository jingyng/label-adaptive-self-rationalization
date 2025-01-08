import os
import wandb
os.environ["WANDB_WATCH"]='false'
os.environ["WANDB_LOG_MODEL"]='false'
# os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_API_KEY"] = "f943c9ab18325b5cd45241ad2df308268a65c7b8"
# os.environ["WANDB_CONFIG_DIR"] = "/ukp-storage-1/jyang/wandb/.config"
# os.environ["WANDB_CACHE_DIR"] = "/ukp-storage-1/jyang/wandb/.cache"
os.environ["WANDB_JOB_TYPE"] = 'test'

from tqdm import tqdm
from datetime import datetime
import random 
import pandas as pd 
import numpy as np
from metrics import evaluate
# from nli_demo import evaluate_score
from load_custom_dataset import load_format_data
from utils import sep_label_explanation

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import datasets
import git
import transformers
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    set_seed
)

CONFIG_MAPPING = {"t5": T5Config}
MODEL_MAPPING = {"t5": T5ForConditionalGeneration}
TOKENIZER_MAPPING = {"t5": T5Tokenizer}

label2text = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

def set_other_seeds(seed):
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
class MyDataset(Dataset):
    def __init__(self, data, tokenizer):

        input_data = tokenizer.batch_encode_plus(data['input_string'], 
                                           return_tensors="pt", 
                                           padding=True, 
                                           return_token_type_ids=False,
                                           return_attention_mask=True,
                                          )
        
        self.input_ids = input_data['input_ids']
        self.attention_mask =input_data['attention_mask']

        
    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index]
    
    def __len__(self):
        return len(self.input_ids)

def inference(model, tokenizer, seed, data, test_bsz, result_path, explanation_sep):

    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    data_set = MyDataset(data,tokenizer)
    data_loader = DataLoader(dataset=data_set, batch_size=test_bsz, shuffle=False, num_workers = 2)

    explanations = []
    labels = []
    label_probabilities = []
    answers_explanations = []
    for i, batch in tqdm(enumerate(data_loader)):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_masks = attention_mask.to(device)
        output = model.generate(input_ids = input_ids, 
                                attention_mask=attention_masks,
                                max_length=200,
                                output_scores=True, 
                                return_dict_in_generate=True)
        
        answer_explanation = tokenizer.batch_decode(output.sequences.squeeze(), skip_special_tokens=True)
        answers_explanations+=answer_explanation
        _, exp = sep_label_explanation(answer_explanation, explanation_sep)

        explanations += exp
        probabilities = F.softmax(output.scores[0], dim=1)
        token_probability = probabilities[:,[3,7163,27252,4989]] # token id for "en", "neutral", "contradiction" and "mixture"
        label_probabilities+=token_probability.cpu().tolist()
        labels+=token_probability.argmax(dim=-1).cpu().tolist()

        torch.cuda.empty_cache()


    generations_list = []
    for i in range(len(labels)):
        ans_exp = answers_explanations[i].replace("\n", " ").replace(tokenizer.eos_token, " ").strip()
        # label = label2text[labels[i]]
        # answer_explanation = explanation_sep.join((label, explanation))
        generations_list.append(ans_exp)
        
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path + '/test_generation_batch.txt', 'w', encoding='utf-8') as f:
        for line in generations_list:
            f.write(f"{line}\n")
    
    return labels, explanations, label_probabilities
                  

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", default='double-transfer-inference', type=str, required=False, 
                       help="wandb project name for the experiment")
    parser.add_argument("--source_dataset_name", default='anli', type=str, required=False, 
                       help="training dataset name")
    parser.add_argument("--target_dataset_name", default='fm2', type=str, required=True, 
                       help="dataset name")
    parser.add_argument("--data_split", default='test', type=str, required=False, 
                       help="dataset split: train, val/dev, test")
    parser.add_argument("--model_class", default='t5', type=str, required=False, 
                       help="model base")
    parser.add_argument("--model_name", default='t5-3b', type=str, required=False, 
                       help="model base")
    parser.add_argument("--n_shots", default= 64, type=int, required=False, 
                       help="number of shots per class")
    parser.add_argument("--sample_selection", default= 'random', type=str, required=False, 
                       help="sample selection method")
    parser.add_argument("--seed", default= 42, type=int, required=False, 
                       help="random seed for each experiment")
    parser.add_argument("--sample_seed", default=42, type=int, required=False, 
                       help="random seed for sampling")
    parser.add_argument("--test_bsz", default= 16, type=int, required=False, 
                       help="test batch size")
    parser.add_argument("--explanation_sep", default= ' "explanation: " ', type=str, required=False, 
                       help="separation string between prediction and explanation")
    parser.add_argument("--explanation_source", default= 'original', type=str, required=False, 
                       help="source of explanations")
    parser.add_argument("--io_format", default= 'standard', type=str, required=False, 
                       help="nli prompt format")
    parser.add_argument("--source_select", action='store_true',
                       help="selecting samples for training source dataset selection")
    parser.add_argument("--model_path", default= '../model', type=str, required=False, 
                       help="path to save model")
    parser.add_argument("--source_model_path", default= 'nt5000', type=str, required=False, 
                       help="path to save model")
    parser.add_argument("--result_path", default= '../result', type=str, required=False, 
                       help="path to save model")
    parser.add_argument("--second_transfer", action='store_true',
                       help="second transfer")
    
    args = parser.parse_args()

    if args.explanation_source=='original':
        end_path='nt'+str(args.n_shots) 
    elif args.explanation_source=='gpt-4':     
        end_path='nt'+str(args.n_shots)+'_gpt4'
    elif args.explanation_source=='gpt-3.5':     
        end_path='nt'+str(args.n_shots)+'_gpt35'
    elif args.explanation_source=='llama-3':     
        end_path='nt'+str(args.n_shots)+'_llama'
    else:
        end_path='nt'+str(args.n_shots)+'_gpt'   
            
    if args.second_transfer:
        relative_path = "/".join((args.model_name.replace('/', "-"), args.source_dataset_name, args.source_model_path, args.target_dataset_name, args.sample_selection, end_path, 'seed_'+str(args.sample_seed)))
    else:
        relative_path = "/".join((args.model_name.replace('/', "-"), args.source_dataset_name, args.sample_selection, end_path, 'seed_'+str(args.sample_seed)))

    if args.n_shots == 0:     
        run_name = "/".join(('zero_shot',args.target_dataset_name))
        result_path = '/'.join((args.result_path, run_name))
        
    elif args.n_shots == 5000:
        if args.second_transfer:
            run_name = "/".join((args.model_name.replace('/', "-"), args.source_dataset_name, args.source_model_path, args.target_dataset_name, 'full'))
        elif args.io_format=='no_exp':
            run_name = "/".join((args.model_name.replace('/', "-"), args.source_dataset_name, args.source_model_path, 'no_exp'))
        else:
            run_name = "/".join((args.model_name.replace('/', "-"), args.source_dataset_name, args.source_model_path, 'full'))
        model_path = '/'.join((args.model_path, run_name))
        result_path = '/'.join((args.result_path, args.target_dataset_name, run_name))
    else:    
        model_path = '/'.join((args.model_path, relative_path))
        run_name = "/".join((args.target_dataset_name, relative_path))
        result_path = '/'.join((args.result_path, run_name))

    wandb.init(project=args.project_name, 
           name=run_name,
           tags=[args.target_dataset_name, args.sample_selection],
           group=args.target_dataset_name,
           config = args,
           save_code = True)    

    set_seed(args.seed)
    set_other_seeds(args.seed)

    print("Loading data...")
    data = load_format_data(args.target_dataset_name, args.data_split, explanation_sep=args.explanation_sep, io_format=args.io_format)
        
    print("Loading model...")
    # print("model path:",model_path)
    if args.n_shots == 0:
        tokenizer_name = TOKENIZER_MAPPING[args.model_class]
        tokenizer = tokenizer_name.from_pretrained(args.model_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        tokenizer_name = TOKENIZER_MAPPING[args.model_class]
        tokenizer = tokenizer_name.from_pretrained(model_path, local_files_only=False)
        model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=False, use_safetensors=True)

    print("Model inferencing...")
    labels, explanations, _ = inference(
        model=model, 
        tokenizer=tokenizer,
        seed=args.seed, 
        data=data, 
        test_bsz=args.test_bsz, 
        result_path=result_path,
        explanation_sep=args.explanation_sep
    )
    
    print("Evaluating results...")
    # evaluate classification accuracy
    results, cm = evaluate(
        result_path,
        data,
        tokenizer,
        "test",
        task=args.target_dataset_name,
        labels=labels,
        explanations=explanations
    )
    
    # evaluate explanation acceptability
#     df_data = data.to_pandas()
#     exp_score = evaluate_score(result_path, df_data, args.target_dataset_name, args.test_bsz//2)
#     results['accept_score'] = exp_score
    print(results)

#     df_cm = pd.DataFrame(cm)

#     wandb.log({"confusion_matrix": wandb.Table(dataframe=df_cm)})

    wandb.log(results)
    # print('Model saved at: ', model_path)
    print('Finished inference.')
    wandb.finish()

if __name__ == "__main__":
    main() 