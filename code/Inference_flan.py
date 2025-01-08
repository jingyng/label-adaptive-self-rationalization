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
from load_custom_dataset import load_raw_data
from utils import sep_label_explanation
from compute_metrics import compute_acc, compute_acceptability

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import datasets
import git
import transformers
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration, set_seed

def merge_answers(ans,num_classes):
    if ans.lower() == 'yes':
        ans=0
    elif ans.lower() == 'no' or num_classes==2:
        ans=2
    else:
        ans=1
    return ans

def set_other_seeds(seed):
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def sep_answer_rationale(lines, explanation_sep):
    
    answer = []
    rationale = []
    for line in lines:
        line_split = line.split(explanation_sep)
        if len(line_split) > 1:
            rationale.append(line_split[0].strip())
            answer.append(line_split[1].strip()) #text label: yes, maybe, no
        else: 
            try:
                # print(f"This line couldn't be processed (most likely due to format issue): {line}")
                rationale.append(line.split()[0]) 
                answer.append(' '.join(line.split()[1:]))
            except:
                rationale.append(line) #the line is totally empty
                answer.append('')  

    return rationale, answer

def formatting(item):

    premise = item["premise"]
    hypothesis = item["hypothesis"]
    
    input_string=f'Read the text and determine if the sentence is true (see options at the end)\ntext: {premise} \nSentence: {hypothesis}\nOPTIONS:\n− Yes\n− It’s impossible to say\n− No\ntrue or false: \nLet\'s think step by step.\n'
    return input_string

def inference(model, tokenizer, data, test_bsz, result_path):

    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(device)
    dataloader = DataLoader(data, batch_size=test_bsz, shuffle=False)
    
    answers_explanations = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model.generate(input_ids=input_ids, do_sample=True, max_new_tokens=256, top_p=0.8, no_repeat_ngram_size=3) #, top_p=0.8, no_repeat_ngram_size=3
        answers_explanations+=outputs

    ans_exp = []
    for ans in answers_explanations:
        ans_exp.append(tokenizer.decode(ans, skip_special_tokens=True))
        
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path + '/test_generation_batch.txt', 'w', encoding='utf-8') as f:
        for line in ans_exp:
            f.write(f"{line}\n")
                  
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", default='double-transfer-inference', type=str, required=False, 
                       help="wandb project name for the experiment")
    parser.add_argument("--source_dataset_name", default='anli', type=str, required=False, 
                       help="training dataset name")
    parser.add_argument("--target_dataset_name", default='fm2', type=str, required=True, 
                       help="dataset name")
    parser.add_argument("--data_split", default='test', type=str, required=False, 
                       help="dataset split: train, val, test")
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
    parser.add_argument("--test_bsz", default= 16, type=int, required=False, 
                       help="test batch size")
    parser.add_argument("--explanation_sep", default= ' "explanation: " ', type=str, required=False, 
                       help="separation string between prediction and explanation")
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
    
    if args.second_transfer:
        relative_path = "/".join((args.model_name.replace('/', "-"), args.source_dataset_name, args.source_model_path, args.target_dataset_name, args.sample_selection, 'nt'+str(args.n_shots)))
    else:
        relative_path = "/".join((args.model_name.replace('/', "-"), args.source_dataset_name, args.sample_selection, 'nt'+str(args.n_shots)))

    if args.n_shots == 0:     
        run_name = "/".join(('zero_shot',args.target_dataset_name))
        result_path = '/'.join((args.result_path, run_name))
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
    data = load_raw_data(args.target_dataset_name, args.data_split)
    input_string = list(map(lambda x: formatting(x), data))
    data = data.add_column("input_string", input_string)
        
    print("Loading model...")
    
    if args.n_shots == 0:
        # tokenizer_name = TOKENIZER_MAPPING[args.model_class]
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    else:
        print(model_path)
        config = T5Config.from_pretrained(model_path, local_files_only=False)
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        # print(tokenizer)
        model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=False)

    def tokenize_function(examples):
        return tokenizer(examples["input_string"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing input...")
    
    tokenized_datasets = data.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    print("Model inferencing...")
    inference(
        model=model, 
        tokenizer=tokenizer,
        data=tokenized_datasets, 
        test_bsz=args.test_bsz, 
        result_path=result_path
    )
    
    print("Evaluating results...")
    result=data.to_pandas()
    with open(result_path+'/test_generation_batch.txt', 'r') as file:
        lines = file.readlines()
    result['ans_exp'] = lines
    rationales, answers = sep_answer_rationale(result['ans_exp'], args.explanation_sep)
    gold_label=result['label']
    num_classes=len(set(gold_label))
    answers=[merge_answers(a,num_classes) for a in answers]
    result['explanation']=rationales
    result['predicted_label']=answers
    
    results = compute_acc(answers,gold_label,result_path)
    results['accept_score'] = compute_acceptability(result,args.target_dataset_name,result_path,batch_size=args.test_bsz)
    
    wandb.log(results)
    
    # print('Model saved at: ', model_path)
    print('Finished inference.')
    wandb.finish()

if __name__ == "__main__":
    main() 