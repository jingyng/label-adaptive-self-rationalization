import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="5,6"

import wandb
os.environ["WANDB_WATCH"]='false'
os.environ["WANDB_LOG_MODEL"]='false'
# os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_API_KEY"] = "f943c9ab18325b5cd45241ad2df308268a65c7b8"
# os.environ["WANDB_CONFIG_DIR"] = "/ukp-storage-1/jyang/wandb/.config"
# os.environ["WANDB_CACHE_DIR"] = "/ukp-storage-1/jyang/wandb/.cache"
os.environ["WANDB_JOB_TYPE"] = 'training'

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

from transformers.integrations import WandbCallback
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

from feature_conversion_methods import format_instance,formatting

from sklearn.metrics import classification_report

import torch
import datasets
from load_custom_dataset import load_raw_data,load_format_data
from copy import deepcopy 

InputDataClass = NewType("InputDataClass", Any)

# CONFIG_MAPPING = {"t5": T5Config}
# TOKENIZER_MAPPING = {"t5": T5Tokenizer}

def set_other_seeds(seed):
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
# class SequenceCollator:
#     def __init__(self, model, pad_token):
#         self.model = model
#         self.pad_token_mapping = {
#             "labels": -100,
#             "attention_mask": 0,
#             "decoder_attention_mask": 0,
#             "input_ids": pad_token,
#         }

#         self.columns = [
#             "input_ids",
#             "attention_mask",
#             "labels",
#             "decoder_attention_mask",
#         ]

#     def __call__(self, examples: List[Dict[str, InputDataClass]]) -> Dict[str, torch.Tensor]:
#         # re-format inputs for training
#         batch = {}
#         for key in examples[0].keys():
#             if key in self.columns:
#                 tmp_list = []
#                 for item in examples:
#                     tmp_list.append(item[key])

#                 # pad lists to max length
#                 if isinstance(tmp_list[0], list):
#                     max_length = max(map(len, tmp_list))
#                     tmp_list = [
#                         el + [self.pad_token_mapping[key]] * (max_length - len(el))
#                         for el in tmp_list
#                     ]

#                 batch[key] = torch.tensor(tmp_list, dtype=torch.long)
#         return batch

def format_data(dataset, task, tokenizer, explanation_sep, io_format):
    
    if dataset is not None:
        dataset = dataset.map(
            lambda x: format_instance(
                x,
                task,
                tokenizer,
                explanation_sep,
                io_format=io_format
            ),
            batched=False,
            load_from_cache_file=False,
        )

    return dataset

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def train(save_model_path, model, tokenizer, train_bsz, gradient_accumulation_steps, random_seed, data, lr, num_epochs, model_class='t5'):

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir = save_model_path,
        do_train=True,
        do_eval=False,
        logging_strategy = 'no',
        save_strategy = 'no',
        evaluation_strategy ='no',
        learning_rate=lr,
        per_device_train_batch_size=train_bsz,
        num_train_epochs=num_epochs,
        push_to_hub=False,
        lr_scheduler_type = 'constant',
        warmup_steps = 0,
        seed = random_seed,
        load_best_model_at_end=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # deepspeed="ds_config.json",  # Point to the DeepSpeed configuration file
        # fp16=True,
    )
        
    callbacks = [WandbCallback()]

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        # data_collator=SequenceCollator(
            # model=model_class, pad_token=tokenizer.pad_token_id
        # ),
        data_collator=data_collator,    
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    
    trainer.train()
    
#     trainer.save_model(save_model_path)
    trainer.model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)

    print('Model saved at: ', save_model_path)

def sep_label_explanation(lines, explanation_sep):
    
    labels = []
    explanations = []
    for line in lines:
        line_split = line.split(explanation_sep)
        if len(line_split) > 1:
            labels.append(line_split[0].strip()) #text label: entailment, neutral, contradiction
            explanations.append(line_split[1].strip())
        else: 
            try:
                # print(f"This line couldn't be processed (most likely due to format issue): {line}")
                labels.append(line.split()[0]) #text label: maybe nonsense, we assume the first word is always the label
                explanations.append(' '.join(line.split()[1:]))
            except:
                labels.append(line) #the line is totally empty
                explanations.append('UNK') 

    return labels, explanations
    # return sum(acc)/len(predictions)



def train_eval(save_model_path, task, model, tokenizer, train_bsz, gradient_accumulation_steps, eval_bsz, random_seed, data, lr, num_epochs, experiment_id, model_class='t5'):
    
    explanation_sep = ' "explanation: " '
    rouge = load('rouge', experiment_id=experiment_id)
    bertscore = load("bertscore", experiment_id=experiment_id)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True)
    
    def compute_metrics(eval_preds):

        preds, labels = eval_preds # preds are predicted tokens, labels are ground truth tokens
#         print(preds)
        print(preds.shape)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
#         decoded_labels = sum(decoded_labels, [])  
        print("Prediction:",decoded_preds)
        print("Gold:",decoded_labels)
        pred_l, pred_e = sep_label_explanation(decoded_preds, explanation_sep)
        gold_l, gold_e = sep_label_explanation(sum(decoded_labels,[]), explanation_sep)

        # bleu_result = bleu.compute(predictions=pred_e, references=[gold_e])
        rouge_result = rouge.compute(predictions=pred_e, references=gold_e)
        bertscore_result = bertscore.compute(predictions=pred_e, references=gold_e, model_type="microsoft/deberta-xlarge-mnli")        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        accuracy = sum([pred_l[i] == gold_l[i] for i in range(len(pred_l))])/len(pred_l)

        # result = {'bleu' : bleu_result['score']}
        result= {"gen_len": np.mean(prediction_lens)}
        result["rouge1"] = np.mean(rouge_result["rouge1"])
        result["rouge2"] = np.mean(rouge_result["rouge2"])
        result["rougeL"] = np.mean(rouge_result["rougeL"])
        result["rougeLsum"] = np.mean(rouge_result["rougeLsum"])
        result["bertscore"] = np.mean(bertscore_result["f1"])
        result["accuracy"] = accuracy
        result['acc_bertscore'] = accuracy + result["bertscore"]
        result = {k: round(v, 4) for k, v in result.items()}
        
        return result
    
    training_args = Seq2SeqTrainingArguments(
        output_dir = save_model_path,
        do_train=True,
        do_eval=True,
        logging_strategy = 'epoch',
        evaluation_strategy ='epoch',
        save_strategy = 'epoch',
        learning_rate=lr,
        per_device_train_batch_size=train_bsz,
        per_device_eval_batch_size=eval_bsz,
        num_train_epochs=num_epochs,
        push_to_hub=False,
        metric_for_best_model = 'eval_acc_bertscore',
        greater_is_better = True,
        lr_scheduler_type = 'constant',
        warmup_steps = 0,
        seed = random_seed,
        save_total_limit = 1,
        remove_unused_columns=True,
        load_best_model_at_end=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        predict_with_generate=True,
    )
        
    callbacks = [WandbCallback()]

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        eval_dataset=data['validation'],
        # data_collator=SequenceCollator(
            # model=model_class, pad_token=tokenizer.pad_token_id
        # ),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    trainer.train()
    
    trainer.save_model(save_model_path)
    trainer.model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)
    
    print('Model saved at: ', save_model_path)

    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", default='double-transfer-anli', type=str, required=False, 
                       help="wandb project name for the experiment")
    parser.add_argument("--source_dataset_name", default='pubhealth', type=str, required=False, 
                       help="training source dataset name")
    parser.add_argument("--target_dataset_name", default='pubhealth', type=str, required=False, 
                       help="training target dataset name")
    parser.add_argument("--model_class", default='t5', type=str, required=False, 
                       help="model base")
    parser.add_argument("--model_name", default= 't5-3b', type=str, required=False, 
                       help="model and tokenizer name")
    parser.add_argument("--random_seed", default= 42, type=int, required=False, 
                       help="random seed for each experiment")
    parser.add_argument("--sample_seed", default=42, type=int, required=False, 
                       help="random seed for sampling")
    parser.add_argument("--lr", default= 3e-6, type=float, required=False, 
                       help="learning rate")
    parser.add_argument("--train_bsz", default= 2, type=int, required=False, 
                       help="training batch size")
    parser.add_argument("--gradient_accumulation_steps", default= 1, type=int, required=False, 
                       help="gradient accumulation steps")
    parser.add_argument("--eval_bsz", default= 4, type=int, required=False, 
                       help="training batch size")                   
    parser.add_argument("--n_shots", default= 5000, type=int, required=False, 
                       help="number of shots per class")
    parser.add_argument("--sample_selection", default= 'random', type=str, required=False, 
                       help="sample selection method")
    parser.add_argument("--explanation_sep", default= ' "explanation: " ', type=str, required=False, 
                       help="separation string between prediction and explanation")
    parser.add_argument("--explanation_source", default= 'gpt_4', type=str, required=False, 
                       help="source of explanations")
    parser.add_argument("--io_format", default= 'standard', type=str, required=False, 
                       help="nli prompt format")
    parser.add_argument("--save_model_path", default= '../model/', type=str, required=False, 
                       help="path to save model")
    parser.add_argument("--source_model_path", default= 'nt5000', type=str, required=False, 
                       help="path to save model")
    parser.add_argument("--num_epochs", default= 10, type=int, required=False, 
                       help="number of training epochs")
    parser.add_argument("--max_length", default= 1024, type=int, required=False, 
                       help="number of training epochs")
    parser.add_argument("--max_length_target", default= 16, type=int, required=False, 
                       help="number of training epochs")
    parser.add_argument("--do_eval", action='store_true',
                       help="whether perform validation or not")
    parser.add_argument("--second_transfer", action='store_true',
                       help="second transfer")
    
    args = parser.parse_args()
    
    learning_rate = args.lr

    if args.io_format=='no_exp':
        run_name = "/".join((args.model_name.replace('/', "-"), args.source_dataset_name, args.source_model_path, 'no_exp'))
        data_train = load_raw_data(args.source_dataset_name, split ='train')

    elif args.n_shots == 5000 and args.io_format!='no_exp':
        if args.second_transfer:
            run_name = "/".join((args.model_name.replace('/', "-"), args.source_dataset_name, args.source_model_path, args.target_dataset_name, 'full'))
        else:
            run_name = "/".join((args.model_name.replace('/', "-"), args.source_dataset_name, args.source_model_path, 'full'))
        data_train = load_raw_data(args.source_dataset_name, split ='train') 
    
    else:
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
            run_name = "/".join((args.model_name.replace('/', "-"), args.source_dataset_name, args.source_model_path, args.target_dataset_name, args.sample_selection, end_path, 'seed_'+str(args.sample_seed)))
        else:
            run_name = "/".join((args.model_name.replace('/', "-"), args.source_dataset_name, args.sample_selection, end_path, 'seed_'+str(args.sample_seed)))

        data_path =  "/".join(('../samples',args.source_dataset_name, args.sample_selection, str(args.n_shots), 'seed_'+str(args.sample_seed)))
        data_train = datasets.load_dataset("json", data_files=data_path+"/train_select.json", split="train")  
    
    wandb.init(project=args.project_name, 
           name=run_name,
           tags=[args.sample_selection,'train'],
           config = args,
           save_code = True)    

    set_seed(args.random_seed)
    set_other_seeds(args.random_seed)

    print('Data loading...')
        
    if args.do_eval:
        data_eval = load_raw_data(args.source_dataset_name, split ='val')

    print('Model initializing...')
    if args.second_transfer:
        source_model_path ='../model/'+args.model_name.replace('/', "-")+'/'+args.source_dataset_name+'/'+args.source_model_path+'/no_exp/'
        config = AutoConfig.from_pretrained(source_model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(source_model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(source_model_path, local_files_only=True,use_safetensors=True, device_map="auto")
    else:
        config = AutoConfig.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, config=config, use_safetensors=True,device_map="auto")

    def update_columns(example):
        input_string, answer_string = formatting(example, args.source_dataset_name, args.explanation_sep, args.io_format, args.explanation_source)
        example['input_string'] = input_string
        example['answer_string'] = answer_string
        return example
    
    def preprocess_function(examples):
        inputs = examples["input_string"]
        targets = examples["answer_string"]
        input_ids = tokenizer.batch_encode_plus(inputs,  padding=True, truncation=True, max_length=args.max_length)  #truncation=True, max_length=512
        target_ids = tokenizer.batch_encode_plus(targets, padding=True, max_length=args.max_length_target) #max_length=256
        return {"input_ids": input_ids["input_ids"], "labels": target_ids["input_ids"]}

    print('Data formatting...')
    data_splits = {'train': None, 'validation': None, 'test': None}

    data_train=data_train.map(update_columns)
    data_splits['train']=data_train.map(preprocess_function, batched=True)
    data_splits['train'] = data_splits['train'].remove_columns(data_train.column_names)
    print(tokenizer.decode(data_splits['train']['input_ids'][0]))
    print(tokenizer.decode(data_splits['train']['labels'][0]))
    
    if args.do_eval:
        data_splits['validation']=data_eval.map(preprocess_function, batched=True)
        data_splits['validation'] = data_splits['validation'].remove_columns(data_eval.column_names)
        print(tokenizer.decode(data_splits['validation']['input_ids'][0]))
        print(tokenizer.decode(data_splits['validation']['labels'][0]))
        
    print('Start training...')
    save_model_path = '/'.join((args.save_model_path,run_name))

    if not args.do_eval:
        train(
            save_model_path, 
            model=model, 
            tokenizer = tokenizer, 
            train_bsz=args.train_bsz, 
            random_seed=args.random_seed, 
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            data=data_splits, 
            lr=learning_rate, 
            num_epochs=args.num_epochs
        )
    else:    
        train_eval(
            save_model_path, 
            task=args.source_dataset_name,
            model=model, 
            tokenizer = tokenizer, 
            train_bsz=args.train_bsz, 
            eval_bsz=args.eval_bsz,
            random_seed=args.random_seed, 
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            data=data_splits, 
            lr=learning_rate, 
            experiment_id=args.n_shots,
            num_epochs=args.num_epochs
        )

    print('Finished training.')
    
    wandb.finish()
                      
if __name__ == "__main__":
    main()