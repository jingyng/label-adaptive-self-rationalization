'''
A (hopefully) Simple API for serving explanation score requests.

question = 'premise: ' + item["Input.premise"].lstrip().rstrip() + ' ' + 'hypothesis: ' + item['Input.hypothesis'].lstrip().rstrip()
if item["Input.gold_label"] == 'TRUE':
    gold_label = 'entailment'
elif item["Input.gold_label"] == 'FALSE':
    gold_label = 'contradiction'
elif item["Input.gold_label"] == 'neither':
    gold_label = 'neutral'

input_string = (
    f"{question} answer: {gold_label}. "
    + f" explanation: {abstr_expl}."
)

here are some example input strings:

premise: A man getting a tattoo on his back. hypothesis: A woman is getting a tattoo. answer: contradiction. explanation: Because the tattoo artist is a man, the person getting the tattoo is not a woman.
premise: Woman is making a half smile at the camera. hypothesis: woman smiling answer: entailment. explanation: Many people do not find smiling to the camera appropriate or fun, so not everyone will be smiling but some will. A person may not make a smile or have a full smile, in which case it can still be true.
premise: some people sitting around a table, one yawning. hypothesis: A person at the table is yawning because he stayed awake all night watching Netflix. answer: neutral. explanation: Just because someone is sitting around a table does not mean they stayed awake all night
watching Netflix.
'''

import argparse
import torch
import transformers
import os
import tqdm
import numpy as np
import pandas as pd
import json
import re

_model, _tokenizer = None, None

model2url = {
    'large': 'https://storage.googleapis.com/ai2-mosaic-public/projects/few-shot-explanations/pretrained_models/nli/valloss%3D0.25146~model%3Dt5-large~lr%3D0.0001~seed%3D1~labelagg%3D0_just_weights.pt',
    '3b': 'https://storage.googleapis.com/ai2-mosaic-public/projects/few-shot-explanations/pretrained_models/nli/valloss%3D0.24209~model%3Dt5-3b~lr%3D0.0001~seed%3D1~labelagg%3D0_just_weights.pt',
    '11b': 'https://storage.googleapis.com/ai2-mosaic-public/projects/few-shot-explanations/pretrained_models/nli/esnli_deepspeed_valloss%3D0.00000~model%3Dt5-11b~lr%3D0.00001~seed%3D1~labelagg%3D0.pt'
}

length_mapping = {
    'sick': 4906, 'add_one': 387, 'joci': 39092, 'mpe': 1000, 'dnc': 60036, 'hans': 30000,
    'wnli': 71, 'glue_diagnostics': 1104, 'conj': 623, 'wanli': 5000, 'robust_nli_3': 74922, 'scinli': 3000,
    'snopes_stance': 1651, 'scifact': 300, 'climate-fever-combined': 1381, 'vitaminc': 55197, 'ambifc':5120, 'anli':1279, 'averitec':500,
    'covid_fact': 4086, 'dialfact': 11809, 'fm2': 1380, 'covert': 300, 'pubhealth': 1235
}

def get_model(model_type, device=None):
    global _model, model2url
    if model_type not in {'11b', '3b', 'large'}:
        raise NotImplementedError('{} is not a valid model please use "3b" or "large" or "11b"'.format(model_type))

    if _model is None:
        hf_model_name = 't5-' + model_type
        print('Loading model: this will run only once.')
        if model_type == 'large':
            model_path = '/home/few-shot-fact-checking/model/evaluation_model/valloss=0.25146~model=t5-large~lr=0.0001~seed=1~labelagg=0_just_weights.pt'
        elif model_type == '3b':
            model_path = '/home/few-shot-fact-checking/model/evaluation_model/valloss=0.24209~model=t5-3b~lr=0.0001~seed=1~labelagg=0_just_weights.pt'
        elif model_type == '11b':
            model_path = '/home/few-shot-fact-checking//model/evaluation_model/esnli_deepspeed_valloss=0.00000~model=t5-11b~lr=0.00001~seed=1~labelagg=0.pt'

        if not os.path.exists(model_path):
            print('Please download weights for {} model and put in current directory.'.format(model_path))
            print('for example, wget {}'.format(model2url[model_type]))
            quit()

        state = torch.load(model_path)
        if 'model_state_dict' in state:
            state = state['model_state_dict']

        _model = transformers.AutoModelForSeq2SeqLM.from_pretrained(hf_model_name,torch_dtype=torch.float16,device_map="auto")
        if model_type == '11b': # need to resize due to deepspeed, these entires are not accessed.
            _model.resize_token_embeddings(len(transformers.AutoTokenizer.from_pretrained(hf_model_name)))
        _model.load_state_dict(state)
        _model.eval()
        if device is not None:
            _model = _model.to(device)

    return _model

def get_tokenizer(model_type):
    global _tokenizer
    if model_type not in {'3b', 'large', '11b'}:
        raise NotImplementedError('{} is not a valid model please use "3b" or "large" or "11b"'.format(model_type))

    if _tokenizer is None:
        hf_model_name = 't5-' + model_type
        _tokenizer = transformers.T5TokenizerFast.from_pretrained(hf_model_name)

    return _tokenizer


class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        res = self.tokenizer(self.data[idx]['input'], truncation=True, max_length=4096)
        res['labels'] = self.tokenizer(self.data[idx]['label']).input_ids
        return res

    def __len__(self):
        return len(self.data)

def get_scores(inputs, model_type, device=None, batch_size=32, verbose=False):
    '''
    Inputs:
      - a list of explanations to score, e.g.,:
        premise: A man getting a tattoo on his back. hypothesis: A woman is getting a tattoo. answer: contradiction. explanation: Because the tattoo artist is a man, the person getting the tattoo is not a woman.
      - model type, either "3b" or "large" or "11b"
      - device: which torch device to load model on, e.g., "cuda:3"
    Outputs:
      - P(good explanation); higher is better
    '''
    assert model_type in {'large', '3b', '11b'}

    if isinstance(inputs, str):
        inputs = [inputs]

    model = get_model(model_type, device=device)
    tokenizer = get_tokenizer(model_type)

    score_itr = T5Dataset([{'input': inp, 'label': 'x'} for inp in inputs], tokenizer) # dummy labels for inference
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        return_tensors='pt'
    )
    score_itr = torch.utils.data.DataLoader(score_itr, shuffle=False, collate_fn=data_collator, batch_size=batch_size)
    score_itr = score_itr if not verbose else tqdm.tqdm(score_itr, total=len(score_itr))

    good_idx, bad_idx = tokenizer('good').input_ids[0], tokenizer('bad').input_ids[0]
    scores = []
    with torch.no_grad():
        for batch in score_itr:
            if device is not None:
                input_ids, attention_mask, targets = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            model_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
            logits_pos = model_output['logits'][:, 0, good_idx].cpu().numpy()
            logits_neg = model_output['logits'][:, 0, bad_idx].cpu().numpy()
            exp_logit_pos, exp_logit_neg = np.exp(logits_pos), np.exp(logits_neg)
            scores.extend(list([float(x) for x in exp_logit_pos / (exp_logit_pos + exp_logit_neg)]))
    return scores
        
def parse_args():
    '''
    Optional args for main function, mostly just to test.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_type',
        default='large',
        choices={'large', '3b', '11b'})
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int)

    args = parser.parse_args()
    return args


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_dataset_name", default='esnli', type=str, required=False, 
                       help="training dataset name")
    parser.add_argument("--target_dataset_name", default=None, type=str, required=True, 
                       help="dataset name")
    parser.add_argument("--data_sub", default= 0, type=int, required=False, 
                       help="subset dataset")
    parser.add_argument("--model_class", default='t5', type=str, required=False, 
                       help="model base")
    parser.add_argument("--model_name", default='t5-large', type=str, required=False, 
                       help="model base")
    parser.add_argument("--sample_selection", default= 'random', type=str, required=False, 
                       help="sample selection method")
    parser.add_argument("--test_bsz", default= 128, type=int, required=False, 
                       help="test batch size")
    parser.add_argument("--result_path", default= '../result', type=str, required=False, 
                       help="path to save model")
    
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(1)

    df=[] # create an empty df as it is not used for the function
#     if os.path.exists(args.result_path+"/test_posthoc_analysis_1.txt"):
#         evaluate_score(args.result_path, df, args.target_dataset_name, args.test_bsz, only_correct=True)
    evaluate_score(args.result_path, df, args.target_dataset_name, args.test_bsz, only_correct=True)



if __name__ == '__main__':
    main()
