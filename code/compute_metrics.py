import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


import pandas as pd
import numpy as np
import json
import datasets
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from nli_demo import length_mapping, get_scores
from feature_conversion_methods import label_mapping_t5
import torch
from load_custom_dataset import load_raw_data, load_format_data

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
    
def merge_answers(ans,num_classes,model_type='t5'):
    if model_type=='t5':
        if ans.lower() == 'entailment':
            ans=0
        elif ans.lower() == 'contradiction' or num_classes==2:
            ans=2
        elif ans.lower() =='mixture':
            ans=3
        else:
            ans=1
            
    elif model_type=='flan_t5':
        if ans.lower() == 'yes':
            ans=0
        elif ans.lower() == 'no' or num_classes==2:
            ans=2
        else:
            ans=1
    return ans

def compute_acc(answers,gold_label,save_path):
    results = {}
    
    split='test'
    
    ## compute label prediction accuracy
    report=classification_report(y_true=gold_label, y_pred=answers, digits=4, output_dict=True)
    num_classes=len(set(gold_label))
    if num_classes==4:
        (
            results[f"{split}_acc"],
            results[f"{split}_entailment"],
            results[f"{split}_neutral"],
            results[f"{split}_contradiction"],
            results[f"{split}_mixture"],
        ) = (report['accuracy'], report['0']['f1-score'], report['1']['f1-score'], report['2']['f1-score'], report['3']['f1-score'])

    # for t
    
    if num_classes==3:
        (
            results[f"{split}_acc"],
            results[f"{split}_entailment"],
            results[f"{split}_neutral"],
            results[f"{split}_contradiction"],
        ) = (report['accuracy'], report['0']['f1-score'], report['1']['f1-score'], report['2']['f1-score'])

    # for tasks with two classes
    else:
        (
            results[f"{split}_acc"],
            results[f"{split}_entailment"],
            results[f"{split}_contradiction"],
        ) = (report['accuracy'], report['0']['f1-score'], report['2']['f1-score'])
    
    results[f"{split}_macro_avg_recall"] = report['macro avg']['recall']
    results[f"{split}_macro_avg_f1"] = report['macro avg']['f1-score']

    with open(os.path.join(save_path, f"results_{split}.json"), "w") as fp:
        json.dump(results, fp)

    with open(os.path.join(save_path, f"report_{split}.json"), "w") as fp:
        json.dump(report, fp)
    return results

def compute_acceptability(df,task,model_type,save_path,batch_size,correct_only=True):
    ## compute explanation acceptability score
    # if correct_only:
    #     df = df.loc[df['label'] == df['predicted_label']]
    # else:
    #     df = df.loc[df['label'] != df['predicted_label']]
        
    inputs = ['premise: '+ row['premise'] + ' hypothesis: ' + row['hypothesis'] + ' answer: '+ label_mapping_t5[task][row['label']] + ' explanation: ' + row['explanation'] for _, row in df.iterrows()] 
#     print(inputs[0])
    model_type = model_type
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scores = get_scores(
    inputs,
    model_type,
    device=device,
    batch_size=batch_size,
    verbose=False)
    
    if correct_only:
        df['accept_score_'+model_type] = scores
        if os.path.exists(save_path +'/results_test.json'):
            f = open(save_path +'/results_test.json')
            data = json.load(f)
            
            data['accept_score_'+model_type] = np.mean(scores)
            
            data['avg_acceptability_correct_'+model_type] = (df['accept_score_'+model_type].sum())/length_mapping[task]
            data['acc90_'+model_type] = len(df.loc[df['accept_score_'+model_type]>=0.90]['accept_score_'+model_type])/length_mapping[task]
            data['acc60_'+model_type] = len(df.loc[df['accept_score_'+model_type]>=0.60]['accept_score_'+model_type])/length_mapping[task]
            data['acc30_'+model_type] = len(df.loc[df['accept_score_'+model_type]>=0.30]['accept_score_'+model_type])/length_mapping[task]
               
            with open(save_path +'/results_test.json', 'w') as f:
                json.dump(data, f)
                
            print(data['accept_score_'+model_type])
            
        df.to_json(save_path + '/exp_correct_scores.json', orient='records', lines=True)
    else:
        df['accept_score_all'+model_type] = scores
        df.to_json(save_path + '/exp_scores.json', orient='records', lines=True)
    return np.mean(scores)