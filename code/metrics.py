from tqdm import tqdm
import os
import pandas as pd
import numpy as np 
import torch
import datasets 
import json
from feature_conversion_methods import label_mapping_t5
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
label2text = {0: 'entailment', 1: 'neutral', 2: 'contradiction', 3: 'mixture'}
def evaluate(
    save_path,
    dataset,
    tokenizer,
    split,
    task,
    labels,
    explanations,
):

    # for accuracy 
    accuracy = None 
    label_true = []
    label_pred = []
    acc = []

    num_classes = len(label_mapping_t5[task])
 
    analysis_file = os.path.join(save_path, "%s_posthoc_analysis.txt" % split)
    if os.path.isfile(analysis_file):
        analysis_file = analysis_file.split(".txt")[0] + "_1.txt"

    with open(analysis_file, "w", encoding='utf-8') as g:
        for _, (pred_l, pred_e, gold) in tqdm(enumerate(zip(labels, explanations, dataset)), total=len(dataset)):
            # broken_generation = False

            pred_l = label2text[pred_l]
            # extract gold label and explanation
            gold_l = gold["label"] # 0,1[,2]   
            gold_l = label_mapping_t5[task][gold_l] #label to text: entailment, neutral, contradiction
            gold_explanations = gold[f"explanation_1"]
            g.write("Hypothesis: "+ gold["hypothesis"] + "\n")
            g.write("Premise: "+ gold["premise"] + "\n")
            g.write(f"Correct: {gold_l} | {gold_explanations} \n")
            
            if num_classes==2: # convert predicted label from 3 classes to 2 classes
                pred_l = "contradiction" if pred_l.lower() in ["neutral", "contradiction"] else "entailment" 

            g.write(f"Predicted: {pred_l} | {pred_e}\n")

            met = gold_l.lower() == pred_l.lower()

            label_pred.append(pred_l)  
            label_true.append(gold_l)  
            acc.append(met)
            g.write("Considered Correct: " + str(met) + "\n")
            g.write("\n")

    results = {}
    report = {}
    accuracy = sum(acc) / len(acc) * 100
    report = classification_report(y_true=label_true, y_pred=label_pred, digits=4, output_dict=True)
    confusion_max = confusion_matrix(y_true=label_true, y_pred=label_pred)
    # for tasks with three classes
    if num_classes==4:
        (
            results[f"{split}_acc"],
            results[f"{split}_entailment"],
            results[f"{split}_neutral"],
            results[f"{split}_contradiction"],
            results[f"{split}_mixture"],
        ) = (accuracy, report['entailment']['f1-score'], report['neutral']['f1-score'], report['contradiction']['f1-score'], report['mixture']['f1-score'])

    elif num_classes==3:
        (
            results[f"{split}_acc"],
            results[f"{split}_entailment"],
            results[f"{split}_neutral"],
            results[f"{split}_contradiction"],
        ) = (accuracy, report['entailment']['f1-score'], report['neutral']['f1-score'], report['contradiction']['f1-score'])

    # for tasks with two classes
    else:
        (
            results[f"{split}_acc"],
            results[f"{split}_entailment"],
            results[f"{split}_contradiction"],
        ) = (accuracy, report['entailment']['f1-score'], report['contradiction']['f1-score'])
    
    results[f"{split}_macro_avg_recall"] = report['macro avg']['recall']
    results[f"{split}_macro_avg_f1"] = report['macro avg']['f1-score']

    with open(os.path.join(save_path, f"results_{split}.json"), "w") as fp:
        json.dump(results, fp)

    with open(os.path.join(save_path, f"report_{split}.json"), "w") as fp:
        json.dump(report, fp)
    # with open(os.path.join(save_path, f"cm_{split}.json"), "w") as fp:
    #     json.dump(confusion_max, fp)    

    return results, confusion_max
