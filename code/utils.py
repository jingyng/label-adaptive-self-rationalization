import os
import pandas as pd
import numpy as np
# from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# since we do not use label, fastvotek is unsupervised
def format_example(example):
    # we format the same way as used in https://github.com/HKUNLP/icl-selective-annotation/blob/c6912ac89a589d654dff3a7f9db18bfa3e4344fd/get_task.py    
    return "{} Based on that information, is the claim {} \"True\", \"False\", or \"Inconclusive\"?".format(example["premise"], example["hypothesis"]) 

def calculate_sentence_transformer_embedding(text_to_encode, embedding_model):
    num = len(text_to_encode)
    emb_model = SentenceTransformer(embedding_model)
    embeddings = []
    bar = tqdm(range(0,num,20),desc='calculate embeddings')
    for i in range(0,num,20):
        embeddings += emb_model.encode(text_to_encode[i:i+20]).tolist()
        bar.update(1)
    embeddings = torch.tensor(embeddings)
    mean_embeddings = torch.mean(embeddings, 0, True)
    embeddings = embeddings - mean_embeddings
    return embeddings

def fast_votek(embeddings, batch_size, k=150):
    
    n = len(embeddings)
    bar = tqdm(range(n),desc=f'voting')
    vote_stat = defaultdict(list)
    for i in range(n):
        cur_emb = embeddings[i].reshape(1, -1)
        cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
        sorted_indices = np.argsort(cur_scores).tolist()[-k-1:-1]
        for idx in sorted_indices:
            if idx!=i:
                vote_stat[idx].append(i)
        bar.update(1)

    votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)
    selected_indices = []
    selected_times = defaultdict(int)
    while len(selected_indices)<batch_size:
        cur_scores = defaultdict(int)
        for idx,candidates in votes:
            if idx in selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    cur_scores[idx] += 10 ** (-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(),key=lambda x:x[1])[0]
        selected_indices.append(int(cur_selected_idx))
        for idx_support in vote_stat[cur_selected_idx]:
            selected_times[idx_support] += 1
    return selected_indices

def sep_label_explanation(lines, explanation_sep):
    
    # broken_generation = 0
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
                explanations.append('')  

    return labels, explanations