import datasets
import pandas as pd
from feature_conversion_methods import formatting
import re
import numpy as np
URL = re.compile('((([A-Za-z]{3,9}:(?:\/\/)?)(?:[\-;:&=\+\$,\w]+@)?[A-Za-z0-9\.\-]+|(?:www\.|[\-;:&=\+\$,\w]+@)[A-Za-z0-9\.\-]+)((?:\/[\+~%\/\.\w\-_]*)?\??(?:[\-\+=&;%@\.\w_]*)#?(?:[\.\!\/\\\w]*))?)')

inverse_label_mapping = {
    "esnli": {'entailment':0, 'neutral':1, 'contradiction':2},
    'sick': {'entailment':0, 'neutral':1, 'contradiction':2},
    "scitail": {'not_entailment':0, 'entailment':1},
    "fm2": {'SUPPORTS':0, 'REFUTES':2},
    "covid_fact": {'SUPPORTED':0, 'REFUTED':2},
    "scifact": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},
    "covert": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},
    "vitaminc": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},
    "efever": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},
    "snopes_stance": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},
    "ambifc": {'supporting':0, 'neutral':1, 'refuting':2},
    # "ambifc": {'supporting':0, 'neutral':1, 'refuting':2},
    "climate-fever-combined": {'SUPPORTS':0, 'NOT_ENOUGH_INFO':1, 'REFUTES':2},
    "climate-fever-separate": {'SUPPORTS':0, 'NOT_ENOUGH_INFO':1, 'REFUTES':2}, 
    "dialfact-no-context": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},  
    "dialfact": {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2},    
}

def load_raw_data(dataset_name, split='test'):

    if dataset_name in ['esnli','efever'] and split =='val':
        data = datasets.load_dataset('json', data_files='../datasets/source/'+dataset_name+'/val_select.json', split='train')
    
    if dataset_name =='anli':
        # df_data = pd.read_json('../data/anli_train_r3_rationle.json', lines=True)
        df_data = pd.read_json('../data/anli/anli_'+split+'_split.json', lines=True)
        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)
        data = data.rename_column("rationale", "explanation_1")
        
    # if dataset_name == 'anli' and split == 'test':
    #     data = datasets.load_dataset(dataset_name)
    #     data = datasets.concatenate_datasets([data["test_r1"], data["test_r2"], data["test_r3"]])
    #     data = data.rename_column("reason", "explanation_1")   

    if dataset_name =='efever' and split =='train':
        fever = pd.read_json('/home/few-shot-fact-checking/datasets/source/efever/fever_train.jsonl', lines=True)
        efever = pd.read_json('/home/few-shot-fact-checking/datasets/source/efever/efever_train_set.jsonl', lines=True)
        df_data = pd.merge(fever, efever, on='id', how='inner')
        df_data = df_data.drop(columns=['id', 'verifiable', 'evidence'])
        df_data = df_data.rename(columns={"retrieved_evidence": "premise", "claim": "hypothesis", 'summary': 'explanation_1'})
        labels = [inverse_label_mapping[dataset_name][row['label']] for _, row in df_data.iterrows()]
        df_data['label'] = labels
        ## next rows are to filter samples with explanations repeat hypothesis while the label is not entailment
        tmp = df_data.loc[df_data['label']!=0]
        tmp = tmp.loc[tmp['explanation_1']==tmp['hypothesis']]
        df_data = df_data[ ~df_data.index.isin(tmp.index) ]
        ## next rows are to filter samples with wrong explanations
        tmp = df_data.loc[df_data['label']!=1]
        tmp = tmp.loc[tmp['explanation_1']=="\"The relevant information about the claim is lacking in the context.\""]
        df_data = df_data[ ~df_data.index.isin(tmp.index)]
        
        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)

    if dataset_name=='fm2':
        data = datasets.load_dataset('json', data_files='/home/few-shot-fact-checking/datasets/target/fact-check/fm2/'+split+'.jsonl', split='train')
        df_data = data.to_pandas()
        evidence = [' '.join([e['text'] for e in row['gold_evidence']]) for _, row in df_data.iterrows()]
        data = data.add_column("evidence", evidence) 

        cols_to_remove = data.column_names
        cols_to_remove.remove("text") # text is claim
        cols_to_remove.remove("evidence")
        cols_to_remove.remove("label")
        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("text", "hypothesis")
        data = data.rename_column("evidence", "premise")
        labels = [inverse_label_mapping[dataset_name][x['label']] for x in data]
        data = data.remove_columns("label")
        data = data.add_column("label", labels)

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name=='averitec':
        df_averi=pd.read_json('../data/averitec/data_'+split+'.json')
        df_averi=df_averi.loc[df_averi['claim']!=""]
        premise=[]
        for index, row in df_averi.iterrows():
            ind_q = 1
            pre = []
            for qa in row['questions']:
                ans = ' '.join([a['answer'] for a in qa['answers']])
                qna = ' '.join(('\nQuestion '+str(ind_q)+': ', qa['question'], '\nAnswer '+str(ind_q)+': ', ans, '\n'))
                ind_q+=1
                pre.append(qna)
            premise.append(" ".join(pre))
        df_averi['premise']=premise
        df_averi = df_averi.rename(columns={"claim": "hypothesis", 'justification': 'explanation_1'})
        data = datasets.Dataset.from_pandas(df_averi)
        labelNames = datasets.ClassLabel(names=["Supported", "Not Enough Evidence", "Refuted", "Conflicting Evidence/Cherrypicking"])
        data = data.cast_column("label", labelNames)

    if dataset_name=='pubhealth':
        df = pd.read_csv('../data/pubhealth/'+split+'.tsv',sep='\t')
        df = df.drop(columns=['date_published', 'fact_checkers', 'sources','subjects'])
        df= df.dropna()
        df = df.rename(columns={"main_text": "premise", "claim": "hypothesis", 'explanation': 'explanation_1'})
        df=df.loc[df['label'].isin(['false', 'mixture', 'unproven', 'true'])]

        data = datasets.Dataset.from_pandas(df)
        labelNames = datasets.ClassLabel(names=["true", "unproven", "false", "mixture"])
        data = data.cast_column("label", labelNames)
        
    if dataset_name=='vitaminc':
        data = datasets.load_dataset('tals/vitaminc')
        data = data[split]
        if split=='train':
            shuffled_dataset = data.shuffle(seed=42)
            data = shuffled_dataset.select(range(10000))
        cols_to_remove = data.column_names
        cols_to_remove.remove("claim")
        cols_to_remove.remove("evidence")
        cols_to_remove.remove("label")
        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("claim", "hypothesis")
        data = data.rename_column("evidence", "premise")

        labels = [inverse_label_mapping[dataset_name][x['label']] for x in data]
        data = data.remove_columns("label")
        data = data.add_column("label", labels)

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column)             
        
    if dataset_name == 'covid_fact':

        data = datasets.load_dataset('json', data_files='/home/few-shot-fact-checking/datasets/target/fact-check/covid_fact/COVIDFACT_dataset.jsonl',split='train')

        cols_to_remove = data.column_names
        cols_to_remove.remove("claim") 
        cols_to_remove.remove("evidence")
        cols_to_remove.remove("label")
        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("claim", "hypothesis")
        data = data.rename_column("evidence", "premise")
        data = data.map(lambda example: {'premise': " ".join(example['premise'])})
        labels = [inverse_label_mapping[dataset_name][x['label']] for x in data]
        data = data.remove_columns("label")
        data = data.add_column("label", labels)

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name == 'covert':
        df_data = pd.read_json('/home/few-shot-fact-checking/datasets/target/fact-check/covert/CoVERT_FC_annotations.jsonl', lines=True)
        claim = [re.sub(URL, 'URL', row['claim']) for _, row in df_data.iterrows()]
        claim = [re.sub('@username', '', c) for c in claim]
        evidence = [' '.join([e[2] for e in row['evidence'] if e[2] is not None]) for _, row in df_data.iterrows()]
        df_data['claim'] = claim
        df_data['evidence'] = evidence
        df_data = df_data.rename(columns={"evidence": "premise", "claim": "hypothesis"})
        labels = [inverse_label_mapping[dataset_name][row['label']] for _, row in df_data.iterrows()]
        df_data['label'] = labels

        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 
        
    if dataset_name == 'snopes_stance':

        df_data = pd.read_json('/home/few-shot-fact-checking/datasets/target/fact-check/snopes/ukp_snopes_corpus/datasets/snopes.stance.'+split+'.jsonl', lines=True)
        df_evidence = pd.read_json('/home/few-shot-fact-checking/datasets/target/fact-check/snopes/ukp_snopes_corpus/datasets/snopes.page.json').T 
        df_data['evidence'] = [" ".join([df_evidence.loc[evi[0]]['lines'][evi[1]] for evi in row['evidence'][0]]) for _, row in df_data.iterrows()]

        df_data = df_data.drop(columns=['id', 'verifiable', 'predicted_evidence', 'predicted_pages'])
        df_data = df_data.rename(columns={"evidence": "premise", "claim": "hypothesis"})
        labels = [inverse_label_mapping[dataset_name][row['label']] for _, row in df_data.iterrows()]
        df_data['label'] = labels

        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    if dataset_name == 'ambifc':
        df_data=pd.read_json('../data/ambifc/ambifc_publish/test.certain.jsonl',lines=True)
        label=[l[0]['label'] for l in df_data['passage_annotations']]
        premise=['\n'.join(d.values()) for d in df_data['sentences']]
        df_data['label']=label
        df_data['premise']=premise

        df_data = df_data.drop(columns=['labels', 'sentences', 'metadata', 'wiki_page', 'wiki_section', 'wiki_passage', 'entity', 'section', 'passage_start_sentence_idx', 'passage_annotations','sentence_annotations'])
        df_data = df_data.rename(columns={"claim": "hypothesis"})
        
        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["supporting", "neutral", "refuting"])
        data = data.cast_column("label", labelNames)

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 
        
    if dataset_name == 'climate-fever-combined':
        data=datasets.load_dataset('json', data_files='/home/few-shot-fact-checking/datasets/target/fact-check/climate-fever/climate-fever-dataset-r1.jsonl')
        data=data['train']

        cols_to_remove = data.column_names
        cols_to_remove.remove("claim")
        cols_to_remove.remove("evidences")
        cols_to_remove.remove("claim_label")
        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("evidences", "premise")
        data = data.rename_column("claim", "hypothesis")
        data = data.rename_column("claim_label", "label")
        data = data.filter(lambda example: example["label"]!='DISPUTED')

        data = data.map(lambda example: {'premise': " ".join([x['evidence'] for x in example['premise']])})
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})
        
        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 
    
    if dataset_name == 'scifact':
        df_data = pd.read_json('/home/few-shot-fact-checking/datasets/target/fact-check/scifact/data/claims_dev.jsonl', lines=True) 
        df_evidence = pd.read_json('/home/few-shot-fact-checking/datasets/target/fact-check/scifact/data/corpus.jsonl', lines=True)
        claim = []
        evidence = []
        label = []
        for item, row in df_data.iterrows():
            claim.append(row['claim'])
            evi = []
            for e in row['cited_doc_ids']:
                df = df_evidence.loc[df_evidence['doc_id']==e]
                evi.append(' '.join(df.iloc[0]['abstract']))
            evidence.append(' '.join(evi))  
            if not row['evidence']:
                label.append('NOT ENOUGH INFO')
            else:
                l = 0
                s = 0
                for _, value in enumerate(row['evidence'].values()):
                    s = s+len(value)
                    for v in value:
                        if v['label']=='CONTRADICT':
                            l+=1
                if l/s<0.5:
                    label.append('SUPPORTS')
                else:
                    label.append('REFUTES')
                    
        dict_prepare = {
            'hypothesis':claim,
            'premise':evidence,
            'label':label
        }
        df_data = pd.DataFrame(dict_prepare, columns = ['hypothesis','premise', 'label'])
        df_data['explanation_1'] = [""]*len(df_data)

        labels = [inverse_label_mapping[dataset_name][row['label']] for _, row in df_data.iterrows()]
        df_data['label'] = labels

        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)    
        
    if dataset_name == 'climate-fever-separate':
        df_data = pd.read_json('/home/few-shot-fact-checking/datasets/target/fact-check/climate-fever/climate-fever-dataset-r1.jsonl', lines=True)
        claim = []
        evidence = []
        label = []
        for item, row in df_data.iterrows():
            for e in row['evidences']:
                claim.append(row['claim'])
                evidence.append(e['evidence'])
                label.append(e['evidence_label'])
        dict_prepare = {
            'hypothesis':claim,
            'premise':evidence,
            'label':label
        }
        df_data = pd.DataFrame(dict_prepare, columns = ['hypothesis','premise', 'label'])
        df_data['explanation_1'] = [""]*len(df_data)

        labels = [inverse_label_mapping[dataset_name][row['label']] for _, row in df_data.iterrows()]
        df_data['label'] = labels

        data = datasets.Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)        
    
    if dataset_name == 'dialfact-no-context': # context are not combined with evidence
        data = datasets.load_dataset('json', data_files='/home/few-shot-fact-checking/datasets/target/fact-check/dialfact/test_split.jsonl', split='train') 
        cols_to_remove = data.column_names
        cols_to_remove.remove("context") 
        cols_to_remove.remove("response") # response is the claim
        cols_to_remove.remove("evidence_list")
        cols_to_remove.remove("response_label")

        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("response", "hypothesis")
        data = data.rename_column("evidence_list", "premise")
        data = data.rename_column("response_label", "label")

        data = data.map(lambda example: {'premise': " ".join([x[2] for x in example['premise']])})
        data = data.map(lambda example: {'context': " ".join(example['context'])})
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 
    
    if dataset_name == 'dialfact': # context are combined with evidence
        data = datasets.load_dataset('json', data_files='/home/few-shot-fact-checking/datasets/target/fact-check/dialfact/test_split.jsonl', split='train') 
        cols_to_remove = data.column_names
        cols_to_remove.remove("context") 
        cols_to_remove.remove("response") # response is the claim
        cols_to_remove.remove("evidence_list")
        cols_to_remove.remove("response_label")

        data = data.remove_columns(cols_to_remove)
        data = data.rename_column("response", "hypothesis")
        data = data.rename_column("evidence_list", "premise")
        data = data.rename_column("response_label", "label")

        data = data.map(lambda example: {'premise': " ".join([x[2] for x in example['premise']])})
        data = data.map(lambda example: {'context': " ".join(example['context'])})
        data = data.map(lambda example: {'premise': " ".join((example['context'],example['premise']))})
        data = data.map(lambda x: {'label': inverse_label_mapping[dataset_name][x['label']]})

        new_column = [""] * len(data)
        data = data.add_column("explanation_1", new_column) 

    return data

def load_format_data(dataset_name, split='test', explanation_sep=' "explanation: " ', io_format='standard'):
    # if split=='train':
    #     data_path =  "/".join(('../samples',dataset_name, 'random', '16'))
    #     data=datasets.load_dataset("json", data_files=data_path+"/train_select.json", split="train")
    #     print(data)
    # else:
    data = load_raw_data(dataset_name, split)

    input_string, answer_string = zip(*list(map(lambda x: formatting(x, dataset_name, explanation_sep, io_format), data)))
    data = data.add_column("input_string", input_string)
    data = data.add_column("answer_string", answer_string)

    return data