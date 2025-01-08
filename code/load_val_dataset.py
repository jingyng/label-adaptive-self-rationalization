import argparse
import os
from random import shuffle
import pandas as pd
import pyarrow as pa
import datasets
from datasets import Dataset
efever_label_mapping = {'SUPPORTS':0, 'NOT ENOUGH INFO':1, 'REFUTES':2}

def random_select(data, sample_size):
    labels = data.features['label'].names
    label_subsets = []
    for label in labels:
        label_int = data.features['label'].str2int(label)
        train_examples = [sample for sample in data if sample['label'] == label_int ]
        label_subsets = label_subsets + train_examples[:sample_size]
    tmp_df = pd.DataFrame(label_subsets)
    pa_tab= pa.Table.from_pandas(tmp_df)
    data = datasets.Dataset(pa_tab)
    return data

# def clustering():
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset", default='esnli', type=str, required=False, 
                       help="source dataset")
    parser.add_argument("--output_dir", default='../datasets/source/', type=str, required=False, 
                       help="output directory")                   
    args = parser.parse_args()
    
    print("Loading validation data...")
    if args.source_dataset == 'esnli':
    # First random select a subset: 5000, we fix the seed for this selection
        data = datasets.load_dataset(args.source_dataset)
        data_sample = data["validation"].shuffle(seed=42)

    if args.source_dataset == 'anli':
        data = datasets.load_dataset(args.source_dataset)
        data = datasets.concatenate_datasets([data["test_r1"], data["test_r2"], data["test_r3"]])
        data_sample = data.shuffle(seed=42) # there are only 3200 samples, so we do not select a subset
        data_sample = data_sample.rename_column("reason", "explanation_1")

    if args.source_dataset == 'efever':    
        fever = pd.read_json('../datasets/source/efever/fever_dev.jsonl', lines=True)
        efever = pd.read_json('../datasets/source/efever/efever_dev_set.jsonl', lines=True)
        df_data = pd.merge(fever, efever, on='id', how='inner')
        df_data = df_data.drop(columns=['id', 'verifiable', 'evidence'])
        df_data = df_data.rename(columns={"retrieved_evidence": "premise", "claim": "hypothesis", 'summary': 'explanation_1'})
        labels = [efever_label_mapping[row['label']] for _, row in df_data.iterrows()]
        df_data['label'] = labels
        ## next rows are to filter samples with explanations repeat hypothesis while the label is not entailment
        tmp = df_data.loc[df_data['label']!=0]
        tmp = tmp.loc[tmp['explanation_1']==tmp['hypothesis']]
        df_data = df_data[ ~df_data.index.isin(tmp.index) ]
        ## next rows are to filter samples with wrong explanations
        tmp = df_data.loc[df_data['label']!=1]
        tmp = tmp.loc[tmp['explanation_1']=="\"The relevant information about the claim is lacking in the context.\""]
        df_data = df_data[ ~df_data.index.isin(tmp.index)]
        
        data = Dataset.from_pandas(df_data)
        labelNames = datasets.ClassLabel(names=["entailment", "neutral", "contradiction"])
        data = data.cast_column("label", labelNames)
        data_sample = data.shuffle(seed=42)

    save_dir = '/'.join((args.output_dir, args.source_dataset))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sample = random_select(data_sample, 120)
    sample.to_json(save_dir + '/val_select.json', orient='records', lines=True)
    print("succefully saved!")

if __name__ == "__main__":
    main()

