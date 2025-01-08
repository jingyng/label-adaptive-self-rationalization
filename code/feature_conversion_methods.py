from collections import defaultdict
import random
"""
Example-to-Feature conversion methods
Modified from
https://github.com/salesforce/cos-e/blob/master/code/generation/train_commonsenseqa_v1.0.py and ""_v1.11.py (identical)
as well as Tensorflow code for WTF?: 
https://github.com/google-research/google-research/blob/master/wt5/wt5/preprocessors.py
"""
# This code is based on https://github.com/allenai/label_rationale_association/blob/main/feature_conversion_methods.py
label_mapping_t5 = {
    "anli": {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    "efever": {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    "fm2": {0: 'entailment', 2: 'contradiction'},
    "covid_fact": {0: 'entailment', 2: 'contradiction'},
    'covert':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'vitaminc':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'snopes_stance':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'ambifc':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'pubhealth':{0: 'entailment', 1: 'neutral', 2: 'contradiction', 3: 'mixture'},
    'averitec':{0: 'entailment', 1: 'neutral', 2: 'contradiction', 3: 'mixture'},
    'scifact':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'climate-fever-combined':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'climate-fever-separate':{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'dialfact':{0: 'entailment', 1: 'neutral', 2: 'contradiction'}
}

label_mapping_flan_t5 = {
    "anli": {0: 'Yes', 1: 'It’s impossible to say', 2: 'No'},
    "efever": {0: 'Yes', 1: 'It’s impossible to say', 2: 'No'},
    "fm2": {0: 'Yes', 2: 'No'},
    "covid_fact": {0: 'Yes', 2: 'No'},
    'covert':{0: 'Yes', 1: 'It’s impossible to say', 2: 'No'},
    'vitaminc':{0: 'Yes', 1: 'It’s impossible to say', 2: 'No'},
    'ambifc':{0: 'Yes', 1: 'It’s impossible to say', 2: 'No'},
    'snopes_stance':{0: 'Yes', 1: 'It’s impossible to say', 2: 'No'},
    'climate-fever-combined':{0: 'Yes', 1: 'It’s impossible to say', 2: 'No'},
    'dialfact':{0: 'Yes', 1: 'It’s impossible to say', 2: 'No'}
}

def format_instance(
        example,
        task,
        tokenizer,
        explanation_sep,
        max_seq_length=None,
        io_format=None, 
):

    input_string, answer_string = formatting(example, task, explanation_sep, io_format)

    input_string = ' '.join(input_string.split())
    answer_string = ' '.join(answer_string.split())

    encodings = tokenizer.encode_plus(
        input_string,
        max_length=max_seq_length,        
        truncation=True, 
        padding=True,
#         return_token_type_ids=False,
        return_attention_mask=True,
    )

    # note even with "lm_labels.shift_right()", the decoder attention mask length is still correct since we remove the last token
    dec = tokenizer.encode_plus(
        answer_string,
        max_length=max_seq_length,
        truncation=True, 
        padding=True,
#         return_token_type_ids=False,
        return_attention_mask=True,
    )

    encodings["labels"] = dec["input_ids"]
    encodings["decoder_attention_mask"] = dec["attention_mask"]
#     encodings["question_encoding"] = encodings["input_ids"]
#     print(encodings)
    return encodings
#     return {**example, **encodings}


def formatting(item, dataset_name, explanation_sep=' "explanation: " ', io_format='standard', explanation_source='original'):

    premise = item["premise"]
    hypothesis = item["hypothesis"]
    if explanation_source=='original':
        expl = item["explanation_1"]
        
    elif explanation_source=='chatgpt':
        expl = item["explanation_gpt4"]

    elif explanation_source=='gpt-4':
        expl = item["explanation_gpt4"]

    elif explanation_source=='gpt-3.5':
        expl = item["explanation_gpt35"]

    elif explanation_source=='llama-3':
        expl = item["explanation_llama"]
    
    else:
        expl = item["explanation_1"]
        
    if io_format == 'standard':
        answer = label_mapping_t5[dataset_name][item["label"]]
        input_string = f"explain nli hypothesis: {hypothesis} premise: {premise}"
        answer_string = f"{answer} {explanation_sep} {expl}"
    
    if io_format == 'flan_t5':
        answer = label_mapping_flan_t5[dataset_name][item["label"]]
        input_string=f'Read the text and determine if the sentence is true (see options at the end)\nText: {premise} \nSentence: {hypothesis}\nOPTIONS:\n− Yes\n− It’s impossible to say\n− No\ntrue or false: \nLet\'s think step by step.\n'
        answer_string=f'{expl} {explanation_sep} {answer}'

    if io_format == 'no_exp':
        answer = label_mapping_t5[dataset_name][item["label"]]
        input_string = f"nli hypothesis: {hypothesis} premise: {premise}"
        answer_string = f"{answer}"
    return input_string, answer_string