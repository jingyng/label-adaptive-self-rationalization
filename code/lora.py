from load_custom_dataset import load_raw_data, load_format_data
from training import format_data, SequenceCollator
import torch
import datasets
from copy import deepcopy 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
print(os.getenv('CUDA_VISIBLE_DEVICES'))

#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#print(torch.cuda.device_count())
#print(torch.cuda.current_device())
#print(torch.cuda.get_device_name(0))

# huggingface hub model id
#model_id = "philschmid/flan-t5-xxl-sharded-fp16"

# load model from the hub
#model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="sequential")
#model_id="google/flan-t5-xxl"
