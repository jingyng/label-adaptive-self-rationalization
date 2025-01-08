import argparse
import os
import subprocess
import queue
import time


n_gpus = [2,6,7]
gpu_queue = queue.Queue()

for i in n_gpus:
    gpu_queue.put(i)

n_shots = 16
sample_seed = [22,32,42]
source_dataset_name='averitec'
explanation_source=['llama-3'] #'chatgpt','gpt-4','gpt-3.5','original'
running = []
# for name in source_dataset_name:
for es in explanation_source:
    for ss in sample_seed:
        while len(running) >= len(n_gpus):
            for process in running:
                if process.poll() is not None:
                    gpu_queue.put(process.gpu_id)
                    running.remove(process)
                    print(f"Process {i} finished with return code {process.returncode}")
            time.sleep(30)
    
        gpu_id = gpu_queue.get()
    
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
        process = subprocess.Popen(["python", "training.py",
                        "--project_name",        'double-transfer-exp',
                        "--source_dataset_name", source_dataset_name,
                        "--target_dataset_name", source_dataset_name,           
                        "--n_shots",             str(n_shots),
                        "--max_length",          "512",
                        "--max_length_target",   "256",
                        "--train_bsz",           "4",
                        "--num_epochs",          "50",
                        "--sample_selection",    "random",
                        "--lr",                  "3e-5",
                        "--explanation_source",  es,
                        "--gradient_accumulation_steps", "1",
                        "--sample_seed",         str(ss),
                        "--second_transfer",            
                       ], env=env)
    
        print(f"Process with {es} as explanation source, random split {ss}, trained on {source_dataset_name} started on GPU {gpu_id}")
        process.gpu_id = gpu_id
        running.append(process)