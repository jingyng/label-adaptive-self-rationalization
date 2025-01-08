# from scipy.stats import qmc
import argparse
import os
import subprocess
import queue
import time

n_gpus = [4,5,6]
gpu_queue = queue.Queue()

for i in n_gpus:
    gpu_queue.put(i)

lora_alpha=[16,64,256]
learning_rate=[2e-4,2e-5]

running = []
for alpha in lora_alpha:
    for lr in learning_rate: 
        while len(running) >= len(n_gpus):
            for process in running:
                if process.poll() is not None:
                    gpu_queue.put(process.gpu_id)
                    running.remove(process)
                    print(f"Process {i} finished with return code {process.returncode}")
            time.sleep(30)

        gpu_id = gpu_queue.get()

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
        process = subprocess.Popen(["python", "hyperparapeter_search.py",
                                        "--project_name",        'olmo-7b-hyperparameter-search',
                                    "--model_name",          'allenai/OLMo-1.7-7B-hf',
                                    "--source_dataset_name", 'esnli',  
                                    "--sample_selection",    "random",                                      
                                    "--lora_alpha",          str(alpha),
                                    "--io_format",           str(lr),
                                   ], env=env)

        print(f"Process with dataset {dataset} started on GPU {gpu_id}")
        process.gpu_id = gpu_id
        running.append(process)