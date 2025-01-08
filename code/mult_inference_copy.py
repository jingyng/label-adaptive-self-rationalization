# from scipy.stats import qmc
import argparse
import os
import subprocess
import queue
import time

n_gpus = [2]
gpu_queue = queue.Queue()

for i in n_gpus:
    gpu_queue.put(i)

n_shots = [8] #,2,4,8,16,32,64,128
# random_seed = [12,22,32,42,52]
# scheduler = ['linear', 'constant']
target_datasets = ['vitaminc']

# target_datasets = ['climate-fever-combined','snopes_stance','covid_fact',"scifact",'factcc','xsum_hallucination','qags_xsum','qags_cnndm']
# target_datasets = ['anli'] #"scifact",'factcc','xsum_hallucination','qags_xsum','qags_cnndm'
# selection_methods=['random','accept-ambiguous']
# target_datasets = ['wnli','add_one','glue_diagnostics','fm2','mpe','joci','hans','conj','dnc','sick','vitaminc'] #

running = []
for dataset in target_datasets:
    for ns in n_shots: 
        while len(running) >= len(n_gpus):
            for process in running:
                if process.poll() is not None:
                    gpu_queue.put(process.gpu_id)
                    running.remove(process)
                    print(f"Process {i} finished with return code {process.returncode}")
            time.sleep(30)

        gpu_id = gpu_queue.get()
    #     print('hparams', scaled)

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
        process = subprocess.Popen(["python", "Inference.py",
#                                         "--project_name",        'few-shot-inference-best_model',
                                    "--source_dataset_name", 'vitaminc',  
                                    "--target_dataset_name", dataset,  
                                    "--sample_selection",    "random",                                      
                                    "--test_bsz",            str(8),                                      
                                    "--n_shots",             str(ns),   
#                                     "--second_transfer",
                                   ], env=env)

        print(f"Process with dataset {dataset} started on GPU {gpu_id}")
        process.gpu_id = gpu_id
        running.append(process)