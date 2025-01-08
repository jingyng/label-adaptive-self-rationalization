# from scipy.stats import qmc
import argparse
import os
import subprocess
import queue
import time

n_gpus = [6]
gpu_queue = queue.Queue()

for i in n_gpus:
    gpu_queue.put(i)

n_shots = 5000 #,2,4,8,16,32,64,128
sample_seed = [42]# 22,32,
# scheduler = ['linear', 'constant']
target_dataset = 'pubhealth'
explanation_source=['original'] #,'chatgpt',,'llama-3',,,'original','gpt-4','llama-3'

# target_datasets = ['climate-fever-combined','snopes_stance','covid_fact',"scifact",'factcc','xsum_hallucination','qags_xsum','qags_cnndm']
# target_datasets = ['anli'] #"scifact",'factcc','xsum_hallucination','qags_xsum','qags_cnndm'
# selection_methods=['random','accept-ambiguous']
# target_datasets = ['wnli','add_one','glue_diagnostics','fm2','mpe','joci','hans','conj','dnc','sick','vitaminc'] #

running = []
for ss in sample_seed:
    for es in explanation_source: 
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
                                    # "--model_name",          'google/flan-t5-xl',
                                    "--source_dataset_name", target_dataset,  
                                    "--target_dataset_name", target_dataset,  
                                    "--io_format",           'no_exp',
                                    "--sample_selection",    "random", 
                                    "--sample_seed",         str(ss),
                                    "--source_model_path",   "nt5000",
                                    "--explanation_source",  es,
                                    "--test_bsz",            str(2),                          
                                    "--n_shots",             str(n_shots), 
                                    "--data_split",          "test",
                                    # "--explanation_sep",     "[ANSWER]",
                                    # "--io_format",           "flan_t5",
                                    # "--second_transfer",
                                   ], env=env)

        print(f"Process with dataset {target_dataset} started on GPU {gpu_id}")
        process.gpu_id = gpu_id
        running.append(process)