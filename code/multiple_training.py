import argparse
import os
import time
import subprocess

n_shots = [8]

source_dataset_name='anli'


for ns in n_shots:
    process = subprocess.run(["python", "training.py",
                    "--project_name",        'double-transfer-anli',
                    "--source_dataset_name", source_dataset_name,
                    "--n_shots",             str(ns),
                    "--sample_selection",    'random'    
                    "--num_epochs",          str(10),
                   ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

