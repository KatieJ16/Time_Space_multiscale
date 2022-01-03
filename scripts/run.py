# imports
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

module_path = os.path.abspath(os.path.join('../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)

import ResNet as tnet
import utils
import training_class as tc

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5, 0)

# paths
data_dir = '../data/toy5a'
model_dir = '../model/toy5a'
result_dir = '../result/toy5a'

obj = tc.training_class(data_dir, model_dir, result_dir, resolve_tol=1e-6, n_inputs=4)

#train the first step
obj, resolved = tc.train_one_step(obj, 1, verbose=True,start_k=2, largest_k=3, plot_all_timesteps=True, result_dir='../result/')

#train the second step
obj, resolved = tc.train_next_step(obj, 2, verbose=True,start_k=2, largest_k=3, plot_fit=True, result_dir='../result/')

#train the next step on 4x4
obj, resolved = tc.train_next_step(obj, 4, verbose=True,start_k=2, largest_k=3, plot_fit=True, result_dir='../result/')
