import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch


import ResNet as tnet
import utils

#====================================================================================

class training_class():
    """
        defining a class that will do all the training and stuff.
    """

    def __init__(self, data_dir, model_dir, result_dir):

        self.data_dir = data_dir
        self.model_dir = model_dir
        self.result_dir = result_dir

        self.train_dict, self.val_dict = utils.load_and_make_dict(data_dir)

        self.unresolved_dict = {}
        self.model_keep = list()
        self.model_used_dict = {}

        self.tol = 0.0003

    def train_one_step(self, current_size, make_new=False, dont_train=True, verbose=True):
        """
            train 1 level
        """
        models, step_sizes, mse_list, idx_lowest,n_forward_list = utils.find_best_timestep(self.train_dict[str(current_size)],
                                                                  self.val_dict[str(current_size)],
                                                                  self.val_dict[str(current_size)], current_size, model_dir=self.model_dir, make_new=make_new,
                                                                 start_k = 2,largest_k = 4, dont_train = dont_train)

        # print if verbose
        if verbose:
            print("best step size = ", step_sizes[idx_lowest])
            print("step_sizes = ", step_sizes)
            print("mse = ", mse_list)
            utils.plot_lowest_error(models[idx_lowest], i=0, title="step_size = " +str(step_sizes[idx_lowest]))

        resolved, loss, unresolved_list = utils.find_error_4(self.val_dict[str(current_size)], models[idx_lowest], val_dict[str(current_size*2)], plot=verbose,tol=self.tol)
        if verbose:
            print("loss shape = ", loss.shape)
            print("loss = ", loss)
