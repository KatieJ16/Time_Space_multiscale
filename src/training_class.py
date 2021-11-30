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
# ==================================================================================
def train_one_step(self, current_size, make_new=False, dont_train=True, verbose=True):
    """
        train 1 level
    """
    models, step_sizes, mse_list, idx_lowest,n_forward_list = utils.find_best_timestep(self.train_dict[str(current_size)],
                                                              self.val_dict[str(current_size)],
                                                              self.val_dict[str(current_size)], 
                                                              current_size, model_dir=self.model_dir, make_new=make_new,
                                                              start_k = 2,largest_k = 4, dont_train = dont_train)

    # print if verbose
    if verbose:
        print("best step size = ", step_sizes[idx_lowest])
        print("step_sizes = ", step_sizes)
        print("mse = ", mse_list)
        utils.plot_lowest_error(models[idx_lowest], i=0, title="step_size = " +str(step_sizes[idx_lowest]))

    resolved, loss, unresolved_list = utils.find_error_4(self.val_dict[str(current_size)], 
                                                         models[idx_lowest], self.val_dict[str(current_size*2)],
                                                         plot=verbose,tol=self.tol)
    if verbose:
        print("loss shape = ", loss.shape)
        print("loss = ", loss)
        
        #plot the errors 
        plt.figure()
        plt.imshow(np.log10(loss))
        plt.title("log loss at size " + str(current_size))
        plt.colorbar()
        plt.show()
        
    self.unresolved_dict[str(current_size)] = torch.tensor(unresolved_list)

    if verbose:
        print("unresolved_list = ", unresolved_list)
        
#     return self, resolved 

    if resolved:
        print("Resolved!!")
        return
    
    #set up next step
    next_train_data = self.unresolved_dict[str(current_size)] * self.train_dict[str(current_size*2)]
    if verbose:
        print(next_train_data.shape)
        plt.imshow(next_train_data[0,0])
        plt.colorbar()
        plt.title("Next data going to size "+ str(current_size*2))
        plt.show()
        
    self.model_keep.append(models[idx_lowest])
    print("appended")
    self.model_used_dict[str(current_size)] = [[len(self.model_keep)-1]]

    if verbose:
        print("model_used_dict = ", self.model_used_dict)
        print("number of models kept = ", len(self.model_keep))
        
    return self,resolved

def train_next_step(self, current_size, verbose=True):
    """
        trains and does everything for 2nd iteration
    """
    
    train_data = self.unresolved_dict[str(int(current_size/2))] * self.train_dict[str(current_size)]

    model_idx_list = np.ones((current_size, current_size))*(-1) #start with all -1

    for i in range(current_size):
        for j in range(current_size):
            if verbose:
                print("i = ", i, ": j = ", j)
            data_this = train_data[:,:,i,j]
            if (torch.min(data_this) == 0) and (torch.max(data_this) == 0):
                print("zero, no need to train")
                #don't need to do anything is model is resolved
                continue
            else:
            #see if the error is low enough on already made model
                for m, model in enumerate(model_keep):
                    loss, resolved = utils.find_error_1(data_this, model,tol=0.0003)
                    step_size = model.step_size
                    print("model ", m, " has loss = ", loss)
                    if resolved:
                        model_idx_list[i,j] = m
                        print("Resolved with loss = ", loss, ": model #", m)
                        break
                    else:
                        pass
                if not resolved:
                    print("not resolved, fitting new model")
                    k = int(np.log2(step_size))
                    #if no model good, train new model
                    models, step_sizes, mse_list, idx_lowest, n_forward_list = utils.find_best_timestep(train_dict[str(current_size)][:,:,i,j], 
                                                                  val_dict[str(current_size)][:,:,i,j], 
                                                                  val_dict[str(current_size)][:,:,i,j], current_size,model_dir=model_dir,#make_new = True,
                                                                  i=i, j=j, start_k = 2, largest_k =3)#, dont_train=False)

                    model_keep.append(models[idx_lowest])
                    model_idx_list[i,j] = len(model_keep)-1
    model_used_dict[str(current_size)] = model_idx_list