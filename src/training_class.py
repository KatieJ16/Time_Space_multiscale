import numpy as np
import matplotlib.pyplot as plt
import torch

import utils

#====================================================================================

class training_class():
    """
        defining a class that will do all the training and stuff.
    """

    def __init__(self, data_dir, model_dir, result_dir, tol):

        self.data_dir = data_dir
        self.model_dir = model_dir
        self.result_dir = result_dir

        self.train_dict, self.val_dict = utils.load_and_make_dict(data_dir)

        self.unresolved_dict = {}
        self.model_keep = list()
        self.model_used_dict = {}

        self.tol = tol
# ==================================================================================
def train_one_step(self, current_size, make_new=False, dont_train=True, verbose=True):
    """
        train 1 level
    """
    models, step_sizes, mse_list, idx_lowest, n_forward_list = utils.find_best_timestep(self.train_dict[str(current_size)],
                                                              self.val_dict[str(current_size)],
                                                              self.val_dict[str(current_size)],
                                                              current_size, model_dir=self.model_dir, make_new=make_new,
                                                              start_k=2, largest_k=4, dont_train=dont_train)

    # print if verbose
    if verbose:
        print("best step size = ", step_sizes[idx_lowest])
        print("step_sizes = ", step_sizes)
        print("mse = ", mse_list)
        utils.plot_lowest_error(models[idx_lowest], i=0, title="step_size = " +str(step_sizes[idx_lowest]))

    resolved, loss, unresolved_list = utils.find_error_4(self.val_dict[str(current_size)],
                                                         models[idx_lowest], self.val_dict[str(current_size*2)],
                                                         plot=verbose, tol=self.tol)
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
        plt.imshow(next_train_data[0, 0])
        plt.colorbar()
        plt.title("Next data going to size "+ str(current_size*2))
        plt.show()

    self.model_keep.append(models[idx_lowest])
    print("appended")
    self.model_used_dict[str(current_size)] = [[len(self.model_keep)-1]]

    if verbose:
        print("model_used_dict = ", self.model_used_dict)
        print("number of models kept = ", len(self.model_keep))

    return self, resolved

def train_next_step(self, current_size, verbose=True, make_new=False, dont_train=True):
    """
        trains and does everything for 2nd iteration
    """

    train_data = self.unresolved_dict[str(int(current_size/2))] * self.train_dict[str(current_size)]

    model_idx_list = np.ones((current_size, current_size))*(-1) #start with all -1

    for i in range(current_size):
        for j in range(current_size):
            if verbose:
                print("i = ", i, ": j = ", j)
            data_this = train_data[:, :, i, j]
            if (torch.min(data_this) == 0) and (torch.max(data_this) == 0):
                print("zero, no need to train")
                #don't need to do anything is model is resolved
                continue
            else:
            #see if the error is low enough on already made model
                for m, model in enumerate(self.model_keep):
                    loss, resolved = utils.find_error_1(data_this, model, tol=self.tol)
                    step_size = model.step_size
                    if verbose:
                        print("model ", m, " has loss = ", loss)
                    if resolved:
                        model_idx_list[i, j] = m
                        print("Resolved with loss = ", loss, ": model #", m)
                        break
                    else:
                        pass
                if not resolved:
                    if verbose:
                        print("not resolved, fitting new model")
                    k = int(np.log2(step_size))
                    #if no model good, train new model
                    models, step_sizes, mse_list, idx_lowest, n_forward_list = utils.find_best_timestep(self.train_dict[str(current_size)][:, :, i, j],
                                                                  self.val_dict[str(current_size)][:, :, i, j],
                                                                  self.val_dict[str(current_size)][:, :, i, j],
                                                                  current_size, model_dir=self.model_dir, make_new=make_new,
                                                                  i=i, j=j, start_k=2, largest_k=3, dont_train=dont_train)

                    self.model_keep.append(models[idx_lowest])
                    model_idx_list[i, j] = len(self.model_keep)-1
    self.model_used_dict[str(current_size)] = model_idx_list
    if verbose:
        print("self.model_used_dict[str(current_size)] = ", self.model_used_dict[str(current_size)])

    #find errors
    #once we have all 4 figured out, need to check the errors on the 2x2 of the 2x2s (the 4x4)
    unresolved_list_big = np.ones((current_size*2, current_size*2))*(-1)
    loss_big = np.ones((current_size*2, current_size*2))*(-1)
    all_resolved = True
    width = 2
    for i in range(current_size):
        for j in range(current_size):
            print("i = ", i, ": j = ", j)
#             print(self.model_used_dict[str(current_size)][i][j])
            model = self.model_keep[int(self.model_used_dict[str(current_size)][i][j])]
            data_next = self.val_dict[str(current_size*2)][:, :, i*width:(i+1)*width, j*width:(j+1)*width]
            resolved, loss, unresolved_list = utils.find_error_4(self.val_dict[str(current_size)][:, :, i, j],
                                            model, data_next, plot=verbose, tol=self.tol)
            unresolved_list_big[i*width:(i+1)*width, j*width:(j+1)*width] = unresolved_list
            loss_big[i*width:(i+1)*width, j*width:(j+1)*width] = loss
            if not resolved:
                all_resolved = False
            if verbose:
                print(loss)
                print(unresolved_list)
    if verbose:
        print("all_resolved = ", all_resolved)
        print("unresolved_list_big = ", unresolved_list_big)
        print("loss_big = ", loss_big)
        plt.figure()
        plt.imshow(np.log10(loss_big), cmap='Greys')
        plt.colorbar()
        plt.title("log(loss)")
        plt.show()
    self.unresolved_dict[str(current_size)] = torch.tensor(unresolved_list_big)
    if verbose:
        print("unresolved_dict = ", self.unresolved_dict)

    return self, all_resolved
