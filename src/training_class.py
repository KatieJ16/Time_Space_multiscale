import numpy as np
import matplotlib.pyplot as plt
import torch

import utils

#====================================================================================

class training_class():
    """
        defining a class that will do all the training and stuff.
    """

    def __init__(self, data_dir, model_dir, result_dir, train_threshold=1e-8, resolve_tol=1e-4, n_inputs=3, device=None):

        self.data_dir = data_dir
        self.model_dir = model_dir
        self.result_dir = result_dir

        print("given device = ", device)
        if device is None:
            print("None")
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print("device = ", self.device)

        self.train_dict, self.val_dict = utils.load_and_make_dict(data_dir, device=self.device)

        self.unresolved_dict = {}
        self.model_keep = list()
        self.model_used_dict = {}

        self.train_threshold = train_threshold
        self.resolve_tol = resolve_tol
        self.n_inputs = n_inputs
# ==================================================================================
def train_one_step(self, current_size, make_new=False, dont_train=True, verbose=True,
                   start_k=2, largest_k=4, plot_all_timesteps=False):
    """
        train 1 level
    """
    output = utils.find_best_timestep(self.train_dict[str(current_size)],
                                      self.val_dict[str(current_size)],
                                      self.val_dict[str(current_size)],
                                      current_size, model_dir=self.model_dir,
                                      make_new=make_new, start_k=start_k,
                                      largest_k=largest_k,
                                      dont_train=dont_train,
                                      n_inputs=self.n_inputs, threshold=self.train_threshold)
    models, step_sizes, mse_list, idx_lowest, n_forward_list = output
    # print if verbose
    if verbose:
        print("best step size = ", step_sizes[idx_lowest])
        print("step_sizes = ", step_sizes)
        print("mse = ", mse_list)
        if plot_all_timesteps:
            for i in range(len(models)):
                utils.plot_lowest_error(models[i], i=0, title="step_size = " +str(step_sizes[i]))
        else:
            utils.plot_lowest_error(models[idx_lowest], i=0, title="step_size = " +str(step_sizes[idx_lowest]))


    resolved, loss, unresolved_list = utils.find_error_4(self.val_dict[str(current_size)],
                                                         models[idx_lowest], self.val_dict[str(current_size*2)],
                                                         plot=verbose, tol=self.resolve_tol)
    if verbose:
        print("loss shape = ", loss.shape)
        print("loss = ", loss)

        #plot the errors
        plt.figure()
        plt.imshow(np.log10(loss), cmap='Greys')
        plt.title("log loss of refinement at size " + str(current_size))
        plt.colorbar()
        plt.show()

    self.unresolved_dict[str(current_size)] = torch.tensor(unresolved_list).to(self.device)

    if verbose:
        print("unresolved_list = ", unresolved_list)
        if resolved:
            print("Resolved!!")

    #set up next step
    next_train_data = self.unresolved_dict[str(current_size)] * self.train_dict[str(current_size*2)]
    if verbose:
        print(next_train_data.shape)
        plt.imshow(next_train_data[0, 0].cpu())
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
# ==================================================================================
def train_next_step(self, current_size, verbose=True, make_new=False,
                    dont_train=True, start_k=2, largest_k=3, plot_fit=False, result_dir='../result/'):
    """
        trains and does everything for 2nd iteration
    """

    if verbose:
        print("self.unresolved_dict[str(int(current_size/2))]= ", self.unresolved_dict[str(int(current_size/2))].is_cuda)
        print("self.train_dict[str(current_size)] = ", self.train_dict[str(current_size)].is_cuda)
    train_data = self.unresolved_dict[str(int(current_size/2))] * self.train_dict[str(current_size)]

    model_idx_list = np.ones((current_size, current_size))*(-1) #start with all -1

    loss_small = np.ones((current_size, current_size))*(-1)
    for i in range(current_size):
        for j in range(current_size):
            if verbose:
                print("i = ", i, ": j = ", j)
            data_this = train_data[:, :, i, j]
            if (torch.min(data_this) == 0) and (torch.max(data_this) == 0):
                print("zero, no need to train")
                try:
                    model_used = self.model_used_dict[str(int(current_size/2))][i//2, j//2]
                except:
                    model_used = self.model_used_dict[str(int(current_size/2))][i//2][j//2]
                print("saved model is ", model_used)
                model_idx_list[i, j] = model_used
                #don't need to do anything is model is resolved
                continue
            else:
            #see if the error is low enough on already made model
                mse_each_model_list = []
                for m, model in enumerate(self.model_keep):
                    loss, resolved = utils.find_error_1(data_this, model, tol=self.resolve_tol)
                    step_size = model.step_size
                    mse_each_model_list.append(loss)
                    if verbose:
                        print("model ", m, " has loss = ", loss)
                    if resolved:
                        model_idx_list[i, j] = m
                        loss_small[i,j] = loss
                        print("Resolved with loss = ", loss, ": model #", m)
                        break

                if not resolved:
                    if verbose:
                        print("not resolved, fitting new model")
                    #if no model good, train new model
                    output = utils.find_best_timestep(self.train_dict[str(current_size)][:, :, i, j],
                                                      self.val_dict[str(current_size)][:, :, i, j],
                                                      self.val_dict[str(current_size)][:, :, i, j],
                                                      current_size, model_dir=self.model_dir, make_new=make_new,
                                                      i=i, j=j, start_k=start_k, largest_k=largest_k,
                                                      dont_train=dont_train, n_inputs=self.n_inputs)
                    models, _, mse_list, idx_lowest, _ = output
                    print("mse_list = ", mse_list)
                    file_name = models[idx_lowest].model_name+"_fitting_error_i"+str(i)+"_j"+str(j)".pdf"
                    loss, resolved = utils.find_error_1(self.val_dict[str(current_size)][:, :, i, j],
                                                        models[idx_lowest], tol=self.resolve_tol, plot=plot_fit,# i=i, j=j,
                                                        title="Error of fitting at i = " +str(i)+": j = "+str(j),
                                                        file_name=os.path.join(result_dir, file_name))

                    print("Error of added model is: ", loss)
                    mse_each_model_list.append(loss)
                    print("mse_each_model_list.append(loss) = ", mse_each_model_list)

                    idx_best_model = np.argmin(np.array(mse_each_model_list))
                    loss_small[i, j] = mse_each_model_list[idx_best_model]
                    print("best model found was ", idx_best_model)

                    #if model that was just made if the best, add to list
                    if idx_best_model == len(self.model_keep):
                        self.model_keep.append(models[idx_lowest])
                    else: #plot a graph with the best model
                    file_name = self.model_keep[idx_best_model].model_name+"_fitting_error_i"+str(i)+"_j"+str(j)"_actual_best.pdf"
                    loss, resolved = utils.find_error_1(self.val_dict[str(current_size)][:, :, i, j],
                                                        self.model_keep[idx_best_model], tol=self.resolve_tol, plot=plot_fit,# i=i, j=j,
                                                        title="Error of fitting at i = " +str(i)+": j = "+str(j),
                                                        file_name=os.path.join(result_dir, file_name))

                    model_idx_list[i, j] = idx_best_model


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
            model = self.model_keep[int(self.model_used_dict[str(current_size)][i][j])]
            #use 1 more refinement to find error. If finest model, just compare to self
            try:
                data_next = self.val_dict[str(current_size*2)][:, :, i*width:(i+1)*width, j*width:(j+1)*width]
            except: #when finest resolution
                data_next = self.val_dict[str(current_size)][:, :, i, j]
            resolved, loss, unresolved_list = utils.find_error_4(self.val_dict[str(current_size)][:, :, i, j],
                                                                 model, data_next, plot=verbose, tol=self.resolve_tol)
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
        print("loss_small = ", loss_small)
        plt.figure()
        try:
            plt.imshow(np.log10(loss_small).cpu(), cmap='Greys')
        except:
            plt.imshow(np.log10(loss_small), cmap='Greys')
        plt.colorbar()
        plt.title("log loss at size " + str(current_size))
        plt.show()

        print("loss_big = ", loss_big)
        plt.figure()
        try:
            plt.imshow(np.log10(loss_big).cpu(), cmap='Greys')
        except:
            plt.imshow(np.log10(loss_big), cmap='Greys')
        plt.colorbar()
        plt.title("log loss of refinement at size " + str(current_size))
        plt.show()


    self.unresolved_dict[str(current_size)] = torch.tensor(unresolved_list_big).to(self.device)
    if verbose:
        print("unresolved_dict = ", self.unresolved_dict)

    return self, all_resolved
