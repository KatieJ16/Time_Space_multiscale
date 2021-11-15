import torch
import random
import numpy as np


class DataSet:
    def __init__(self, train_data, val_data, test_data, dt, step_size, n_forward,n_input_points=2):
        """
        :param train_data: array of shape n_train x train_steps x input_dim
                            where train_steps = max_step x (n_steps + 1)
        :param val_data: array of shape n_val x val_steps x input_dim
        :param test_data: array of shape n_test x test_steps x input_dim
        :param dt: the unit time step
        :param step_size: an integer indicating the step sizes
        :param n_forward: number of steps forward
        """
        n_train, train_steps, n_dim = train_data.shape
        n_val, val_steps, _ = val_data.shape
        n_test, test_steps, _ = test_data.shape
        assert step_size*n_forward+1 <= train_steps and step_size*n_forward+1 <= val_steps

        # params
        self.dt = dt
        self.n_dim = n_dim *2
        self.step_size = step_size
        print("step_size = ", step_size)
        self.n_forward = n_forward
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.n_input_points = n_input_points

        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        # data
        x_idx = 0
        x_end_idx = x_idx + step_size*(self.n_input_points-1)+1
        print("x_end_idx = ", x_end_idx)
        #y_starts after x ends
        y_start_idx = step_size*(self.n_input_points)# + step_size
        print("y_start_idx = ", y_start_idx)
        y_end_idx = y_start_idx + step_size*n_forward + 1
        print("y_end_idx = ", y_end_idx)
        print(range(x_idx,x_end_idx,step_size))


#         self.train_x = torch.tensor(train_data[:, x_idx, :]).float().to(self.device)
        self.train_x = torch.tensor(train_data[:, x_idx:x_end_idx:step_size, :]).float().to(self.device)
        self.train_x = torch.flatten(self.train_x, start_dim = 1)
        print("self.train_x shape = ", self.train_x.shape)

        self.train_ys = torch.tensor(train_data[:, y_start_idx:y_end_idx:step_size, :]).float().to(self.device)
        # self.train_ys = torch.flatten(self.train_ys, start_dim = 1)
        print("train_ys shape = ", self.train_ys.shape)

#         self.val_x = torch.tensor(val_data[:, x_idx, :]).float().to(self.device)
        self.val_x = torch.tensor(val_data[:, x_idx:x_end_idx:step_size, :]).float().to(self.device)
        self.val_x = torch.flatten(self.val_x, start_dim = 1)
        self.val_ys = torch.tensor(val_data[:, y_start_idx:y_end_idx:step_size, :]).float().to(self.device)
        # self.val_ys = torch.flatten(self.val_ys, start_dim = 1)
        self.test_x = torch.tensor(test_data[:, 0, :]).float().to(self.device)
        self.test_ys = torch.tensor(test_data[:, 1:, :]).float().to(self.device)

def make_data_longer(data,n_points,n_repeats, points_needed, n_dim):
    #need to reform training data to be 700,63,1
    data_repeats = torch.zeros((n_points*n_repeats, points_needed, n_dim))
    for i in range(n_repeats):

        data_repeats[i*n_points:(i+1)*n_points, :,:] = data[:,i*points_needed:(i+1)*points_needed,:]

    return(data_repeats)
def make_dataset_max_repeat(train_data, val_data, test_data, dt, step_size, n_forward=5,n_input_points=2):
    max_step = n_forward + n_input_points

    points_needed = max_step *(step_size + 1)
    n_points, n_timesteps, n_dim = train_data.shape
    n_val_points,n_timesteps_val, _ = val_data.shape
    n_repeats = int(n_timesteps/points_needed)
    n_repeats_val = int(n_timesteps_val/points_needed)

    #need to reform training data to be 700,63,1
    train_data_repeats = make_data_longer(train_data,n_points,n_repeats, points_needed, n_dim)
    val_data_repeats = make_data_longer(val_data,n_val_points,n_repeats_val, points_needed, n_dim)
    print("train_data_repeats shape = ", train_data_repeats.shape)
    dataset = DataSet(train_data_repeats,val_data_repeats, test_data,dt, step_size, n_forward,n_input_points)

    return dataset