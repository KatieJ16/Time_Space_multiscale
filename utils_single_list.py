import torch
import random
import numpy as np


class DataSet:
    def __init__(self, train_data, val_data, test_data, dt, step_size, n_forward):
        """
        :param train_data: array of shape n_train x train_steps x input_dim
                            where train_steps = max_step x (n_steps + 1)
        :param val_data: array of shape n_val x val_steps x input_dim
        :param test_data: array of shape n_test x test_steps x input_dim
        :param dt: the unit time step
        :param step_size: an integer indicating the step sizes
        :param n_forward: number of steps forward
        """
        n_train = 8440
        train_steps = 5
        n_dim = 1
        n_val = val_data.shape
        n_test = test_data.shape

#         n_train, train_steps, n_dim = train_data.shape
#         n_val, val_steps, _ = val_data.shape
#         n_test, test_steps, _ = test_data.shape
#         assert step_size*n_forward+1 <= train_steps and step_size*n_forward+1 <= val_steps

        # params
        self.dt = dt
        self.n_dim = n_dim
        self.step_size = step_size
        self.n_forward = n_forward
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test

        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # data
        x_idx_start = 0
        x_idx_end = x_idx_start + n_forward 
        y_start_idx = x_idx_end 
        y_end_idx = x_idx_end + n_forward 
        
        print("x_idx_start  =", x_idx_start)
        print("x_idx_end  =", x_idx_end)
        print("y_start_idx  =", y_start_idx)
        print("y_end_idx  =", y_end_idx)

        
        train = np.zeros((n_train, 2*train_steps+1, 1))
        print("train.shape = ", train.shape)
        print("train_data.shape = ", train_data.shape)
        print("2*step_size*n_forward = ", 2*step_size*n_forward)
        for i in range(n_train):
#             print("train[i, : ,0] shape = ", train[i, : ,0].shape)
#             print("train_data[i:i+y_end_idx :step_size]shape = ", train_data[i:i+y_end_idx :step_size].shape)
            train[i, : ,0] = train_data[i:i+y_end_idx :step_size]
            
        
        
        self.train_x = torch.tensor(train[:, x_idx_start:x_idx_end, :]).float().to(self.device)
        print("train-x shape = ", self.train_x.shape) 
        self.train_ys = torch.tensor(train[:, y_start_idx:y_end_idx, :]).float().to(self.device)
        print("train_ys shape = ", self.train_ys.shape) 
        
        val = np.zeros((n_train, 2*train_steps+1, 1))
        for i in range(n_train):
            val[i, : ,0] = val_data[i:i+y_end_idx :step_size]
            
        
        self.val_x = torch.tensor(val[:, x_idx_start:x_idx_end, :]).float().to(self.device)
        print("train-x shape = ", self.train_x.shape) 
        self.val_ys = torch.tensor(val[:, y_start_idx:y_end_idx, :]).float().to(self.device)
        print("val_xshape = ", self.val_x.shape) 
        print("val_ys shape = ", self.val_ys.shape) 
        
        test = np.zeros((n_train, 2*train_steps+1, 1))
        for i in range(n_train):
            test[i, : ,0] = test_data[i:i+y_end_idx :step_size]
            
        self.test_x = torch.tensor(test[:, x_idx_start:x_idx_end, :]).float().to(self.device)
        self.test_ys = torch.tensor(test[:, x_idx_end:, :]).float().to(self.device)
