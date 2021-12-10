import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
import scipy.interpolate
import utils


print("using new ResNet thing")


class ResNet(torch.nn.Module):
    def __init__(self, train_data, val_data, step_size=1, out_dim=1,
                 n_hidden_nodes=20, n_hidden_layers=5, model_name="model.pt",
                 activation=nn.ReLU(), n_epochs=500000, threshold=1e-8,
                 n_inputs=3, print_every=1000, save_every=100):

        super(ResNet, self).__init__()


        self.step_size = step_size
        self.model_name = model_name
        self.n_inputs = n_inputs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device = ", self.device)

        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.n_epochs = n_epochs
        self.threshold = threshold
        self.save_every = save_every
        self.print_every = print_every

        self.train_data = train_data
        self.val_data = val_data
        self.inputs, self.outputs = self.form_data(train_data, step_size)
        self.val_inputs, self.val_outputs = self.form_data(val_data, step_size)

        self.hidden = nn.Linear(n_inputs, n_hidden_nodes)   # hidden layer
        for i in range(self.n_hidden_layers):
            self.add_module('Linear_{}'.format(i), torch.nn.Linear(n_hidden_nodes, n_hidden_nodes))
        self.predict = nn.Linear(n_hidden_nodes, out_dim)   # output layer


        self.activation = activation


    def forward(self, x):
        """ forward step for the ResNet NN """
        #relu
        x = self.activation(self.hidden(x))#.float()))      # activation function for hidden layer
        for i in range(self.n_hidden_layers):
            x = self.activation(self._modules['Linear_{}'.format(i)](x))
        x = self.predict(x)             # linear output
        return x


    def train_model(self, optimizer, loss_func, n_no_improve=1000):
        """
            trains model. will train until the validation error has not improved
            in n_no_improve steps

            inputs:
            optimizer: such as torch.optim.Adam
            loss_func: such as torch.nn.MSELoss()
            n_no_improve: (1000) number of step to keep training when validation
                error is not increasing

            outputs:
                no returned value, but model will be trained.
        """

        min_val_loss = torch.tensor(1e5) #some big number
        no_improve = 0
        for epoch in range(self.n_epochs):
            outputs = self.outputs.reshape(-1, 1)

            prediction = self.forward(self.inputs.float())

            loss = loss_func(prediction.float(), outputs.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #check validation and save if needed
            if epoch % self.save_every == 0:
                if self.val_inputs is not None:
                    val_pred = self.forward(self.val_inputs.float().to(self.device))
                    val_loss = loss_func(val_pred.float(), self.val_outputs.float())
                    if epoch % self.print_every == 0:
                        print("epoch ", epoch, ": train_error: ", loss.cpu().detach().numpy(), ": val_loss ", val_loss.cpu().detach().numpy(), ": min_val_loss ", min_val_loss.cpu().detach().numpy())
                    if val_loss < min_val_loss:
                        no_improve = 0
                        min_val_loss = val_loss
                        torch.save(self, self.model_name)
                        if val_loss < self.threshold:
                            print("Threshold of ", self.threshold, "met. Stop training")
                            return
                    else:

                        no_improve += self.save_every
                        if no_improve >= n_no_improve:
                            print("Model finished training at epoch ", epoch, " with loss ", min_val_loss.cpu().detach().numpy())
                            return

        return

    def predict_mse(self, data=None):
        """
            Predicts system in time and finds mse between predicted and truth

            inputs:
                data (None), data of size (n_points, n_timesteps, 1, 1) optional for truth data.
                default is the validation data of the model
            outputs:
                predicted: tensor of size (n_points, n_timesteps) for the predictions
                mse: float, mean squared error between predicted and truth

        """


        if data is None:
            #use data in model if none imported
            data = self.val_data[:, ::self.step_size]

        n_points, n_timesteps, _, _ = data.shape
        mse_list = np.zeros(n_points)
        pred_list_all = torch.ones(data.shape[:(self.n_inputs-1)])*(-1)
        for num in range(n_points):
            y_pred = data[num, :self.n_inputs, 0, 0].float().T.to(self.device)
            pred = y_pred

            for i in range(n_timesteps-self.n_inputs):
                y_next = self.forward(y_pred)
                pred = torch.cat((pred, y_next))
                y_next = torch.cat((y_pred[1:], y_next))


                y_pred = y_next

            mse = np.mean((pred.cpu().detach().numpy() - data[num, :, 0, 0].cpu().detach().numpy())**2)
            mse_list[num] = mse

            try:
                pred_list_all[num, :, 0] = pred
            except:
                pred_list_all[num, :] = pred

        return pred_list_all, np.mean(mse_list)

    def form_data(self, data, step_size=1, verbose=False):
        """
        Forms data to input to network.

        inputs:
            data: torch. shape, (n_points, n_timesteps, 1, 1)
            step_size: int

        outputs:
            inputs, torch shape (max_points, self.n_inputs)
            outputs, torch shape (max_points, 1)
        """
        if verbose:
            print("data shape = ", data.shape)

        train_data = data[:, ::step_size]
        train_data = torch.flatten(train_data, start_dim=2)
        if verbose:
            print("train_data ", train_data.shape)
        inputs = torch.cat((train_data[:, :-self.n_inputs],
                            train_data[:, 1:(-self.n_inputs+1)]), axis=2)
        for i in range(2, self.n_inputs):
            inputs = torch.cat((inputs, train_data[:, i:(-self.n_inputs+i)]), axis=2)
        inputs = torch.flatten(inputs, end_dim=1)
        outputs = train_data[:, self.n_inputs:]
        outputs = torch.flatten(outputs, end_dim=1)

        return inputs.to(self.device), outputs.to(self.device)
