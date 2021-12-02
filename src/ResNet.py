import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.interpolate
# from utils_time2 import DataSet
import utils


print("using new ResNet thing")
class NNBlock(torch.nn.Module):
    def __init__(self, arch, activation=torch.nn.ReLU(inplace=False)):
        """
        :param arch: architecture of the nn_block
        :param activation: activation function
        """
        super(NNBlock, self).__init__()

        # param
        self.n_layers = len(arch)-1
        self.activation = activation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # network arch
        # print("arch -= ", arch)
        for i in range(self.n_layers):
            self.add_module('Linear_{}'.format(i), torch.nn.Linear(arch[i], arch[i+1]).to(self.device))

    def forward(self, x):
        """
        :param x: input of nn
        :return: output of nn
        """
        for i in range(self.n_layers - 1):
            x = self.activation(self._modules['Linear_{}'.format(i)](x))
        x = self._modules['Linear_{}'.format(self.n_layers - 1)](x)
        return x

class ResNet(torch.nn.Module):
    def __init__(self, train_data, val_data,
        step_size = 1, dim = 3, out_dim = 1,n_hidden_nodes=20, n_hidden_layers=5,
        model_name="model.pt", activation=nn.ReLU(), n_epochs = 1000, threshold = 1e-8):

        super(ResNet, self).__init__()

        self.step_size = step_size
        self.model_name = model_name

        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.n_epochs = n_epochs
        self.threshold = threshold

        self.train_data = train_data
        self.val_data = val_data
        self.inputs, self.outputs = form_data(train_data, step_size)
        self.val_inputs, self.val_outputs = form_data(val_data, step_size)

        self.hidden = nn.Linear(dim, n_hidden_nodes)   # hidden layer
        for i in range(self.n_hidden_layers):
            self.add_module('Linear_{}'.format(i), torch.nn.Linear(n_hidden_nodes, n_hidden_nodes))
        self.predict = nn.Linear(n_hidden_nodes, out_dim)   # output layer


        self.activation = activation

    def forward(self, x):
        #relu
        x = self.activation(self.hidden(x))#.float()))      # activation function for hidden layer
        for i in range(self.n_hidden_layers):
            x = self.activation(self._modules['Linear_{}'.format(i)](x))
        x = self.predict(x)             # linear output
        return x


    def train_model(self,optimizer, loss_func, n_no_improve = 1000):

        #train until val loss min (doesn't improve in n_no_improve)
        min_val_loss = torch.tensor(1e5) #some big number
        going = True
        epoch = 0
        no_improve = 0
        print_every = 100
        while going:
        # for epoch in range(self.n_epochs):
            epoch += 1
            outputs = self.outputs.reshape(-1, 1)

            prediction = self.forward(self.inputs.float())

            loss = loss_func(prediction.float(), outputs.float())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #check validation and save if needed
            if epoch % print_every == 0:
                if self.val_inputs is not None:
                    val_pred = self.forward(self.val_inputs.float())
                    val_loss = loss_func(val_pred.float(), self.val_outputs.float())
                    print("epoch ", epoch, ": train_error: ", loss.detach().numpy(), ": val_loss ", val_loss.detach().numpy(), ": min_val_loss ", min_val_loss.detach().numpy())
                    if val_loss < min_val_loss:
                        no_improve = 0
                        min_val_loss = val_loss
                        torch.save(self, self.model_name)
                        print("Model improved. Saved at epoch = ", epoch)
                        if val_loss < self.threshold:
                            print("Threshold of ", self.threshold, "met. Stop training")
                            return
                    else:

                        no_improve += print_every
                        if no_improve >= n_no_improve:
                            return
        return

    def predict_mse(self, data=None):
        mse_list = np.zeros(10)
        pred_list_all = []
        for num in range(10):
            if data is None:
                #use data in model if none imported
                data = self.val_data[:,::self.step_size]

            inputs = torch.cat((data[num,:-3,0], data[num,1:-2,0], data[num,2:-1,0]), axis = 1)

            y_pred = self.forward(inputs[0:3].float())
            y_pred = torch.cat((inputs[0:3,0:2].float(),y_pred), axis = 1)
            pred = [y_pred.detach().numpy()[0,0]]
            for i in range(len(inputs[:,0])-1):
                y_next = self.forward(y_pred)
                y_next = torch.cat((y_pred[:, 1:3],y_next), axis = 1)
                pred.append(y_next.detach().numpy()[0,0])
                y_pred = y_next

            mse = np.mean((np.array(pred) - inputs[:,0].detach().numpy())**2)
            mse_list[num] = mse
            pred_list_all.append(pred)
        return np.array(pred_list_all), np.mean(mse_list)

    def uni_scale_forecast(self, x_init, n_steps, interpolate = True):
        """
        :param x_init: array of shape n_test x input_dim
        :param n_steps: number of steps forward in terms of dt
        :return: predictions of shape n_test x n_steps x input_dim and the steps
        """
        print("x_init shape = ", x_init.shape)
        steps = list()
        preds = list()
        sample_steps = range(n_steps)

        # forward predictions
        x_prev = x_init
        cur_step = self.step_size - 1
        while cur_step < n_steps + self.step_size:
            x_next = self.forward(x_prev)
            steps.append(cur_step)
            preds.append(x_next)
            cur_step += self.step_size
            x_prev = x_next

        # include the initial frame
        steps.insert(0, 0)
        preds.insert(0, torch.tensor(x_init).float().to(self.device))

        # interpolations
        preds = torch.stack(preds, 2).detach().numpy()

        cs = scipy.interpolate.interp1d(steps, preds, kind='linear')
        y_preds = torch.tensor(cs(sample_steps)).transpose(1, 2).float()

        return y_preds

#     def train_net(self, dataset, max_epoch, batch_size, w=1.0, lr=1e-3, model_path=None, threshold = 1e-8, print_every=1000):
#         """
#         :param dataset: a dataset object
#         :param max_epoch: maximum number of epochs
#         :param batch_size: batch size
#         :param w: l2 error weight
#         :param lr: learning rate
#         :param model_path: path to save the model
#         :return: None
#         """
#         # check consistency
#         self.check_data_info(dataset)

#         # training
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#         epoch = 0
#         best_loss = 1e+5
#         while epoch < max_epoch:
#             epoch += 1
#             # ================= prepare data ==================
#             n_samples = dataset.n_train
#             new_idxs = torch.randperm(n_samples)
#             batch_x = dataset.train_x[new_idxs[:batch_size], :]
#             batch_ys = dataset.train_ys[new_idxs[:batch_size], :, :]
#             # =============== calculate losses ================
#             train_loss = self.calculate_loss(batch_x, batch_ys, w=w)
#             # print("train_loss = ", train_loss)
#             val_loss = self.calculate_loss(dataset.val_x, dataset.val_ys, w=w)
#             # ================ early stopping =================
#             if best_loss <= threshold:
#                 print('--> model has reached an accuracy of ', threshold, '! Finished training!')
#                 break
#             # =================== backward ====================
#             optimizer.zero_grad()
#             train_loss.backward()#retain_graph=True)
#             # print("step")
#             optimizer.step()
#             # =================== log =========================
#             if epoch % print_every == 0:
#                 print('epoch {}, training loss {}, validation loss {}'.format(epoch, train_loss.item(),
#                                                                               val_loss.item()))
#                 if val_loss.item() < best_loss:
#                     best_loss = val_loss.item()
#                     if model_path is not None:
#                         print('(--> new model saved @ epoch {})'.format(epoch))
#                         torch.save(self, model_path)

#         # if to save at the end
#         if val_loss.item() < best_loss and model_path is not None:
#             print('--> new model saved @ epoch {}'.format(epoch))
#             torch.save(self, model_path)

    def calculate_loss(self, x, ys, w=1.0):
        """
        :param x: x batch, array of size batch_size x n_dim
        :param ys: ys batch, array of size batch_size x n_steps x n_dim
        :return: overall loss
        """
        batch_size, n_steps, n_dim = ys.size()
        # print("x shape in loss funct = ", x.shape)
#         assert n_dim == self.n_dim
        with torch.autograd.set_detect_anomaly(True):
            # forward (recurrence)
            y_preds = torch.zeros(batch_size, n_steps, n_dim*2).float().to(self.device)
            y_prev = x.clone()#[:,0]
    #         y_prev_1 = x[:,1]
            for t in range(n_steps-1):
                # print("y_prev = ", y_prev.shape)
    #             ghj
                y_next = self.forward(y_prev)
                # print("y_next shape = ", y_next.shape)
                y_preds[:, t, :] = y_next.clone()
            # for i in range(len(y_prev)-1):
                y_prev = torch.cat((y_prev[:,1:2].clone(),y_next[:,0:1].clone()), axis = 1)
                # y_prev[:,0] = y_prev[:,1].clone()
                # y_prev[:,1] = y_next[:,0].clone()

            # compute loss
            criterion = torch.nn.MSELoss(reduction='none')
            loss = w * criterion(y_preds, ys).mean() + (1-w) * criterion(y_preds, ys).max()

        return loss


def form_data(data, step_size = 1):
    """
    Forms data to input to network.

    inputs:
        data: torch. shape, (n_points, n_timesteps, 1, 1)
        step_size: int

    outputs:
        inputs, torch shape (max_points, 3)
        outputs, torch shape (max_points, 1)
    """
    print("data shape = ", data.shape)
    train_data = data[:,::step_size]
    inputs = torch.cat((train_data[:,:-3,0], train_data[:,1:-2,0], train_data[:,2:-1,0]), axis = 2)
    inputs = torch.flatten(inputs, end_dim=1)
    outputs = train_data[:,3:,0]
    outputs = torch.flatten(outputs, end_dim=1)

    return inputs, outputs


def multi_scale_forecast(x_init, n_steps, models):
    """
    :param x_init: initial state torch array of shape n_test x n_dim
    :param n_steps: number of steps forward in terms of dt
    :param models: a list of models
    :return: a torch array of size n_test x n_steps x n_dim

    This function is not used in the paper for low efficiency,
    we suggest to use vectorized_multi_scale_forecast() below.
    """
    # sort models by their step sizes (decreasing order)
    step_sizes = [model.step_size for model in models]
    models = [model for _, model in sorted(zip(step_sizes, models), reverse=True)]

    # parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_extended_steps = n_steps + min(step_sizes)
    sample_steps = range(1, n_steps+1)

    steps = list()
    preds = list()
    steps.insert(0, 0)
    preds.insert(0, torch.tensor(x_init).float().to(device))
    for model in models:
        tmp_steps = list()
        tmp_preds = list()
        for j in range(len(steps)):
            if j < len(steps) - 1:
                end_step = steps[j+1]
            else:
                end_step = n_extended_steps
            # starting point
            cur_step = steps[j]
            cur_x = preds[j]
            tmp_steps.append(cur_step)
            tmp_preds.append(cur_x)
            while True:
                step_size = model.step_size
                cur_step += step_size
                if cur_step >= end_step:
                    break
                cur_x = model(cur_x)
                tmp_steps.append(cur_step)
                tmp_preds.append(cur_x)
        # update new predictions
        steps = tmp_steps
        preds = tmp_preds

    # interpolation
    preds = torch.stack(preds, 2).detach().numpy()
    cs = scipy.interpolate.interp1d(steps, preds, kind='linear')
    y_preds = torch.tensor(cs(sample_steps)).transpose(1, 2).float()

    return y_preds


def vectorized_multi_scale_forecast(x_init, n_steps, models):
    """
    :param x_init: initial state torch array of shape n_test x n_dim
    :param n_steps: number of steps forward in terms of dt
    :param models: a list of models
    :return: a torch array of size n_test x n_steps x n_dim,
             a list of indices that are not achieved by interpolations
    """
    # sort models by their step sizes (decreasing order)
    step_sizes = [model.step_size for model in models]
    models = [model for _, model in sorted(zip(step_sizes, models), reverse=True)]

    # we assume models are sorted by their step sizes (decreasing order)
    n_test, n_dim = x_init.shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    indices = list()
    extended_n_steps = n_steps + models[0].step_size
    preds = torch.zeros(n_test, extended_n_steps + 1, n_dim).float().to(device)

    # vectorized simulation
    indices.append(0)
    preds[:, 0, :] = x_init
    total_step_sizes = n_steps
    for model in models:
        n_forward = int(total_step_sizes/model.step_size)
        y_prev = preds[:, indices, :].reshape(-1, n_dim)
        indices_lists = [indices]
        for t in range(n_forward):
            y_next = model(y_prev)
            shifted_indices = [x + (t + 1) * model.step_size for x in indices]
            indices_lists.append(shifted_indices)
            preds[:, shifted_indices, :] = y_next.reshape(n_test, -1, n_dim)
            y_prev = y_next
        indices = [val for tup in zip(*indices_lists) for val in tup]
        total_step_sizes = model.step_size - 1

    # simulate the tails
    last_idx = indices[-1]
    y_prev = preds[:, last_idx, :]
    while last_idx < n_steps:
        last_idx += models[-1].step_size
        y_next = models[-1](y_prev)
        preds[:, last_idx, :] = y_next
        indices.append(last_idx)
        y_prev = y_next

    # interpolations
    sample_steps = range(1, n_steps+1)
    valid_preds = preds[:, indices, :].detach().numpy()
    cs = scipy.interpolate.interp1d(indices, valid_preds, kind='linear', axis=1)
    y_preds = torch.tensor(cs(sample_steps)).float()

    return y_preds
