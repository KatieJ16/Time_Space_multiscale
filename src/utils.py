import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch


import ResNet as tnet

print("reloaded")

#====================================================================================
def isPowerOfTwo(n):
    """
    checks if n is a power of two

    input: n, int

    output: boolean
    """
    return np.ceil(np.log2(n)) == np.floor(np.log2(n))
#====================================================================================

def ave_one_level(data, device='cpu', verbose=False):
    '''
    takes averages to shrink data 1 level

    inputs:
        data: tensor of size (n_points, n_timesteps, dim, dim) that will shrink

    output:
        processed data: tensor of size (n_points, n_timesteps, dim/2, dim/2)
    '''

    if not torch.is_tensor(data): #needs to be a tensor
        data = torch.tensor(data)

    assert len(data.shape) == 4
    n_points, n_timesteps, dim, _ = data.shape

    #dim needs to be even
    assert dim % 2 == 0

    data_right_size = torch.flatten(data, 0, 1).unsqueeze(1).float().to(device)

    op = torch.nn.Conv2d(1, 1, 2, stride=2, padding=0).to(device)

    op.weight.data = torch.zeros(op.weight.data.size()).to(device)
    op.bias.data = torch.zeros(op.bias.data.size()).to(device)
    op.weight.data[0, 0, :, :] = torch.ones(op.weight.data[0, 0, :, :].size()).to(device) / 4

    # make them non-trainable
    for param in op.parameters():
        param.requires_grad = False

    if verbose:
        print("Transforming")

    shrunk = op(data_right_size)

    if verbose:
        print("reshape to print")

    return shrunk.squeeze(1).reshape((n_points, n_timesteps, dim//2, dim//2))


#====================================================================================

def make_dict_all_sizes(data, device='cpu', verbose=False):
    """
    Makes a dictionary of data at every refinedment size from current->1

    inputs:
        data: tensor(or array) of size (n_points, n_timesteps, dim, dim)

    outputs:
        dic: dictionary of tensors. Keys are dim size,
             tensors are size (n_points, n_timesteps, dim, dim)

    """

    n_points, n_timesteps, dim, _ = data.shape

    if not torch.is_tensor(data): #needs to be a tensor
        data = torch.tensor(data)

    assert isPowerOfTwo(dim)

    dic = {str(dim): data}

    for i in range(int(np.log2(dim))):
        #decrease
        if verbose:
            print("i = ", i)
        data = ave_one_level(data, device, verbose)
        dic[str(data.shape[-1])] = data.to(device)

    if verbose:
        print(dic.keys())

    return dic

#====================================================================================
def train_one_timestep(step_size, train_data, val_data=None, test_data=None, current_size=1,
                       dt=1, n_forward=5, noise=0, make_new=False, dont_train=True,
                       lr=1e-3, max_epochs=500000, batch_size=50, threshold=1e-8,
                       model_dir='./models/toy2a', i=None, j=None, print_every=1000,
                       n_inputs=3, criterion=torch.nn.MSELoss(), save_every=100):

    """
    fits or loads model at 1 timestep

    inputs:
        step_size: int
        train_data: tensor size (n_points, n_timesteps, dim**2)
        val_data:tensor size (n_val_points, n_timesteps, dim**2)
        test_data:tensor size (n_test_points, n_timesteps, dim**2)
        current_size: int, only used in file naming
        dt = 1: float
        n_forward = 5: int, number of steps to consider during training
        noise=0: float, level of noise, (right now just used in file naming)
        make_new = False: boolean, whether or not to make a new model if old already exists
        dont_train = True: boolean, whether or not to train more if model loaded
        lr = 1e-3: float, learning rate
        max_epochs = 10000: int
        batch_size = 50: int
        threshold=1e-4: float, stop training when validation gets below threshold


    outputs:
        model_time: ResNet object of trained model. Also saved
    """
    print("inside train_one_timestep")
    if (i is not None) and (j is not None):
        model_name = 'model_L{}_D{}_noise{}_i{}_j{}.pt'.format(current_size, step_size, noise, i, j)
    else:
        model_name = 'model_L{}_D{}_noise{}.pt'.format(current_size, step_size, noise)
    model_path_this = os.path.join(model_dir, model_name)

    try: #if we already have a model saved
        if make_new:
            print("Making a new model. Old one deleted. model {}".format(model_name))
            assert False
        model_time = torch.load(model_path_this)
        print("model loaded: ", model_name)
        if dont_train: #just load model, no training
            print("Model not trained more")
            return model_time
    except:
        print('create model {} ...'.format(model_path_this))
        model_time = tnet.ResNet(train_data, val_data, step_size,
                                 model_name=model_path_this, n_inputs=n_inputs,
                                 n_hidden_nodes=20, n_hidden_layers=5,
                                 activation=nn.ReLU(), n_epochs=max_epochs,
                                 threshold=threshold, print_every=print_every,
                                 save_every=save_every)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device = ", device)
        model_time.to(device)

    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model_time.parameters())

    model_time.train_model(optimizer, criterion)

    return model_time


#====================================================================================

def find_best_timestep(train_data, val_data, test_data, current_size, start_k=0, largest_k=7,
                       dt=1, n_forward=5, noise=0, make_new=False, dont_train=True,
                       lr=1e-3, max_epochs=10000, batch_size=50, threshold=1e-4,
                       criterion=torch.nn.MSELoss(), model_dir="./models/toy2",
                       i=None, j=None, print_every=1000, n_inputs=3):
    """
    Trains models with different timestep sizes and finds lowest error

    inputs:
     n_forward = 5, noise=0, make_new = False, dont_train = False):

        train_data: tensor size (n_points, n_timesteps, dim, dim), or  size (n_points, n_timesteps)
        val_data:tensor size (n_val_points, n_timesteps, dim, dim) , or  size (n_val_points, n_timesteps)
        test_data:tensor size (n_test_points, n_timesteps, dim, dim) , or  size (n_test_points, n_timesteps)
        current_size: int, only used in file naming
        start_k = 0: int, smallest timestep will be 2**start_k
        largest_k = 7:int, largest timestep will be 2**largest_k
        dt = 1: float
        n_forward = 5: int, number of steps to consider during training
        noise=0: float, level of noise, (right now just used in file naming)
        make_new = False: boolean, whether or not to make a new model if old already exists
        dont_train = False: boolean, whether or not to train more if model loaded
        lr = 1e-3: float, learning rate
        max_epochs = 10000: int
        batch_size = 50: int
        threshold=1e-4: float
        criterion = torch.nn.MSELoss(reduction='none'))


    outputs:
        models: list of ResNet models
        step_sizes: list of ints for the steps_sizes of models
        mse_list: list of floats, mse of models
        idx_lowest: int, index value with lowest mse

    """


    #transform data shapes if needed
    if len(train_data.shape) == 2:
        train_data = train_data.unsqueeze(2).unsqueeze(3)
        val_data = val_data.unsqueeze(2).unsqueeze(3)
        test_data = test_data.unsqueeze(2).unsqueeze(3)
    assert len(train_data.shape) == 4
    assert len(val_data.shape) == 4
    assert len(test_data.shape) == 4

    models = list()
    step_sizes = list()
    n_forward_list = list()
    mse_lowest = 1e10 #big number
    mse_list = list()
    mse_less = 0
    idx_lowest = -1

    for idx, k in enumerate(range(start_k, largest_k)):
        step_size = 2**k
        step_sizes.append(step_size)
        model_time = train_one_timestep(step_size, train_data, val_data,
                                        test_data=test_data, current_size=current_size,
                                        dt=dt, n_forward=n_forward, noise=noise,
                                        make_new=make_new, dont_train=dont_train,
                                        lr=lr, max_epochs=max_epochs,
                                        batch_size=batch_size, threshold=threshold,
                                        model_dir=model_dir, i=i, j=j,
                                        print_every=print_every, n_inputs=n_inputs,
                                        criterion=criterion)
        models.append(model_time)


        _, mse = model_time.predict_mse()
        mse_list.append(mse)

        if (mse < mse_lowest) or (math.isnan(mse_lowest)) or (math.isnan(mse)):
            mse_lowest = mse
            idx_lowest = idx

    return models, step_sizes, mse_list, idx_lowest, n_forward_list
#====================================================================================
#====================================================================================
def plot_lowest_error(model, i=0, title=None):
    """
    Plot data at model, idx

    inputs:
        data: tensor of shape (n_points, n_timesteps, dim, dim)
        model: Resnet model to predict on
        i: int, which validation point to graph
    outputs:
        No returned values, but graph shown


    """
    print("plot lowest error")
    plt.figure()
    y_preds, mse = model.predict_mse()
    try:
        plt.plot(y_preds[i, :, 0].detach().numpy(), label="Predicted")
    except:
        plt.plot(y_preds[i].detach().numpy(), label="Predicted")
    plt.plot(model.val_data[i, ::model.step_size, 0, 0].cpu(), label="Truth")
#     plt.ylim([-.1, 1.1])
    plt.legend()
    if title is not None:
        plt.title(title+": mse = "+str(mse))

    plt.show()
#====================================================================================
def find_error_4(data, model, truth_data, tol=2e-2, plot=False, verbose=False):
    """
    Find error over the 4 squares

    inputs:
        data: tensor of size (n_points, n_timesteps, dim, dim) to be predicted or size (n_points, n_timesteps)
        model: Resnet object to predict data on
        truth_data: tensor of size (n_points, n_timesteps, dim_larger, dim_larger) compared on
        tol = 2e-2: tolerance level to mark points as resolved or not
        criterion = torch.nn.MSELoss(reduction='none')

    outputs:
        resolved: boolean whether complete area is resolved or not
        loss: array of floats for size (dim, dim) with mse of each square
        unresolved: array of booleans, whether that part is resolved or not. (1 unresolved, 0 resolved)
    """
    if(len(data.shape)) == 2:
        data = data.unsqueeze(2).unsqueeze(3)
    assert len(data.shape) == 4

    if verbose:
        print("truth_data shape =", truth_data.shape)
        print("data shape = ", data.shape)
    _, _, dim, _ = data.shape
    data = torch.flatten(data, 2, 3)
    y_preds, _ = model.predict_mse()

    _, _, truth_dim, _ = truth_data.shape
    assert truth_dim >= dim

    truth_with_step_size = truth_data[:, ::model.step_size]

    if verbose:
        print("truth_data shape = ", truth_data.shape)
        print("y_preds shape =", y_preds.shape)
        print("truth_with_step_size shape =", truth_with_step_size.shape)
        print("truth_with_step_size[:, :-3] shape =", truth_with_step_size[:, :-3].shape)
    loss = mse(y_preds, truth_with_step_size)
    if plot:
        try:
            y_preds = y_preds.cpu()
        except:
            pass
        try:
            truth_with_step_size = truth_with_step_size.cpu()
        except:
            pass
        print("y_pred shape = ", y_preds.shape)
        print("truth_with_step_size[:,3:] shape = ", truth_with_step_size[:, 3:].shape)
        plt.title("(0,0) ")
        plt.plot(y_preds[0, :].detach().numpy(), label="y_preds")
        plt.plot(truth_with_step_size[0, :, 0, 0], label="truth")
        # plt.xlim([-2,30])
#         plt.ylim([-.1, 1.1])
        plt.legend()
        plt.show()

        plt.title("(1,0) ")
        plt.plot(y_preds[0, :].detach().numpy(), label="y_preds")
        plt.plot(truth_with_step_size[0, :, 1, 0], label="truth")
        # plt.xlim([-2,30])
#         plt.ylim([-.1, 1.1])
        plt.legend()
        plt.show()

        plt.title("(0,1) ")
        plt.plot(y_preds[0, :].detach().numpy(), label="y_preds")
        plt.plot(truth_with_step_size[0, :, 0, 1], label="truth")
        # plt.xlim([-2,30])
#         plt.ylim([-.1, 1.1])
        plt.legend()
        plt.show()

        plt.title("(1,1)")
        plt.plot(y_preds[0, :].detach().numpy(), label="y_preds")
        plt.plot(truth_with_step_size[0, :, 1, 1], label="truth")
        plt.legend()
#         plt.ylim([-.1, 1.1])
        plt.show()

    resolved = loss.max() <= tol
    unresolved_array = torch.tensor(loss <= tol)

    return resolved, loss, 1-unresolved_array.float()

#====================================================================================

def mse(data1, data2, verbose=False):
    """
    Finds Mean Squared Error between data1 and data2

    inputs:
        data1: tensor of shape (n_points, n_timestep, dim1, dim1)
        data2: tensor of shape (n_points, n_timestep, dim2, dim2)

    output:
        mse: array of size (min_dim, min_dim) with mse

    """
    if verbose:
        print("data1 shape =", data1.shape)
        print("data2 shape =", data2.shape)
    #made 4 dims
    if len(data1.shape) == 2:
        data1 = torch.tensor(data1).unsqueeze(2).unsqueeze(3)
    if len(data2.shape) == 2:
        data2 = torch.tensor(data2).unsqueeze(2).unsqueeze(3)
    if len(data1.shape) == 3:
        assert data1.shape[2] == 1
        data1 = torch.tensor(data1).unsqueeze(3)
    if len(data2.shape) == 3:
        assert data2.shape[2] == 1
        data2 = torch.tensor(data2).unsqueeze(3)

    #need to be 4d now
    assert len(data1.shape) == 4
    assert len(data2.shape) == 4

    #find bigger dim
    size1 = data1.shape[-1]
    size2 = data2.shape[-1]
    if verbose:
        print("sizes = ", size1, size2)

    size_max = max(size1, size2)

    #grow to save sizes and find mse
    mse = np.mean((grow(data1, size_max) - grow(data2, size_max))**2, axis=(0, 1))
    return mse
#====================================================================================

def grow(data, dim_full=128):
    '''
    Grow tensor from any size to a bigger size
    inputs:
        data: tensor to grow, size (n_points, n_timesteps, dim_small, dim_small)
        dim_full = 128: int of size to grow data to

    outputs:
        data_full: tensor size (n_points, n_timesteps, size_full, size_full)
    '''
    try:
        data = data.cpu()
    except:
        pass

    n_points, n_timesteps, dim_small, _ = data.shape
    assert dim_full % dim_small == 0 #need small to be multiple of full

    divide = dim_full // dim_small

    data_full = np.zeros((n_points, n_timesteps, dim_full, dim_full))
    for i in range(dim_small):
        for j in range(dim_small):
            repeated = np.repeat(np.repeat(data[:, :, i, j].reshape(n_points, n_timesteps, 1, 1),
                                           divide, axis=2), divide, axis=3)
            data_full[:, :, i*divide:(i+1)*divide, j*divide:(j+1)*divide] = repeated
    return data_full
#====================================================================================
def find_error_1(data, model, tol=2e-2, plot=False, i=0, j=0, title="find_error_1"):
    """
    Find error over the 1 square

    inputs:
        data: tensor of size (n_points, n_timesteps, dim, dim)
                to be predicted or size  (n_points, n_timesteps)
        model: Resnet object to predict data on
        tol=1e-5: tolerance level to mark points as resolved or not
        plt=False: boolean whether or not to plot
        i=0, j=0: which index to plot

    outputs:
        loss: float of mse
        resolved: boolean whether resolved or not
    """
    #reshape if needed
    if len(data.shape) == 2:
        data = data[:, ::model.step_size].unsqueeze(2).unsqueeze(3)

    assert len(data.shape) == 4

    y_preds, mse = model.predict_mse(data)

    if plot:
        truth = model.val_data[:, ::model.step_size, i, j]#[:,3:]
        plt.plot(truth[0], label="Truth")

        print('mse = ', mse)
        plt.title(title + ", mse: "+str(mse))
        plt.plot(y_preds[0], label="Predicted")
        plt.show()


    return mse, mse <= tol
#====================================================================================
def load_and_make_dict(data_dir, have_test=False):
    """load data and make dictionary with all levels"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device = ", device)
    #load data
    train_data = torch.tensor(np.load(os.path.join(data_dir, 'train_data.npy')))
    val_data = torch.tensor(np.load(os.path.join(data_dir, 'val_data.npy')))

    train_dict = make_dict_all_sizes(train_data, device)
    val_dict = make_dict_all_sizes(val_data, device)

    #if also doing test data
    if have_test:
        test_data = torch.tensor(np.load(os.path.join(data_dir, 'test_data.npy')))
        test_dict = make_dict_all_sizes(test_data, device)
        return train_dict, val_dict, test_dict
    return train_dict, val_dict
#====================================================================================
