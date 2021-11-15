import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import os

import ResNet as tnet

#====================================================================================
def isPowerOfTwo(n):
    """
    checks if n is a power of two

    input: n, int

    output: boolean
    """
    return (np.ceil(np.log2(n)) == np.floor(np.log2(n)));
#====================================================================================
def shrink(data, low_dim):
    '''
    Shrinks data to certain size; either averages or takes endpoints

    inputs:
        data: array of size (n_points, n_timesteps, dim, dim) that will shrink
        low_dim: int, size to shrink to, low_dim must be less than or equal to dim

    output:
        data: array of size (n_points, n_timesteps, low_dim, low_dim)
    '''

    #check inputs
    assert len(data.shape) == 4
    n_points, n_timesteps, dim, _ = data.shape
    assert dim >= low_dim
    assert isPowerOfTwo(low_dim)

    if dim == low_dim: #same size, no change
        return data

    while(dim > low_dim):
        #shrink by 1 level until same size
        data = apply_local_op(data.float(), 'cpu', ave=average)
        current_size = data.shape[-1]

    return data
#====================================================================================
def ave_one_level(data):
    '''
    takes averages to shrink data 1 level

    inputs:
        data: tensor of size (n_points, n_timesteps, dim, dim) that will shrink

    output:
        processed data: tensor of size (n_points, n_timesteps, dim/2, dim/2)
    '''
    device = 'cpu'
    if not torch.is_tensor(data): #needs to be a tensor
        data = torch.tensor(data)

    assert len(data.shape) == 4
    n_points, n_timesteps, dim, _ = data.shape

    #dim needs to be even
    assert dim % 2 == 0

    data_right_size = torch.flatten(data, 0,1).unsqueeze(1).float()

    op = torch.nn.Conv2d(1, 1, 2, stride=2, padding=0).to(device)

    op.weight.data = torch.zeros(op.weight.data.size()).to(device)
    op.bias.data = torch.zeros(op.bias.data.size()).to(device)
    op.weight.data[0,0, :, :] = torch.ones(op.weight.data[0,0, :, :].size()).to(device) / 4

    # make them non-trainable
    for param in op.parameters():
        param.requires_grad = False

    print("Transforming")

    shrunk = op(data_right_size)

    print("reshape to print")

    return shrunk.squeeze(1).reshape((n_points, n_timesteps, dim//2, dim//2))


#====================================================================================

def make_dict_all_sizes(data):
    """
    Makes a dictionary of data at every refinedment size from current->1

    inputs:
        data: tensor(or array) of size (n_points, n_timesteps, dim, dim)

    outputs:
        dic: dictionary of tensors. Keys are dim size, tensors are size (n_points, n_timesteps, dim, dim)

    """

    n_points, n_timesteps, dim, _ = data.shape

    if not torch.is_tensor(data): #needs to be a tensor
        data = torch.tensor(data)

    assert isPowerOfTwo(dim)

    dic = {str(dim): data}

    for i in range(int(np.log2(dim))):
        #decrease
        print("i = ", i)
        data = ave_one_level(data)
        dic[str(data.shape[-1])] = data

    print(dic.keys())

    return dic

#===================================================================================
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
#====================================================================================
def train_one_timestep(step_size, train_data, val_data=None, test_data=None, current_size=1,
                       dt = 1, n_forward = 5, noise=0, make_new = False, dont_train = True,
                       lr = 1e-3, max_epochs = 10000, batch_size = 50,threshold = 1e-4,
                       model_dir = './models/toy2a',i=None, j = None,print_every=1000):

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
        model_name = 'model_L{}_D{}_noise{}_i{}_j{}.pt'.format(current_size,step_size, noise, i, j)
    else:
        model_name = 'model_L{}_D{}_noise{}.pt'.format(current_size,step_size, noise)
    model_path_this = os.path.join(model_dir, model_name)

    try: #if we already have a model saved
        if make_new:
            print("Making a new model. Old one deleted. model {}".format(model_name))
            assert False
        model_time = torch.load(model_path_this)
        print("model loaded: ", model_name)
        print("don't train = ", dont_train)
        if dont_train: #just load model, no training
            return model_time
    except:
        print('create model {} ...'.format(model_name))
        model_time = tnet.ResNet(train_data,val_data,step_size, model_name=model_name)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model_time.parameters())

    model_time.train_model(optimizer, criterion)

    return model_time

#===============================================================================

#====================================================================================

def find_best_timestep(train_data, val_data, test_data, current_size, start_k = 0, largest_k = 7,
                       dt = 1, n_forward = 5, noise=0, make_new = False, dont_train = True,
                       lr = 1e-3, max_epochs = 10000, batch_size = 50,threshold = 1e-4,
                       criterion = torch.nn.MSELoss(reduction='none'), model_dir = "./models/toy2",
                       i=None, j = None,print_every= 1000):
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
    if(len(train_data.shape)== 2):
        train_data = train_data.unsqueeze(2).unsqueeze(3)
        val_data = val_data.unsqueeze(2).unsqueeze(3)
        test_data = test_data.unsqueeze(2).unsqueeze(3)
    assert(len(train_data.shape)== 4)
    assert(len(val_data.shape)== 4)
    assert(len(test_data.shape)== 4)

    models = list()
    step_sizes = list()
    n_forward_list = list()
    mse_lowest = 1e10 #big number
    mse_list = list()
    mse_less = 0
    idx_lowest = -1

    n_points, n_timesteps, _,_ = train_data.shape

    for idx, k in enumerate(range(start_k, largest_k)):
        step_size = 2**k
        step_sizes.append(step_size)
        model_time = train_one_timestep(step_size, train_data, val_data,model_dir=model_dir, make_new=make_new)
        models.append(model_time)


        pred, mse = model_time.predict_mse()
        mse_list.append(mse)

        if (mse < mse_lowest) or (math.isnan(mse_lowest)) or (math.isnan(mse)):
            mse_lowest = mse
            idx_lowest = idx

    return models, step_sizes, mse_list, idx_lowest, n_forward_list
#====================================================================================
#====================================================================================
def plot_lowest_error(data, model, i = 0, title=None):
    """
    Plot data at model, idx

    inputs:
        data: tensor of shape (n_points, n_timesteps, dim, dim)
        model: Resnet model to predict on
        i: int, which validation point to graph
    outputs:
        No returned values, but graph shown


    """
    data  = torch.flatten(data, 2,3)
    _, total_steps, _ = data.shape
    y_preds, mse = model.predict_mse()
    plt.plot(y_preds[i], label = "Predicted")
    plt.plot(model.val_data[i,::model.step_size,0,0], label = "Truth")
    plt.ylim([-.1, 1.1])
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()
#====================================================================================
#====================================================================================
def find_error_4(data, model, truth_data, tol = 1e-5):
    """
    Find error over the 4 squares

    inputs:
        data: tensor of size (n_points, n_timesteps, dim, dim) to be predicted or size (n_points, n_timesteps)
        model: Resnet object to predict data on
        truth_data: tensor of size (n_points, n_timesteps, dim_larger, dim_larger) compared on
        tol = 1e-5: tolerance level to mark points as resolved or not
        criterion = torch.nn.MSELoss(reduction='none')

    outputs:
        resolved: boolean whether complete area is resolved or not
        loss: array of floats for size (dim, dim) with mse of each square
        unresolved: array of booleans, whether that part is resolved or not. (1 unresolved, 0 resolved)
    """
    if(len(data.shape))==2:
        data = data.unsqueeze(2).unsqueeze(3)
    assert len(data.shape) == 4
    n_points, n_timesteps, dim, _ = data.shape
    data  = torch.flatten(data, 2,3)
    y_preds, mse_avg = model.predict_mse()

    _,_, truth_dim, _ = truth_data.shape
    assert truth_dim >= dim

    truth_with_step_size = truth_data[:,::model.step_size]

    loss = mse(y_preds, truth_with_step_size[:,3:])

    resolved =  loss.max() <= tol
    unresolved_array = torch.tensor(loss <= tol)

    return resolved, loss, 1-unresolved_array.float()

#====================================================================================

def mse(data1, data2):
    """
    Finds Mean Squared Error between data1 and data2

    inputs:
        data1: tensor of shape (n_points, n_timestep, dim1, dim1)
        data2: tensor of shape (n_points, n_timestep, dim2, dim2)

    output:
        mse: array of size (min_dim, min_dim) with mse

    """
    #made 4 dims
    if len(data1.shape) != 4:
        data1 = torch.tensor(data1).unsqueeze(2).unsqueeze(3)
    if len(data2.shape) != 4:
        data2 = torch.tensor(data2).unsqueeze(2).unsqueeze(3)

    #need to be 4d now
    assert len(data1.shape) ==4
    assert len(data2.shape) ==4

    #find bigger dim
    size1 = data1.shape[-1]
    size2 = data2.shape[-1]
    print(size1, size2)
    size_max = max(size1, size2)

    #grow to save sizes and find mse
    mse = np.mean((grow(data1, size_max) - grow(data2, size_max))**2, axis = (0, 1))
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
    n_points, n_timesteps, dim_small, _ = data.shape
    assert dim_full % dim_small == 0 #need small to be multiple of full

    divide = dim_full // dim_small

    data_full = np.zeros((n_points, n_timesteps, dim_full,dim_full))
    for i in range(dim_small):
        for j in range(dim_small):
            repeated = np.repeat(np.repeat(data[:,:,i,j].reshape(n_points,n_timesteps,1,1), divide, axis = 2), divide, axis = 3)
            data_full[:,:,i*divide:(i+1)*divide, j*divide:(j+1)*divide] = repeated
    return data_full
#====================================================================================
