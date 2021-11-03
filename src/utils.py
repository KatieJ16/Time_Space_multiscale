import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate(snapshots, file_name = "animation.gif"):


    fps = 30
    nSeconds = len(snapshots)/fps
    # snapshots = [ np.random.rand(5,5) for _ in range( nSeconds * fps ) ]

    # a=output
    # snapshots = output
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(8,8) )

    a = snapshots[0,:,:,:][0].T
    im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0.0, vmax=1.0)
    plt.colorbar()

    print("Animating, may take a little while...")

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )

        im.set_array(snapshots[i,:,:,:,][0].T)
        return [im]

    anim = animation.FuncAnimation(
                                   fig,
                                   animate_func,
                                   frames = int(nSeconds * fps),
                                   interval = 1000 / fps, # in ms
                                   )
    writergif = animation.PillowWriter(fps=30)
    anim.save(file_name, writer=writergif)#, fps=30)


    print('Done! gif saved to ', file_name)


#helper functions

def check_pixel_level_loss(d1, d2, tol, device, w=0.5):
    """
    :param d1: data 1
    :param d2: data 2
    :param tol: a float, represent the tolerance
    :param device: device
    :param w: loss = w * mse_loss + (1 - w) * max_loss
    :return: a boolean value, if error satisfies the tolerance,
             a torch tensor of overall loss distribution,
             and a boolean torch tensor
    """
    assert isinstance(tol, float), print('tol should be a float!')

    loss1 = torch.mean((d1 - d2)**2, dim=0, keepdim=True)
    loss2 = torch.max((d1 - d2)**2, dim=0, keepdim=True)[0]
    loss = w * loss1 + (1 - w) * loss2
    loss_summary = apply_local_op(loss, device).squeeze()

    return loss_summary.max() <= tol, loss_summary, loss_summary <= tol

def obtain_data_at_current_level(data, n_levels, level,average = True):
    """ goes from n_levels to level
    """
    for i in range(n_levels - level - 1):
        data = apply_local_op(data, 'cpu', ave=average)
    return data

def decrease_to_size(data, level,average = True):
    """ goes from n_levels to level"""
    current_size = data.shape[-1]
    if(current_size<= level):
        return data
    while(current_size > level):
        data = apply_local_op(data.float(), 'cpu', ave=average)
        current_size = data.shape[-1]
    return data

def apply_local_op(data, device, mode='conv', ave=True):
    """
    :param data: data to be processed
    :param device: which device is the data placed in?
    :param mode: string, 'conv' or 'deconv'
    :param ave: if to use local average or sample the center
    :return: processed data
    """
    in_channels, out_channels = 1,1#, _, _ = data.size()
#     print("data.size() = ", data.size())
    n = min(in_channels, out_channels)
    if mode == 'conv':
        op = torch.nn.Conv2d(out_channels, out_channels, 2, stride=2, padding=0).to(device)
    elif mode == 'deconv':
        op = torch.nn.ConvTranspose2d(out_channels, out_channels, 3, stride=2, padding=0).to(device)
    else:
        raise ValueError('mode can only be conv or deconv!')
    op.weight.data = torch.zeros(op.weight.data.size()).to(device)
    op.bias.data = torch.zeros(op.bias.data.size()).to(device)

    for i in range(n):
        if mode == 'conv':
            if ave:
                op.weight.data[i, i, :, :] = torch.ones(op.weight.data[i, i, :, :].size()).to(device) / 4
            else:
                op.weight.data[i, i, 1, 1] = torch.ones(op.weight.data[i, i, 1, 1].size()).to(device)
        elif mode == 'deconv':
            op.weight.data[i, i, :, :] = torch.ones(op.weight.data[i, i, :, :].size()).to(device) / 4
            op.weight.data[i, i, 0, 1] += 1 / 4
            op.weight.data[i, i, 1, 0] += 1 / 4
            op.weight.data[i, i, 1, 2] += 1 / 4
            op.weight.data[i, i, 2, 1] += 1 / 4
            op.weight.data[i, i, 1, 1] += 1 / 4
            op.weight.data[i, i, 1, 1] += 1 / 2

    # make them non-trainable
    for param in op.parameters():
        param.requires_grad = False

    return op(data)


def grow(data, size_full=128):
    '''
    Grow tensor from any size to the full size default 128
    
    Takes data of a size  and converts to size (n_points, size_full, size_full) (default is 128)
    
    Inputs:
        data: array or tensor of size (dim, dim), (n_points, dim, dim), or (n_points, 1, dim, dim)
        size: dim size of final answer, must be equal or larger than input dim, default 128
        
    Output:
        averaged_full: tensor size (n_points, size_full, size_full)
    '''
    data = make_size_4(data)
    n_points, _, size_small, _ = data.shape
    divide = size_full // size_small

    averaged_full = np.zeros((n_points, size_full,size_full))
    for i in range(size_small):
        for j in range(size_small):
            repeated = np.repeat(np.repeat(data[:,0,i,j].reshape(n_points,1,1), divide, axis = 1), divide, axis = 2)
            averaged_full[:,i*divide:(i+1)*divide, j*divide:(j+1)*divide] = repeated
    return averaged_full

def make_next_layer(data_big, data_small, unresolved, size = 128):
    '''determine the next level make up with new map, old map and unresolved
    All tensors
    '''
    # size_small = data_small.shape[-1]
    size_max = data_big.shape[-1]
    multi = grow(data_big, size_max) * (1-grow(unresolved, size_max)) + grow(data_small, size_max)*grow(unresolved, size_max)
    return multi

def data_of_size(data,size):
    """
    Takes data of size (n_points, dim, dim) and shrinks to size (n_points, size, size)
    takes averages to shrink
    """
    return decrease_to_size(torch.tensor(data).unsqueeze(1), size)[:,0,:,:]


def MSE(data1, data2, size_small, tol = 1e-5, size = 128, keep_small = True):
    '''
    Finds the mean square error of two tensors. will return (bol, array, bol array).
    Arrays are of size (size_small, size_small)
    '''
    data1 = make_size_4(data1)
    data2 = make_size_4(data2)
    size1 = data1.shape[-1]
    size2 = data2.shape[-1]
    size_small = min(size1, size2)
    size_max = max(size1, size2)
    print("size_small = ", size_small)
    print("size_max = ", size_max)
    # loss = torch.nn.MSELoss()
    # mse = loss(grow(data1, size_max), grow(data2, size_max))
    mse = np.mean((grow(data1, size_max) - grow(data2, size_max))**2, axis = 0)


    print("torch.tensor(mse).unsqueeze(0).unsqueeze(0).shape = ", torch.tensor(mse).unsqueeze(0).unsqueeze(0).shape)
    if keep_small:
        mse_smaller = decrease_to_size(torch.tensor(mse).unsqueeze(0).unsqueeze(0), size_small)

        loss_summary = mse_smaller[0,0]
    else:
        mse = torch.tensor(mse).unsqueeze(0).unsqueeze(0)
        loss_summary = mse[0,0]
        
    return loss_summary.max() <= tol, loss_summary, loss_summary <= tol

def make_size_4(data):
    '''Makes data array/tensor to the right size of (n,1,n_dim,d_dim)'''
    data = torch.tensor(data)
    if len(data.shape) == 2:
        data = data.unsqueeze(0)
    if len(data.shape) == 3:
        data = data.unsqueeze(1)
    assert (len(data.shape) == 4)
    return data
