import numpy as np

def cal_fan(tensor, mode='fan_in'):
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError('Dimension of tensor must be greater than 2.')
    if len(shape) == 2:
        fan_in, fan_out = shape[:2]
    else:
        fan_out, fan_in = shape[:2]
    receptive_field_size = 1
    if len(shape) > 2:
        receptive_field_size = np.prod(shape[2:])
    fan_in *= receptive_field_size
    fan_out *= receptive_field_size
    return fan_in if mode=='fan_in' else fan_out

def kaiming_uniform(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = cal_fan(tensor, mode)

    param = 1
    if nonlinearity == 'relu':
        param = np.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if a == None:
            a = 0.01
        param = np.sqrt(2.0 / (1 + a ** 2))
    elif nonlinearity == 'tanh':
        param = 5.0 / 3
    std = param / np.sqrt(fan)
    bound = np.sqrt(3.0) * std

    return np.random.uniform(-bound, bound, size=tensor.shape)