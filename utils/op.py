import tensorflow as tf
import numpy as np


def generate_fc_weight(shape, name):
    threshold = 1.0 / np.sqrt(shape[0])
    weight_matrix = tf.random_uniform(shape, minval=-threshold, maxval=threshold)
    weight = tf.Variable(weight_matrix, name=name)
    return weight

def generate_fc_bias(shape, name):
    # bias_distribution = np.zeros(shape)
    bias_distribution = tf.constant(0.0, shape=shape)
    bias = tf.Variable(bias_distribution, name=name)
    return bias

def clip(x,min,max):
    if x<min:
        return min
    elif x>max:
        return max
    else:
        return x

def fill_zero(list):
    max_one = max(list)
    max_index = list.index(max_one)
    error = 1 - sum(list)
    list[max_index] = max_one + error
    return list

def average(target):
    sum = 0
    for item in target:
        sum = sum + item
    mean = sum * 1.0 / len(target)
    return mean