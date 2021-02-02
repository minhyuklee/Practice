from generic import *
from splearn import *

from colorama import init
init(autoreset=True)
from colorama import Fore
import numpy as np
import os

def dataconvert(filelist, datatype):
    data = []
    for i in range(len(filelist)):
        data_temp = csvread(filelist[i], datatype[i])
        for j in range(len(data_temp)):
            data.append(data_temp[j])
    data = chartofloatarray(data)
    return(data)

path = '/Sensor_suit/2020_02_17/data'
os.chdir(path)

# parameters
train_filename = (['AA', 'FE', 'ML', 'AF', 'AM', 'FM'])
# train_filename = (['AA'])
valid_filename = (['CB1', 'CB2', 'CB3', 'CB4', 'CB5'])
# valid_filename = (['CB1'])
test_filename1 = (['AA', 'FE', 'ML', 'HF', 'AF', 'AM', 'FM'])
# test_filename1 = (['AA'])
test_filename2 = (['CB1', 'CB2', 'CB3', 'CB4', 'CB5', 'RD'])
test_filename3 = (['AA', 'FE', 'ML', 'AF', 'AM', 'FM'])
# test_filename2 = (['CB1'])

#learning rate, preset_epoch, window_size
parameters = ([0.00005, 50000, 1])

input_type = 'strain'
output_type = 'angle'

# training data
print(Fore.MAGENTA+'File import\r')
input_train = list(range(2))
input_train[0] = dataconvert([train_filename], ['training_'+input_type])
input_train[1] = dataconvert([train_filename], ['training_'+output_type])
output_train = dataconvert([train_filename], ['training_'+output_type])
    
# validation data
print(Fore.MAGENTA+'File import\r')
input_valid = list(range(2))
input_valid[0] = dataconvert([valid_filename], ['_'+input_type])
input_valid[1] = dataconvert([valid_filename], ['_'+output_type])
output_valid = dataconvert([valid_filename], ['_'+output_type])

# test data
print(Fore.MAGENTA+'File import\r')
input_test = list(range(2))
input_test[0] = dataconvert([test_filename1, test_filename2, test_filename3], ['test_'+input_type, '_'+input_type, 'training_'+input_type])
input_test[1] = dataconvert([test_filename1, test_filename2, test_filename3], ['test_'+output_type, '_'+output_type, 'training_'+output_type])
output_test = dataconvert([test_filename1, test_filename2, test_filename3], ['test_'+output_type, '_'+output_type, 'training_'+output_type])

test_filelist = listappend(listappend(test_filename1, test_filename2), test_filename3)
# test_filelist = listappend(test_filename1, test_filename2)

# machine learning
L = Learning(input_train, output_train, input_valid, output_valid, input_test, output_test)
model1, val_loss_hist1 = L.decoupling(parameters)
# model2, val_loss_hist1 = L.strain_refine(parameters1)
L.test(parameters, model1, test_filelist)
# L.test_temp(parameters1, model2, test_filelist)
# weightexport(model2, [3, 7, 9])