from colorama import init
init(autoreset=True)
from colorama import Fore
from itertools import permutations
from collections import OrderedDict
from keras import backend as K
from tkinter import *
import xlsxwriter
import xlrd
import csv
import numpy as np
import pandas as pd
import os
import math

array_data_size = 32

def csvread(filelist, filetype):
    data = []
    filename_history = []
    for i in range(0, len(filelist)):
        filename = filelist[i]
        filename_history.append(filename)
        print(Fore.BLUE+'file type :', filetype, Fore.CYAN+'//', Fore.BLUE+'file name :', filename_history, end = '\r')
        data_parsing = read_csv(filename+filetype+'.csv')
        data.append(data_parsing)
    print()        
    print(str(i+1), 'files', Fore.YELLOW+'done')
    return(data)

def csvread_temp(filelist, filetype):
    data = []
    filename_history = []
    for i in range(0, len(filelist)):
        filename = filelist[i]
        filename_history.append(filename)
        print(Fore.BLUE+'file type :', filetype, Fore.CYAN+'//', Fore.BLUE+'file name :', filename_history, end = '\r')
        data_parsing = read_csv(filename+filetype+'_angle.csv')
        temp_data = read_csv(filename+filetype+'_esty.csv')
        data_parsing[:, 1] = temp_data[:, 0]
        data.append(data_parsing)
    print()        
    print(str(i+1), 'files', Fore.YELLOW+'done')
    return(data)

def chartofloatarray(data):
    data_out = []
    for i in range(0, len(data)):
        data_temp = data[i]
        data_out_temp = np.zeros((len(data_temp), len(data_temp[0])))
        data_out_temp[:, :] = data_temp[:, :]
        data_out.append(data_out_temp)
    return(data_out)

def dataselection(data, position):
    data_out = np.zeros((len(data), len(position)))
    for count in range(len(position)):
        data_out[:, count] = data[:, position[count]-1]
    return(data_out)

def listtoarray(list_data):
    row_size = len(list_data)
    col_size = 0
    for line in list_data:
        if len(line) > col_size:
            col_size = len(line)

    array_data = np.chararray((row_size, col_size), itemsize = array_data_size, unicode = True)

    for row_count, line in enumerate(list_data):
        for col_count, data in enumerate(line):
            if data == '':
                data = 0
            array_data[row_count, col_count] = data

    # print(row_size, col_size)
    return(array_data)

def polyval(p, data):
    data_out = 0
    for i in range(len(p)):
        data_out += p[i]*data**(len(p)-1-i)
    return(data_out)

def listtoarray2(data):
    if len(data) != 1:
        for count in range(len(data)):
            if count == 0:
                data_out = data[count]
            else:
                data_out = np.vstack((data_out, data[count]))
    else:
        data_out = data[0]
    return(data_out)
   
def removefiles(filepath):
    for file in os.scandir(filepath):
        os.remove(file.path)

def probar(iteration, total, prefix, length, verbose = 1):
    if verbose == 1:
        decimals = 1
        fill = '█'
        space_init = math.floor(math.log10(total))
        if iteration == 0:
            space = ' '*space_init
        else:
            space = ' '*(space_init-math.floor(math.log10(iteration)))
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * (iteration) // (total))
        bar = fill * filledLength + '-' * (length - filledLength)
        print(Fore.BLUE+prefix, ':', space+'%d/%d |%s| %s%%' %(iteration, total, bar, percent), end = '\r')
        if iteration == total: 
            print()

def read_csv(filename):
    with open(filename, newline='', encoding = 'utf-8') as csvfile:
        data_read = list(csv.reader(csvfile))

    data_temp = listtoarray(data_read)
    if data_temp[0, 0] == 'Format Version':
        data_out = data_temp[7:, 2:]
    else:
        data_out = data_temp

    return(data_out)

def reshape2to3(data, window_size, data_type):
    row = len(data)
    col = len(data[0])
    if data_type == 'input':
        data_out = np.zeros((row-window_size+1, window_size, col))
        for j in range(0, row-window_size+1):
            for i in range(j, window_size+j):
                data_out[j, i-j, :] = data[i, :]
    elif data_type == 'output':
        data_out = np.zeros((row-window_size+1, col))
        for j in range(0, row-window_size+1):
            data_out[j, :] = data[j+window_size-1, :]
    return(data_out)

def dataconvert(data, position, window_size = None, data_type = None):
    data = listtoarray2(data)
    data = dataselection(data, position)
    data = reshape2to3(data, window_size, data_type)
    return(data)

def dataconvert_input(strain, window_size, input_index):
    strain = listtoarray2(strain)
    strain_temp = np.zeros((len(strain), len(input_index)))
    for i in range(0, len(strain)):
        for j in range(len(input_index)):
            strain_temp[i, j] = strain[i, input_index[j]-1]
    strain_split = reshape2to3(strain_temp, window_size, 'input')
    return(strain_split)

def dataconvert_output(angle, window_size, output_index):
    angle = listtoarray2(angle)
    angle_temp = np.zeros((len(angle), len(output_index)))
    for i in range(0, len(angle)):
        for j in range(len(output_index)):
            angle_temp[i, j] = angle[i, output_index[j]-1]
    angle_split = reshape2to3(angle_temp, window_size, 'output')
    return(angle_split)

def dataconvert_output2(angle, window_size, output_index):
    angle = listtoarray2(angle)
    angle_temp = np.zeros((len(angle), 1))
    for i in range(0, len(angle)):
        for j in range(len(output_index)):
            angle_temp[i, j] = customsigmoid(angle[i, output_index[j]-1])
    angle_split = reshape2to3(angle_temp, window_size, 'output')
    return(angle_split)

def init_display(x_train, y_train):
    print(Fore.MAGENTA + 'Supervised learning')
    print(Fore.BLUE+'sensor count', ':', len(x_train[0][0]), Fore.CYAN+'//', Fore.BLUE+'angle count', ':', len(y_train[0]))
    print(Fore.BLUE+'window size :', len(x_train[0]))
    print(Fore.BLUE+'batch size', ':', len(x_train))

def customsigmoid(x):
    c1 = 0.1
    c2 = 70
    # c1 = 1
    # c2 = 40
    return (1/(1+math.exp(-c1*(x-c2))))

# def customsigmoid(x):
#     # c1 = 0.1
#     # c2 = 70
#     c1 = 1
#     c2 = 40
#     return (K.sigmoid(-c1*(x-c2)))

def data_analysis(est, ref):
    # rmse
    error_sum = 0
    for i in range(len(est)):
        error_sum += math.pow((est[i, 0]-ref[i, 0]), 2)/len(est)
    rmse = math.sqrt(error_sum)

    # max error
    max_list = []
    for i in range(len(est)):
        max_list.append(abs(est[i, 0]-ref[i, 0]))
    max_error = max(max_list)

    return(rmse, max_error)

def csv_append(filename, data, sensor_position, row_offset, column_offset):
    count = 0
    row_count = 0
    row_data = []
    [data_size1, data_size2] = np.shape(data)
    with open(filename, 'r') as read_obj:
        csv_read = csv.reader(read_obj)
        for row in csv_read:
            if count >= row_offset:
                start_index= data_size2*sensor_position[0]-1-column_offset
                end_index = data_size2*sensor_position[0]-1-column_offset+data_size2
                row[start_index:end_index] = data[row_count, 0:data_size2]
                row_count += 1
            row_data.append(row)
            count += 1

    with open(filename, 'w', newline = '') as write_obj:
        csv_writer = csv.writer(write_obj)
        for i in range(len(row_data)):
            csv_writer.writerow(row_data[i])

def listappend(list1, list2):
    list_out = []
    for i in range(len(list1)):
        list_out.append(list1[i])
    for i in range(len(list2)):
        list_out.append(list2[i])
    return(list_out)