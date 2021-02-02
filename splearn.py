from generic import *

from keras.initializers import glorot_normal
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Reshape, concatenate, Multiply, Add, LSTM
from keras.optimizers import Nadam, Adamax, Adam
from tkinter import *
from colorama import init
init(autoreset=True)
from colorama import Fore
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import time
import math
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

class Learning:
    # filename = ''
    running = True
    MODEL_SAVE_FOLDER_PATH = './model/'
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)

    root = Tk()
    root.title("Title")
    root.geometry("450x450")
    app = Frame(root)
    app.grid()
    label0 = Label(app, text = "Sensor position")
    num0 = Entry(app)
    num1 = Entry(app)
    num2 = Entry(app)
    num3 = Entry(app)
    num4 = Entry(app)
    label1 = Label(app, text = "Epoch")
    label2 = Label(app, fg = "red", text = "training loss")
    label3 = Label(app, fg = "green", text = "validation loss")
    label4 = Label(app, fg = "blue", text = "filename")
    label_filelist = Label(app, text = "filename list")
    text_filelist = Text(app, width = 60, height = 15)
    scr = Scrollbar(app, orient = VERTICAL, command = text_filelist.yview)
    scr.grid(row = 4, column = 2, pady = 2)

    def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test

    def stop():
        Learning.running = False

    def TKinitializer():
        button1 = Button(Learning.app, text="Stop", command = Learning.stop, width = 60, height = 3)
        button1.grid(row = 6, columnspan = 2, pady = 2)

        Learning.num0.grid(row = 0, column = 1, pady = 2)
        Learning.num1.grid(row = 1, column = 1, pady = 2)
        Learning.num2.grid(row = 2, column = 1, pady = 2)
        Learning.num3.grid(row = 3, column = 1, pady = 2)
        Learning.label0.grid(row = 0, column = 0, pady = 2)
        Learning.label1.grid(row = 1, column = 0, pady = 2)
        Learning.label2.grid(row = 2, column = 0, pady = 2)
        Learning.label3.grid(row = 3, column = 0, pady = 2)
        Learning.label_filelist.grid(row = 4, columnspan = 2, pady = 2)
        Learning.text_filelist.grid(row = 5, columnspan = 2, pady = 2)
        Learning.label4.grid(row = 7, column = 0, pady = 2)
        Learning.num4.grid(row = 7, column = 1, pady = 2)

        Learning.num0.delete(0, 'end')
        Learning.num1.delete(0, 'end')
        Learning.num2.delete(0, 'end')
        Learning.num3.delete(0, 'end')

        Learning.running = True

        Learning.root.update()

    def TKdisplay(input_position, idx, loss_hist, filename):        
        Learning.num0.delete(0, 'end')
        Learning.num0.insert(0, input_position)   
        Learning.num1.delete(0, 'end')
        Learning.num1.insert(0, idx)
        Learning.num2.delete(0, 'end')
        Learning.num2.insert(0, round(loss_hist[0][len(loss_hist[0])-1], 8))
        Learning.num3.delete(0, 'end')
        Learning.num3.insert(0, round(loss_hist[1][len(loss_hist[1])-1], 8))
        Learning.num4.delete(0, 'end')
        Learning.num4.insert(0, str(filename)+'.h5')
        Learning.root.update()

    def modelcheck(model, idx, loss, loss_hist, model_name_hist):
        # training loss : loss[0][0] / validation loss : loss[1][0]
        # training loss history : loss_hist[0] / validation loss history : loss_hist[1]
        loss_hist[1].append(loss[1][0])
        loss_hist[0].append(loss[0][0])

        if idx <= 3:
            model_path = Learning.MODEL_SAVE_FOLDER_PATH + str(loss[1][0])+'.h5'
            model.save(model_path)
            model_name_hist.append(loss[1][0])
            Learning.text_filelist.insert(END, str(loss[1][0])+'.h5\n') 
            filename = ''

        if idx > 3:
            filename = min(model_name_hist)
            if loss[1][0] < filename:
                slope_current = loss_hist[1][len(loss_hist[1])-1]-loss_hist[1][len(loss_hist[1])-2]
                slope_past = loss_hist[1][len(loss_hist[1])-2]-loss_hist[1][len(loss_hist[1])-3]

                if slope_current < 0:
                    if slope_past > 0:
                        model_path = Learning.MODEL_SAVE_FOLDER_PATH + str(loss_hist[1][len(loss_hist[1])-2])+'.h5'
                        model.save(model_path)
                        model_name_hist.append(loss_hist[1][len(loss_hist[1])-2])
                        Learning.text_filelist.insert(END, str(loss_hist[1][len(loss_hist[1])-2])+'.h5\n') 

                elif slope_current > 0:
                    if slope_past < 0:
                        model_path = Learning.MODEL_SAVE_FOLDER_PATH + str(loss_hist[1][len(loss_hist[1])-2])+'.h5'
                        model.save(model_path)
                        model_name_hist.append(loss_hist[1][len(loss_hist[1])-2])
                        Learning.text_filelist.insert(END, str(loss_hist[1][len(loss_hist[1])-2])+'.h5\n') 

                else:
                    model_path = Learning.MODEL_SAVE_FOLDER_PATH + str(loss[1][0])+'.h5'
                    model.save(model_path)
                    model_name_hist.append(loss[1][0])
                    Learning.text_filelist.insert(END, str(loss[1][0])+'.h5\n') 

        return(loss_hist, model_name_hist, filename)

    def decoupling(self, parameters):
        idx = 1
        removefiles(Learning.MODEL_SAVE_FOLDER_PATH)
        [learning_rate, preset_epoch, window_size] = parameters

        x_train = dataconvert_input(self.x_train[0], window_size, [19])
        x_valid = dataconvert_input(self.x_valid[0], window_size, [19])
        x_train1 = dataconvert_input(self.x_train[0], window_size, [33])
        x_valid1 = dataconvert_input(self.x_valid[0], window_size, [33])
        x_train2 = dataconvert_output(self.x_train[1], window_size, [1])
        x_valid2 = dataconvert_output(self.x_valid[1], window_size, [1])
        x_train3 = dataconvert_output(self.x_train[1], window_size, [3])
        x_valid3 = dataconvert_output(self.x_valid[1], window_size, [3])
        y_train = dataconvert_output(self.y_train, window_size, [2])
        y_valid = dataconvert_output(self.y_valid, window_size, [2])

        batch_size = len(x_train)
        sensor_count = len(x_train[0][0])
        init_display(x_train, y_train)

        opt = Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0)
        with tf.device('/gpu:0'):
            hidden_size = 128
            act_function = 'relu'

            input1 = Input(shape=(window_size, 1))
            input2 = Input(shape=(window_size, 1))
            input3 = Input(shape=(1, ))
            input4 = Input(shape=(1, ))
            y1 = Flatten()(input1)
            y2 = Flatten()(input2)

            y1_1 = concatenate([y1, input3])
            y1_2 = concatenate([y1, input4])
            y2_1 = concatenate([y2, input3])
            y2_2 = concatenate([y2, input4])

            y1_1 = Dense(32, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y1_1)
            y1_1 = Dense(32, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y1_1)
            y1_1 = Dense(1, activation = 'linear', kernel_initializer=glorot_normal(seed=1))(y1_1)

            y1_2 = Dense(32, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y1_2)
            y1_2 = Dense(32, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y1_2)
            y1_2 = Dense(1, activation = 'linear', kernel_initializer=glorot_normal(seed=1))(y1_2)

            y2_1 = Dense(32, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y2_1)
            y2_1 = Dense(32, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y2_1)
            y2_1 = Dense(1, activation = 'linear', kernel_initializer=glorot_normal(seed=1))(y2_1)

            y2_2 = Dense(32, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y2_2)
            y2_2 = Dense(32, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y2_2)
            y2_2 = Dense(1, activation = 'linear', kernel_initializer=glorot_normal(seed=1))(y2_2)

            # y4 = concatenate([y1_1, y1_2, y2_1, y2_2])
            y4 = concatenate([y1, y2])
            # y4 = y1
            # y1 = LSTM(32, activation = 'relu', recurrent_activation = 'sigmoid', kernel_initializer=glorot_normal(seed=1))(input1)
            # y4 = concatenate([y1, y2_3])
            y4 = Dense(hidden_size, activation = act_function, kernel_initializer=glorot_normal(seed=1))(y4)
            y4 = Dense(hidden_size, activation = act_function, kernel_initializer=glorot_normal(seed=1))(y4)
            y4 = Dense(hidden_size, activation = act_function, kernel_initializer=glorot_normal(seed=1))(y4)
            y4 = Dense(hidden_size, activation = act_function, kernel_initializer=glorot_normal(seed=1))(y4)
            y4 = Dense(hidden_size, activation = act_function, kernel_initializer=glorot_normal(seed=1))(y4)
            y4 = Dense(hidden_size, activation = act_function, kernel_initializer=glorot_normal(seed=1))(y4)
            o1 = Dense(1, activation = 'linear', kernel_initializer=glorot_normal(seed=1))(y4)

            model = Model(inputs = [input1, input2, input3, input4], outputs = [o1])
            model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
            model.summary()

            Learning.TKinitializer()
            loss_hist = [[], []]
            model_name_hist = []
            while Learning.running:
                
                hist = model.fit([x_train, x_train1, x_train2, x_train3], y_train, validation_data = ([x_valid, x_valid1, x_valid2, x_valid3], y_valid), epochs=1, batch_size=batch_size, verbose = 0)
                loss = [hist.history['loss'], hist.history['val_loss']]
                val_loss = hist.history['val_loss']

                loss_hist, model_name_hist, filename = Learning.modelcheck(model, idx, loss, loss_hist, model_name_hist)
                Learning.TKdisplay([132], idx, loss_hist, filename)

                if idx == preset_epoch:
                    break
                idx += 1
            Learning.root.update()
            Learning.root.destroy()
        return(model, loss_hist)
    
    def strain_refine(self, parameters):
        idx = 1
        removefiles(Learning.MODEL_SAVE_FOLDER_PATH)
        [learning_rate, preset_epoch, window_size, input_position, output_position] = parameters
        learning_rate = 0.1

        x_train1 = dataconvert_input(self.x_train, 100, [100])
        x_train2 = dataconvert_output2(self.y_train, 100, [2])
        x_train3 = dataconvert_output(self.x_train, 100, [100])

        x_valid1 = dataconvert_input(self.x_valid, 100, [100])
        x_valid2 = dataconvert_output2(self.y_valid, 100, [2])
        x_valid3 = dataconvert_output(self.x_valid, 100, [100])

        y_train = dataconvert_output(self.y_train, 100, [1])
        y_valid = dataconvert_output(self.y_valid, 100, [1])

        batch_size = len(x_train1)
        sensor_count = len(x_train1[0][0])
        init_display(x_train1, y_train)

        opt = Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0)
        with tf.device('/gpu:0'):
            hidden_size = 128
            act_function = 'relu'

            input1 = Input(shape=(100, 1))
            i1 = Flatten()(input1)
            input2 = Input(shape=(1, ))
            input3 = Input(shape=(1, ))

            # y1_1 = Dense(16, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(i1)
            y1_1 = Dense(1, activation = 'linear', kernel_initializer=glorot_normal(seed=1))(i1)
            # y1_2 = Dense(1, activation = customsigmoid, kernel_initializer=glorot_normal(seed=1))(input2)
            y1_2 = input2

            y2_1 = Multiply()([y1_1, y1_2])
            y2_2 = Dense(1, activation = 'linear', kernel_initializer=glorot_normal(seed=1))(input3)

            y3 = Add()([y2_1, y2_2])
            # y3 = Dense(128, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y3)
            # y3 = Dense(128, activation = 'relu', kernel_initializer=glorot_normal(seed=1))(y3)
            o1 = Dense(1, activation = 'linear', kernel_initializer=glorot_normal(seed=1))(y3)

            model = Model(inputs = [input1, input2, input3], outputs = [o1])
            model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
            model.summary()

            Learning.TKinitializer()
            loss_hist = [[], []]
            model_name_hist = []
            while Learning.running:
                
                hist = model.fit([x_train1, x_train2, x_train3], y_train, validation_data = ([x_valid1, x_valid2, x_valid3], y_valid), epochs=1, batch_size=batch_size, verbose = 0)
                loss = [hist.history['loss'], hist.history['val_loss']]
                val_loss = hist.history['val_loss']

                loss_hist, model_name_hist, filename = Learning.modelcheck(model, idx, loss, loss_hist, model_name_hist)
                Learning.TKdisplay(input_position, idx, loss_hist, filename)

                if idx == preset_epoch:
                    break
                idx += 1
            Learning.root.update()
            Learning.root.destroy()
        return(model, loss_hist)

    def test(self, parameters, model, filelist):
        print(Fore.MAGENTA+'Test')
        [learning_rate, preset_epoch, window_size] = parameters

        probar(0, len(filelist), 'verification procedure', 50)
        # model = load_model(MODEL_SAVE_FOLDER_PATH+str(filename)+'.h5')

        rmse_table = np.zeros((len(filelist), 3))
        # max_error_table = np.zeros((len(filelist), len(output_position)))

        filecount = 0
        for filecount in range(0, len(filelist)):
            x_test_temp1 = []
            x_test_temp2 = []
            y_test_temp = []
            x_test_temp1.append(self.x_test[0][filecount])
            x_test_temp2.append(self.x_test[1][filecount])
            y_test_temp.append(self.y_test[filecount])

            input1 = dataconvert_input(x_test_temp1, window_size, [19])
            input2 = dataconvert_input(x_test_temp1, window_size, [33])
            input3 = dataconvert_output(x_test_temp2, window_size, [1])
            input4 = dataconvert_output(x_test_temp2, window_size, [3])
            y_test_temp = dataconvert_output(y_test_temp, window_size, [2])

            output = model.predict([input1, input2, input3, input4])
            if filecount >= 13:
                pd.DataFrame(output).to_csv(str(filelist[filecount])+'training_est.csv', header = False, index = False)
            elif filecount <= 6:
                pd.DataFrame(output).to_csv(str(filelist[filecount])+'test_est.csv', header = False, index = False)
            else:
                pd.DataFrame(output).to_csv(str(filelist[filecount])+'_est.csv', header = False, index = False)

            output_position = [2]
            for i in range(len(output_position)):
                rmse, max_error = data_analysis(output, y_test_temp)
                rmse_table[filecount, i] = rmse
                # max_error_table[filecount, i] = max_error

            probar(filecount+1, len(filelist), 'verification procedure', 50) 

        pd.DataFrame(rmse_table).to_csv('rmse.csv', header = False, index = False)
        # pd.DataFrame(max_error_table).to_csv('max_error_table.csv', header = False, index = False)
        print(Fore.YELLOW+'done\n')

    def test_temp(self, parameters, model, filelist):
        print(Fore.MAGENTA+'Test')
        [learning_rate, preset_epoch, window_size, input_position, output_position] = parameters

        probar(0, len(filelist), 'verification procedure', 50)
        # model = load_model(MODEL_SAVE_FOLDER_PATH+str(filename)+'.h5')

        rmse_table = np.zeros((len(filelist), len(output_position)))
        max_error_table = np.zeros((len(filelist), len(output_position)))

        for filecount in range(0, len(filelist)):
            x_test_temp = []
            y_test_temp = []
            x_test_temp.append(self.x_test[filecount])
            y_test_temp.append(self.y_test[filecount])

            x_test_temp1 = dataconvert_input(x_test_temp, 100, [100])
            x_test_temp2 = dataconvert_output2(y_test_temp, 100, [2])
            x_test_temp3 = dataconvert_output(x_test_temp, 100, [100])

            y_test_temp = dataconvert_output(y_test_temp, 100, [1])

            output = model.predict([x_test_temp1, x_test_temp2, x_test_temp3])
            if filecount >= 13:
                pd.DataFrame(output).to_csv(str(filelist[filecount])+'training_rstrain.csv', header = False, index = False)
            elif filecount <= 6:
                pd.DataFrame(output).to_csv(str(filelist[filecount])+'test_rstrain.csv', header = False, index = False)
            else:
                pd.DataFrame(output).to_csv(str(filelist[filecount])+'_rstrain.csv', header = False, index = False)

            for i in range(len(output_position)):
                rmse, max_error = data_analysis(output, y_test_temp)
                rmse_table[filecount, i] = rmse
                max_error_table[filecount, i] = max_error

            probar(filecount+1, len(filelist), 'verification procedure', 50) 

        pd.DataFrame(rmse_table).to_csv('rmse.csv', header = False, index = False)
        pd.DataFrame(max_error_table).to_csv('max_error_table.csv', header = False, index = False)
        print(Fore.YELLOW+'done\n')

def weightexport(model, layer):
        for i in range(len(layer)):
            weight = model.layers[layer[i]-1].get_weights()[0]
            bias = model.layers[layer[i]-1].get_weights()[1]
            pd.DataFrame(weight).to_csv('weight'+str(layer[i])+'.csv', header = False, index = False)
            pd.DataFrame(weight).to_csv('bias'+str(layer[i])+'.csv', header = False, index = False)