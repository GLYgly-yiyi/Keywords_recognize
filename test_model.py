from pathlib import Path
import time

from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy import signal
from python_speech_features import logfbank,ssc,fbank
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import random



from tensorflow.keras.layers import Softmax,Conv2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten, ReLU,Reshape,Conv1D, MaxPooling1D,GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard



def GetMfcc_logbank(wavsignal, fs):
    # 获取输入特征
    # 获取输入特征
    zero_padding = np.zeros(16352 - len(wavsignal), dtype=np.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    equal_length = np.append(wavsignal, zero_padding)
    pre_emphasis = 0.96875
    emphasized_signal = np.append(equal_length[0], equal_length[1:] - pre_emphasis * equal_length[:-1])
    #filter_banks = logfbank(signal=emphasized_signal, samplerate=fs, nfilt=64,winlen=0.032)
    filter_banks,energy = fbank(signal=emphasized_signal,preemph=pre_emphasis, samplerate=fs, nfilt=64,winlen=0.032,winfunc=np.hamming)
    filter_banks = np.log(filter_banks+1)
    filter_banks = np.array(filter_banks, dtype=np.float32)


    # # # 开始对得到的特征应用SpecAugment
    # mode = random.randint(1, 100)
    # #print("mode",mode)
    # h_start = random.randint(1, filter_banks.shape[0])
    # h_width = random.randint(1, 50)
    # #print("h_start",h_start,"h_width",h_width)
    #
    # v_start = random.randint(1, filter_banks.shape[1])
    # v_width = random.randint(1, 32)
    # #print("v_start", v_start, "v_width", v_width)
    #
    # if (mode <= 50):  # 正常特征 60%
    #     pass
    # elif (mode > 50 and mode <= 70):  # 横向遮盖 15%
    #     filter_banks[h_start:h_start + h_width, :] = 0
    #     pass
    # elif (mode > 70 and mode <= 90):  # 纵向遮盖 15%
    #     filter_banks[:, v_start:v_start + v_width] = 0
    #     pass
    # else:  # 两种遮盖叠加 10%
    #     filter_banks[h_start:h_start + h_width, :v_start:v_start + v_width] = 0
    #     pass

    return  filter_banks  # Float32 precision is enough here.

def get_specgrams(paths, nsamples=16000):
    '''
    Given list of paths, return specgrams.
    '''

    # read the wav files
    wavs = [wavfile.read(x)[1] for x in paths]

    # zero pad the shorter samples and cut off the long ones.
    data = []
    for wav in wavs:
        if wav.size < 16000:
            d = np.pad(wav, (nsamples - wav.size, 0), mode='constant')
        else:
            d = wav[0:nsamples]
        data.append(d)

    # get the specgram
    #specgram = [signal.spectrogram(d, nperseg=256, noverlap=128)[2] for d in data]
    specgram = [GetMfcc_logbank(d, 16000) for d in data]
    #print(specgram,specgram[1].shape,len(specgram))
    specgram = [s.reshape(100, 64, -1) for s in specgram]
    #print(specgram.shape)
    return specgram

def get_new_model(shape):
    '''Create a keras model.'''
    inputlayer = Input(shape=shape)

    model = BatchNormalization()(inputlayer)
    model = Conv2D(8, (3, 3), activation='elu')(model)
    model = Dropout(0.25)(model)
    model = MaxPooling2D((2, 2))(model)

    model = Conv2D(16, (3, 3), activation='elu')(model)
    model = Dropout(0.25)(model)
    model = MaxPooling2D((2, 2))(model)

    model = Conv2D(32, (3, 3), activation='elu')(model)
    model = Dropout(0.25)(model)
    model = MaxPooling2D((2, 2))(model)


    model = Flatten()(model)
    # model = Dense(32, activation='elu')(model)
    # #model = BatchNormalization()(model)
    model = Dropout(0.25)(model)

    # 11 because background noise has been taken out
    model = Dense(6, activation='softmax')(model)

    model = Model(inputs=inputlayer, outputs=model)

    return model

def get_model(shape):
    '''Create a keras model.'''
    inputlayer = Input(shape=shape)

    model = BatchNormalization()(inputlayer)
    model = Conv2D(4, (3, 3), activation='elu')(model)
    #model = Dropout(0.25)(model)
    model = MaxPooling2D((2, 2))(model)

    model = Conv2D(8, (3, 3), activation='elu')(model)
    #model = Dropout(0.25)(model)
    model = MaxPooling2D((2, 2))(model)

    model = Conv2D(16, (3, 3), activation='elu')(model)
    #model = Dropout(0.25)(model)
    model = MaxPooling2D((2, 2))(model)


    model = Flatten()(model)
    # model = Dense(32, activation='elu')(model)
    # #model = BatchNormalization()(model)
    # model = Dropout(0.25)(model)

    # 11 because background noise has been taken out
    model = Dense(6, activation='softmax')(model)

    model = Model(inputs=inputlayer, outputs=model)

    return model

shape = (100, 64, 1)
model = get_model(shape)
model.load_weights('model/keywords_weights_home_1.h5')         # 'down','left','right','stop','up'
model.summary()

path = 'G:/gly/keywords/test/test/three.wav'
#path = 'G:/gly/keywords/train/my_audio/z_my_stop3.wav'
origional = wavfile.read(path)[1]
jiduan_original = origional[12000:27999]
specgram = GetMfcc_logbank(origional, 16000)
print('intermediate_output:\n', specgram[0,0])
specgram = specgram.reshape(1,100, 64, 1)
pred = model.predict(specgram)
print('pred:\n', pred)

layer_name = 'batch_normalization'  # 获取层的名称
keras_out = model.input
intermediate_layer_model = Model(inputs=[keras_out],outputs=model.get_layer(layer_name).output)  # 创建的新模型

intermediate_output = intermediate_layer_model.predict(x = np.array(specgram))
print(intermediate_output.shape)
#layer_h10 = Reshape((200, 3200))(intermediate_output) #Reshape层
print('intermediate_output:\n', intermediate_output[0,0,0,0])
print('intermediate_output:\n', intermediate_output)
intermediate_layer_model.summary()


#
# #获取第一层卷积层权重
# conv2d = intermediate_layer_model.get_layer(index=5)
# print(conv2d.name)
# conv2d_kernel = []
# for weight in conv2d.weights:
#     print(weight.name, weight.shape, weight.dtype)
#
#     print(weight)
#     #weight_np_int = K.cast(weight, dtype='uint8')  #tensor数据转换类型
#
#     # weight_np = K.eval(weight)
#     # weight_np_int = weight_np.astype(np.int32)
#     # print(weight_np_int,weight_np_int.dtype)
#     # type(weight_np_int)