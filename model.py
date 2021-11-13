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
import librosa
import matplotlib.pyplot as plt


from tensorflow.keras.layers import Softmax,ELU,Conv2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten, ReLU,Reshape,Conv1D, MaxPooling1D,GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard



lb = LabelBinarizer()
aaa = lb.fit_transform([ 'up','down','left','right','stop','_background_noise_'])
print(aaa)


def get_data(path):
    ''' Returns dataframe with columns: 'path', 'word'.'''
    datadir = Path(path)
    files = [(str(f), f.parts[-2]) for f in datadir.glob('**/*.wav') if f]
    df = pd.DataFrame(files, columns=['path', 'word'])

    return df


def prepare_data(df):
    '''Transform data into something more useful.'''
    #train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    #train_words = [ 'up','down','left','right','stop']
    train_words = [ 'on','off','one','two','three']
    #train_words = ['down','left','right','stop','up']
    #train_words = [ 'dog', 'one', 'down', 'right', 'cat', 'bed', 'up', 'eight', 'marvin', 'six', 'nine', 'four',
      # 'five', 'yes', 'three', 'wow', 'sheila', 'zero', 'seven', 'happy', 'go', 'bird', 'two', 'stop', 'off', 'tree',  'house', 'on', 'left', 'no']
    words = df.word.unique().tolist()
    #silence = ['_background_noise_']
    #unknown = [w for w in words if w not in silence + train_words]
    unknown = [w for w in words if w not in train_words]

    # there are only 6 silence files. Mark them as unknown too.
    #df.loc[df.word.isin(silence), 'word'] = 'unknown'
    df.loc[df.word.isin(unknown), 'word'] = 'unknown'

    return df


def time_stretch(x, rate):
    # rate：拉伸的尺寸，
    # rate > 1 加快速度
    # rate < 1 放慢速度
    return librosa.effects.time_stretch(x, rate)

# Augmentation = time_stretch(wav_data, rate=2)

def Add_noise(data,p):

    wn = np.random.normal(0,1,len(data))

    data_noise = p*wn+data

    return data_noise

#emphasized_signal = Add_noise(data=emphasized_signal, p=10)

def time_shift(x, shift):
    # shift：移动的长度
    return np.roll(x, int(shift))

#Augmentation = time_shift(wav_data, shift=fs//2)

def pitch_shifting(x, sr, n_steps, bins_per_octave=12):
    # sr: 音频采样率
    # n_steps: 要移动多少步
    # bins_per_octave: 每个八度音阶(半音)多少步
    return librosa.effects.pitch_shift(x, sr, n_steps, bins_per_octave=bins_per_octave)

# # 向上移三音（如果bins_per_octave为12，则六步）
# Augmentation = pitch_shifting(wav_data, sr=fs, n_steps=6, bins_per_octave=12)
# # 向上移三音（如果bins_per_octave为24，则3步）
# Augmentation = pitch_shifting(wav_data, sr=fs, n_steps=3, bins_per_octave=24)
# # 向下移三音（如果bins_per_octave为12，则六步）
# Augmentation = pitch_shifting(wav_data, sr=fs, n_steps=-6, bins_per_octave=12)

def GetMfcc_logbank(wavsignal, fs):
    # 获取输入特征
    # 获取输入特征
    zero_padding = np.zeros(16352 - len(wavsignal), dtype=np.float32)
    #print(wavsignal)
    #wavsignal_ADD = time_stretch(wavsignal, rate=0.9)
    #wavsignal_ADD = pitch_shifting(wavsignal, sr=16000, n_steps=6, bins_per_octave=12)
    #wavsignal_ADD = Add_noise(data=wavsignal, p=10)
    #print(wavsignal_ADD)
    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    equal_length = np.append(wavsignal, zero_padding)
    # plt.plot(equal_length)
    # plt.show()
    #equal_length = pitch_shifting(equal_length, sr=fs, n_steps=1, bins_per_octave=12)
    # plt.plot(equal_length)
    # plt.show()
    #equal_length = Add_noise(data=equal_length, p=10)
    pre_emphasis = 0.96875
    emphasized_signal = np.append(equal_length[0], equal_length[1:] - pre_emphasis * equal_length[:-1])
    #Augmentation = pitch_shifting(wav_data, sr=fs, n_steps=6, bins_per_octave=12)
    #filter_banks = logfbank(signal=emphasized_signal, samplerate=fs, nfilt=64,winlen=0.032)
    filter_banks,energy  = fbank(signal=emphasized_signal,preemph=pre_emphasis, samplerate=fs, nfilt=64,winlen=0.032,winfunc=np.hamming)
    filter_banks = np.log(filter_banks+1)
    filter_banks = np.array(filter_banks, dtype=np.float32)


    #开始对得到的特征应用SpecAugment
    # mode = random.randint(1, 100)
    # #print("mode",mode)
    # h_start = random.randint(1, filter_banks.shape[0])
    # h_width = random.randint(1, 16)
    # #print("h_start",h_start,"h_width",h_width)
    #
    # v_start = random.randint(1, filter_banks.shape[1])
    # v_width = random.randint(1, 16)
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

    return  filter_banks     # Float32 precision is enough here.


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

    specgram  = [GetMfcc_logbank(d, 16000) for d in data]
    #specgram = [specgram,all_length]
    # specgram  = [len(data)]
    # all_length = [len(data)]
    #i= 0
    # for d in data:
    #
    #     specgram[i] ,all_length[i] = GetMfcc_logbank(d, 16000)
    #     i = i+1

    #print(specgram,specgram[1].shape,len(specgram))

    specgram = [s.reshape(100, 64, -1) for s  in specgram]

    #specgram = [s.reshape(100, 64, -1) for s  in specgram]

    # for s in specgram :
    #     for lnth in all_length :
    #         specgram  = s.reshape(lnth, 64, -1)

    #print(specgram.shape)
    return specgram


def batch_generator(X, y, batch_size=16):
    '''
    Return a random image from X, y
    '''

    while True:
        # choose batch_size random images / labels from the data
        idx = np.random.randint(0, X.shape[0], batch_size)
        im = X[idx]
        label = y[idx]

        specgram = get_specgrams(im)


        yield np.concatenate([specgram]), label


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
    model = MaxPooling2D((2, 2))(model)

    model = Conv2D(8, (3, 3), activation='elu')(model)
    model = MaxPooling2D((2, 2))(model)

    model = Conv2D(16, (3, 3), activation='elu')(model)
    model = MaxPooling2D((2, 2))(model)


    model = Flatten()(model)
    # model = Dense(32, activation='elu')(model)
    # #model = BatchNormalization()(model)
    model = Dropout(0.25)(model)

    # 11 because background noise has been taken out
    model = Dense(6, activation='softmax')(model)

    model = Model(inputs=inputlayer, outputs=model)

    return model


train = prepare_data(get_data('G:/gly/keywords/train/audio_2/'))
shape = (100, 64, 1)
model = get_model(shape)
opt = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


# create training and test data.

labelbinarizer = LabelBinarizer()
X = train.path
y = labelbinarizer.fit_transform(train.word)
X, Xt, y, yt = train_test_split(X, y, test_size=0.3, stratify=y)

batch_size = 16

#tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=32)


train_gen = batch_generator(X.values, y, batch_size=batch_size)
valid_gen = batch_generator(Xt.values, yt, batch_size=batch_size)

#load_weights('model/speech_res_10_base_model_800_256.h5')
#model.load_weights('model/keywords_weights_my_scf_org.h5')
model.load_weights('model/keywords_weights_home_1.h5')

# model.fit(          x=train_gen,
#           y=valid_gen,
#           batch_size=batch_size,
#           epochs=2,)

model.fit_generator(
    generator=train_gen,
    epochs=2,
    steps_per_epoch=X.shape[0] // batch_size,
    validation_data=valid_gen,
    validation_steps=Xt.shape[0] // batch_size
   # callbacks=[tensorboard]
)

model.save('model/keywords_home_1.h5')
model.save_weights('model/keywords_weights_home_1.h5')


test = prepare_data(get_data('G:/gly/keywords/train/audio_2/on'))
#1gly/keywords/train/audio/left'))

predictions = []
paths = test.path.tolist()

for path in paths:
    specgram = get_specgrams([path])
    pred = model.predict(np.array(specgram))
    predictions.extend(pred)


labels = [labelbinarizer.inverse_transform(p.reshape(1, -1), threshold=0.5)[0] for p in predictions]
test['labels'] = labels
test.path = test.path.apply(lambda x: str(x).split('/')[-1])
submission = pd.DataFrame({'fname': test.path.tolist(), 'label': labels})
submission.to_csv('G:/gly/gly_test/shuqixuexiao/submission.csv', index=False)
