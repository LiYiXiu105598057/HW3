
# coding: utf-8

# In[1]:


import os
import re
from glob import glob
import pandas as pd
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPool2D, concatenate, Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils


# In[2]:


from scipy.signal import stft
def process_wav_file(fname):
    wav = read_wav_file(fname)
    
    L = 16000
    
    if len(wav) > L:
        i = np.random.randint(0, len(wav) - L) # 這是錯誤
        wav = wav[i: (i+L)]
    elif len(wav) < L:
        rem_len = L - len(wav)
        i = np.random.randint(0, len(silence_data) - rem_len)
        silence_part = silence_data[i: (i+L)]
        j = np.random.randint(0, rem_len)
        silence_part_left = silence_part[0: j]
        silence_part_right = silence_part[j: rem_len]
        wav = np.concatenate([silence_part_left, wav, silence_part_right])
    
    specgram = stft(wav, 16000, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
    phase = np.angle(specgram[2]) / np.pi
    amp = np.log1p(np.abs(specgram[2]))
    
    return np.stack([phase, amp], axis = 2)


# In[3]:


POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}
len(id2name)

def load_data(data_dir):
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))
    
    with open(os.path.join(data_dir, 'validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))
            
    possible = set(POSSIBLE_LABELS)
    train, val = [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_silence_':
                continue
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'
            
            label_id = name2id[label]
            
            sample = (label, label_id, uid, entry)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)
    columns_list = ['label', 'label_id', 'user_id', 'wav_file']
    train_df = pd.DataFrame(train, columns = columns_list)
    valid_df = pd.DataFrame(val, columns = columns_list)
    return train_df, valid_df


# In[4]:


train_df, valid_df = load_data('./data/')
train_df.head()
train_df.label.value_counts()
silence_files = train_df[train_df.label == 'silence']
train_df = train_df[train_df.label != 'silence']

from scipy.io import wavfile

def read_wav_file(fname):
    _, wav = wavfile.read(fname)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav

silence_data = np.concatenate([read_wav_file(x) for x in silence_files.wav_file.values])


# In[5]:


test_paths = glob(os.path.join('./data/', 'test/*wav'))
def test_generator(test_batch_size):
    while True:
        for start in range(0, len(test_paths), test_batch_size):
            x_batch = []
            end = min(start + test_batch_size, len(test_paths))
            this_paths = test_paths[start: end]
            for x in this_paths:
                x_batch.append(process_wav_file(x))
            x_batch = np.array(x_batch)
            yield x_batch


# In[6]:


filePath = 'model/20_800.h5'
model = load_model(filePath)

predictions = model.predict_generator(test_generator(64), int(np.ceil(len(test_paths)/64)))
classes = np.argmax(predictions, axis = 1)

submission = dict()
for i in range(len(test_paths)):
    fname, label = os.path.basename(test_paths[i]), id2name[classes[i]]
    submission[fname] = label
    
with open('starter_submission.csv', 'w') as fout:
    fout.write('fname,label\n')
    for fname, label in submission.items():
        fout.write('{},{}\n'.format(fname, label))

