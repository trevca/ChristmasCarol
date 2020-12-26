#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Activation, Flatten, Dropout, Dense, Embedding, TimeDistributed, CuDNNLSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# In[2]:


data = pd.read_csv("carol_data.csv")
data = data.drop("Unnamed: 0", axis = 1)
all_text = ""
for i in range(len(data['song_lyrics'])):
    data['song_titles'][i] = data['song_titles'][i].lower()
    data['song_lyrics'][i] = data['song_lyrics'][i].lower()
    all_text += data['song_lyrics'][i] + '\n'


# In[3]:


print(all_text)


# In[4]:


# Mapping chars to ints :
chars = sorted(list(set(all_text)))
int_chars = dict((i, c) for i, c in enumerate(chars))
chars_int = dict((i, c) for c, i in enumerate(chars))


# In[5]:


n_chars = len(all_text)
n_vocab = len(chars)
print('Total Characters :' , n_chars) # number of all the characters in lyricsText.txt
print('Total Vocab :', n_vocab) # number of unique characters


# In[6]:


# process the dataset:
seq_len = 100
data_X = []
data_y = []

for i in range(0, n_chars - seq_len, 1):
    # Input Sequeance(will be used as samples)
    seq_in  = all_text[i:i+seq_len]
    # Output sequence (will be used as target)
    seq_out = all_text[i + seq_len]
    # Store samples in data_X
    data_X.append([chars_int[char] for char in seq_in])
    # Store targets in data_y
    data_y.append(chars_int[seq_out])
n_patterns = len(data_X)
print( 'Total Patterns :', n_patterns)


# In[7]:


# Reshape X to be suitable to go into LSTM RNN :
X = np.reshape(data_X , (n_patterns, seq_len, 1))
# Normalizing input data :
X = X/ float(n_vocab)
# One hot encode the output targets :
y = np_utils.to_categorical(data_y)


# In[8]:


LSTM_layer_num = 4 # number of LSTM layers
layer_size = [256,256,256,256] # number of nodes in each layer
model = Sequential()
model.add(LSTM(layer_size[0], input_shape =(X.shape[1], X.shape[2]), return_sequences = True))
for i in range(1,LSTM_layer_num) :
    model.add(LSTM(layer_size[i], return_sequences=True))
model.add(Flatten())
model.add(Dense(y.shape[1]))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
model.summary()


# In[9]:


checkpoint_name = 'Weights-LSTM-improvement-{epoch:03d}-{loss:.5f}-bigger.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose = 1, save_best_only = True, mode ='min')
callbacks_list = [checkpoint]


# In[ ]:





# In[ ]:


# # Fit the model :
model_params = {'epochs':30,
                'batch_size':128,
                'callbacks':[],
                'verbose':1,
                'validation_split':0.2,
                'validation_data':None,
                'shuffle': True,
                'initial_epoch':0,
                'steps_per_epoch':None,
                'validation_steps':None}
model.fit(X,
          y,
          epochs = model_params['epochs'],
           batch_size = model_params['batch_size'],
           callbacks= model_params['callbacks'],
           verbose = model_params['verbose'],
           validation_split = model_params['validation_split'],
           validation_data = model_params['validation_data'],
           shuffle = model_params['shuffle'],
           initial_epoch = model_params['initial_epoch'],
           steps_per_epoch = model_params['steps_per_epoch'],
           validation_steps = model_params['validation_steps'])

