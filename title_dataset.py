# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:09:17 2020

@author: Allen Qiu
"""
import pickle
import json
import numpy as np
from title_parameters import Hyparameters  as hp
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TitleDataset:
    
    def __init__(self,vocab_size,encoder_time_steps, decoder_time_steps):
        self.dic_token_index = None
        self.dic_index_token = None
        title = []
        text  = []

        with open('title.json') as f:
            for line in f.readlines():
                dic=json.loads(line)
                title.append('<start> '+dic['Title']+' <end>')
                text.append(dic['Abstract'])
        
        t=Tokenizer(num_words=vocab_size,
                    oov_token=None, 
                    filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',)
        t.fit_on_texts(text+title)
        self.dic_token_index = t.word_index
        self.dic_index_token = {t.word_index[item]:item for item in t.word_index}
        
        encoded_texts=t.texts_to_sequences(text)
        encoded_titles=t.texts_to_sequences(title)
        
        encoded_inputs  = []
        encoded_targets = []
        
        for item in encoded_titles:
            encoded_inputs.append(item[0:(len(item)-1)])
            encoded_targets.append(item[1:(len(item))])
        
        encoder_input  = pad_sequences(encoded_texts, 
                                            maxlen=encoder_time_steps, 
                                            padding='post')
        decoder_input  = pad_sequences(encoded_inputs, 
                                            maxlen=decoder_time_steps, 
                                            padding='post')
        decoder_target = pad_sequences(encoded_targets, 
                                            maxlen=decoder_time_steps, 
                                            padding='post')
        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(encoder_input)))
        encoder_input_shuffled  = encoder_input [shuffle_indices]
        decoder_input_shuffled  = decoder_input [shuffle_indices]
        decoder_target_shuffled = decoder_target[shuffle_indices]
        
        # Split train/test set
        self.encoder_input_train = encoder_input_shuffled[:-100]
        self.encoder_input_dev   = encoder_input_shuffled[-100:]
        self.decoder_input_train  = decoder_input_shuffled[:-100]
        self.decoder_input_dev    = decoder_input_shuffled[-100:]
        self.decoder_target_train = decoder_target_shuffled[:-100]
        self.decoder_target_dev   = decoder_target_shuffled[-100:]


dataset=TitleDataset(hp.vocab_size,hp.encoder_time_steps,hp.decoder_time_steps)
with open('titledataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
# json.dump(dataset)





