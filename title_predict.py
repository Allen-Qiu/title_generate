# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:51:37 2020

@author: Allen Qiu
"""
from tensorflow.keras.models import load_model
import numpy as np
import random
import pickle
from title_parameters import Hyparameters as hp

with open('titledataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

encoder_model=load_model('encoder.h5')
decoder_model=load_model('decoder.h5')

# predict
def decode_sequence(input_seq):
    encoder_outputs,encoder_state = encoder_model.predict(input_seq)
    decoder_input_seq = dataset.dic_token_index['<start>']
    stop_condition = False
    decoded_sentence = []
    decoder_state = encoder_state
    decoder_input_seq=np.array([[decoder_input_seq]])
    
    while not stop_condition:
        outputs, decoder_state = decoder_model.predict(
            [decoder_input_seq, decoder_state, encoder_outputs])

        decoder_input_seq=outputs
        if outputs[0,0]==0: break
        token = dataset.dic_index_token[outputs[0,0]]
        decoded_sentence.append(token)

        if (token == '<end>' or
           len(decoded_sentence) > hp.max_decoder_seq_length):
            stop_condition = True


    return decoded_sentence

def get_original_title(one_encoded_text):
    text=[dataset.dic_index_token[item] for item in one_encoded_text if item>0]
    return ' '.join(text)

sampled_idx=random.sample(range(len(dataset.encoder_input_dev)),5)
for idx in sampled_idx:
    input_seq = dataset.encoder_input_dev [idx: idx + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print(get_original_title(dataset.decoder_input_dev [idx]))
    print('Decoded sentence:', decoded_sentence)

# explore fittness of model
sampled_idx=random.sample(range(len(dataset.encoder_input_train)),5)
for idx in sampled_idx:
    input_seq = dataset.encoder_input_train [idx: idx + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print(get_original_title(dataset.decoder_input_train [idx]))
    print('Decoded sentence:', decoded_sentence)

