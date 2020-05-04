# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:33:54 2020
save all hyper-parameters in this class
@author: Allen Qiu
"""

class Hyparameters:
    encoder_time_steps         = 150
    decoder_time_steps         = 20
    vocab_size                 = 5000
    embed_size                 = 100
    hidden_size                = 500  # encoder, decoder have same hidden_size
    encoder_output_hidden_size = 2*hidden_size # use a bidirectional RNN
    max_decoder_seq_length     = 20
    batch_size                 = 32
    epochs                     = 100

