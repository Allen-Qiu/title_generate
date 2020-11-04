# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:36:41 2020
training a title-generation model
@author: Allen Qiu
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Embedding 
from tensorflow.keras.layers import Bidirectional, Concatenate
from tensorflow.keras import backend as K
import tensorflow as tf
import pickle
from title_parameters import Hyparameters as hp
from title_dataset import TitleDataset

with open('titledataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# build model
# encoder
encoder_inputs=Input(shape=(hp.encoder_time_steps,), name="einput")
embed = Embedding(hp.vocab_size, hp.embed_size, name='embed')
encoder_input_embed = embed(encoder_inputs)
encoder=Bidirectional(GRU(hp.hidden_size,
                          return_sequences=True, 
                          return_state=True 
                          ),name='encoder')
encoder_outputs, encoder_states, _ = encoder(encoder_input_embed)

# decoder
decoder_inputs = Input(shape=(hp.decoder_time_steps,), name='dinput')
decoder_input_embed = embed(decoder_inputs)
decoder = GRU(hp.hidden_size, return_state=True, name='decoder')
c = K.mean(encoder_outputs, axis=1)
expanded_c=K.expand_dims(c,axis=1)
decoder_state = encoder_states
decoder_outputs = []
times=decoder_input_embed.shape[1]

attention_layer=Dense(hp.hidden_size, 
                      name='attention_layer',
                      input_shape=(hp.encoder_time_steps, 
                                   hp.encoder_output_hidden_size))
temp = attention_layer(encoder_outputs)
t = tf.transpose(temp,[0,2,1])

for i in range(times):
    one_decoder_input = K.concatenate([decoder_input_embed[:,i:i+1,:],expanded_c], axis=-1)
    one_output,decoder_state = decoder(one_decoder_input, initial_state=decoder_state)
    expanded_s_tm1=K.expand_dims(decoder_state,axis=1)
    E = tf.matmul(expanded_s_tm1,t)
    alpha = K.softmax(E)
    c = tf.matmul(alpha, encoder_outputs)
    expanded_c=K.expand_dims(c,axis=1)
    decoder_outputs.append(K.expand_dims(one_output,axis=1))

concat_decoder_outputs=Concatenate(axis=1)(decoder_outputs)    
decoder_dense = Dense(hp.vocab_size, activation='softmax', name='outputs')
outputs = decoder_dense(concat_decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], outputs)
loss=tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer='adam', 
              loss=loss, 
              metrics=['sparse_categorical_accuracy'])
model.fit([dataset.encoder_input_train, dataset.decoder_input_train], 
          dataset.decoder_target_train,
          batch_size=hp.batch_size, epochs=hp.epochs, verbose=1)

# inference encoder model
encoder_model=Model(inputs=encoder_inputs, outputs=[encoder_outputs,encoder_states])
encoder_model.save('encoder.h5')

# inference decoder model
decoder_inputs_infer = Input(shape=(1, ), name='dinput')
decoder_state_infer = Input(shape=(hp.hidden_size,), 
                            name='sinput')
encoder_outputs_infer = Input(shape=(hp.encoder_time_steps,
                                     hp.encoder_output_hidden_size),
                              name='einput')

decoder_input_embed_infer = embed(decoder_inputs_infer)
temp_infer = attention_layer(encoder_outputs_infer)
t_infer = tf.transpose(temp_infer,[0,2,1])
expanded_s_tm1_infer=K.expand_dims(decoder_state_infer,axis=1)
E_infer = tf.matmul(expanded_s_tm1_infer,t_infer)
alpha_infer = K.softmax(E_infer)
expanded_c_infer = tf.matmul(alpha_infer, encoder_outputs_infer)

one_decoder_input_infer = K.concatenate([decoder_input_embed_infer,
                                         expanded_c_infer], axis=-1)
one_output_infer,state_infer = decoder(one_decoder_input_infer, 
                                       initial_state=decoder_state_infer)

decoder_outputs_infer=K.expand_dims(one_output_infer,axis=1)

output_layer_infer = decoder_dense(decoder_outputs_infer)
outputs_infer=K.argmax(output_layer_infer,axis=-1)

decoder_model = Model(
    inputs=[decoder_inputs_infer,decoder_state_infer, encoder_outputs_infer],
    outputs=[outputs_infer,state_infer])

decoder_model.save('decoder.h5')
