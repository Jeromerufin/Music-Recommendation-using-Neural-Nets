import pandas as pd
import pickle
import glob
from os import path
import numpy as np
from scipy import ndimage, misc
import matplotlib.pyplot as plt

# Keras to build the model
# import keras
# from keras import backend as K
# from keras.models import Model, model_from_json
# from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
# from keras.layers.merge import Concatenate
# from keras.layers.core import Flatten, Dense, Dropout, Activation
# from keras.preprocessing.sequence import pad_sequences
# from keras.callbacks import EarlyStopping

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Flatten,Input, Dense, Dropout, Activation,Conv2D, MaxPooling2D, UpSampling2D,Concatenate
from tensorflow.python.keras.models import Sequential,Model, model_from_json

#AutoEncoder

# Initialize the input shape
input_layer = Input(shape=(16,512,1))

# Encoder: Increase dimensionality in convolutions, then reduce to 16,16,1
# encoded = Conv2D(248, (3, 3), activation='relu', padding='same')(input_layer)
# encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)
# encoded = Conv2D(256, (3, 3), activation='relu', padding='same')(input_layer)
# encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)
encoded = Conv2D(256, (3, 3), activation='relu', padding='same')(input_layer)
encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)
encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded)
encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)
encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D((2, 2), padding='same')(encoded)
# encoded = Conv2D(1, (3, 3), activation='relu', padding='same')(encoded)
# encoded = MaxPooling2D((2, 2), padding='same')(encoded)

# Reduced dimensionailty layer that can be taken as an embedding of the original image
encoded_layer = Flatten(name = 'encoded')(encoded)

# Decoder: Reverse the process of the encoder
decoded = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded) 
decoded = UpSampling2D((2,2))(decoded)
decoded = Conv2D(64, (3, 3), activation='relu', padding='same')(decoded) 
decoded = UpSampling2D((2,2))(decoded)
decoded = Conv2D(128, (3, 3), activation='relu', padding='same')(decoded) 
decoded = UpSampling2D((2,2))(decoded)
decoded = Conv2D(256, (3, 3), activation='relu', padding='same')(decoded) 
decoded = UpSampling2D((2,2))(decoded)

# Final layer that is the same shape as the input. This is the result that should return the same image as the input
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)
# Check the output is the same shape as the input
print ('shape of decoded', K.int_shape(decoded))

# Initialize input and output
autoencoder = Model(input_layer, decoded)

# Model that will return the embedding rather than the predicted image, but trained using the autoencoded model
encoder = Model(input_layer, encoded_layer)
print ('shape of encoded', K.int_shape(encoded))

# Save the architextures as strings
json_autoencoder = autoencoder.to_json()
json_encoder = encoder.to_json()

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

# model_checkpoints = tensorflow.keras.callbacks.ModelCheckpoint("checkpoint-{val_loss:.3f}.h5", monitor='val_loss', verbose=0, save_best_only=False,save_weights_only=False, mode='auto', save_freq ='epoch')

#load audio array
with open('audio_array.pkl','rb') as picklefile:
    audio_array = pickle.load(picklefile)

# Establish the models from the jsons
autoencoder = model_from_json(json_autoencoder)
encoder = model_from_json(json_encoder)

# Fit the Model
autoencoder.compile(optimizer='adam', loss='mse')
print('Fitting model -> autoencoder')
autoencoder.fit(audio_array, audio_array, epochs=50,batch_size=128, validation_split=.15, callbacks=[early_stopping],verbose=1)

# Save the model and and encoder
autoencoder_name = 'autoencoder_model_2.h5'
autoencoder.save(autoencoder_name)
encoder_name = 'encoder_model_2.h5'
encoder.save(encoder_name)

# Delete the images and models to make room in memory
del(features)
del(autoencoder)
del(encoder)