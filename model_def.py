import tensorflow as tf
import numpy as np
import keras
import streamlit as st
import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

class MLP():
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(3, activation=tf.nn.relu, input_shape=[3]),
            tf.keras.layers.Dense(6, activation=tf.nn.relu),
            tf.keras.layers.Dense(3, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(1)
        ])

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=opt,
                           loss='mse',  
                           metrics='mae')
    
    def load_model(self):
        print("model loaded.")
        return tf.keras.models.load_model('save-models/my-model.keras')

    def training(self, df, features, output_variable):

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00001, patience=20)
        trained_model = self.model.fit(df[features], df[[output_variable]], 
                                       epochs=500,validation_split=0.2,callbacks=callback)
        self.model.save('save-models/my-model.keras')
        # self.model.save_weights("/checkpoints/my_checkpoint")
        return trained_model.history
    
    def predict(self, df, features):
        predictions = self.model.predict(df[features], verbose=0)
        return predictions