import tensorflow as tf
import numpy as np
import keras
import streamlit as st

class MLP():
    def __init__(self):
        super().__init__()
        self.model = keras.Sequential([
            tf.keras.layers.Dense(3, activation=tf.nn.relu, input_shape=[3]),
            tf.keras.layers.Dense(6, activation=tf.nn.relu),
            tf.keras.layers.Dense(3, activation=tf.nn.leaky_relu),
            tf.keras.layers.Dense(1)
        ])

        opt = keras.optimizers.legacy.Adam(learning_rate=0.001)
        self.model.compile(optimizer=opt,
                           loss='mse',  
                           metrics='mae')
    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def training(self, df, features, output_variable):

        progress_text = "Training in Progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        patience = 20
        min_delta=0.0001
        loss_patience = []
        loss_history = []
        val_loss_history = []
        for epoch in range(500):
            trained_model = self.model.fit(df[features], df[[output_variable]], 
                                        epochs=1, validation_split=0.2, verbose=0)
            loss_history.append(trained_model.history['loss'][0])
            val_loss_history.append(trained_model.history['val_loss'][0])
            if int(len(loss_history)/patience)>=1:
                    if np.mean(trained_model.history['loss'][-20:])/patience < min_delta:
                        break
            my_bar.progress(int((epoch + 1)/5), text=progress_text)
        my_bar.text('Training Completed.')
        return dict(loss=loss_history, val_loss=val_loss_history)
    
    def predict(self, df, features):
        predictions = self.model.predict(df[features], verbose=0)
        return predictions