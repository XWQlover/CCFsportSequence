# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras import Sequential
from tensorflow.keras import Model
def get_model():
    X = Input(shape=(240))
    dense =Sequential([Dense(64,activation="relu"),
                Dense(2,activation="sigmoid")])
    x = dense(X)
    y = tf.nn.softmax(x)

    model = Model(inputs=[X],outputs=[y])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"])

    return model

if __name__=="__main__":
    pass