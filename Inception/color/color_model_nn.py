from keras import regularizers
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv('datasets/rgb_colorname.csv')
dataset = pd.get_dummies(df, columns=['label'])


# train = 80%,  random_state = any int value means every time when you run your program you will get the same output for train and test dataset, random_state is None by default which means every time when you run your program you will get different output because of splitting between train and test varies within
train_dataset = dataset.sample(frac=0.8, random_state=9)
# remove train_dataset from dataframe to get test_dataset
test_dataset = dataset.drop(train_dataset.index)


# print(train_dataset)


train_labels = pd.DataFrame([train_dataset.pop(x) for x in ['label_Red', 'label_Green', 'label_Blue', 'label_Yellow',
                            'label_Orange', 'label_Pink', 'label_Purple', 'label_Brown', 'label_Grey', 'label_Black', 'label_White']]).T
test_labels = pd.DataFrame([test_dataset.pop(x) for x in ['label_Red', 'label_Green', 'label_Blue', 'label_Yellow',
                           'label_Orange', 'label_Pink', 'label_Purple', 'label_Brown', 'label_Grey', 'label_Black', 'label_White']]).T


model = keras.Sequential([
    layers.Dense(3, kernel_regularizer=regularizers.l2(0.001), activation='relu',
                 input_shape=[len(train_dataset.keys())]),  # inputshape=[3]
    layers.Dense(24, kernel_regularizer=regularizers.l2(
        0.001), activation='relu'),
    layers.Dense(11)
])


optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()

history = model.fit(x=train_dataset, y=train_labels,
                    validation_split=0.2,
                    epochs=1000,
                    batch_size=256,
                    verbose=1,
                    shuffle=True)

model.save('colormodel.h5')

# reference: https://github.com/AjinkyaChavan9/RGB-Color-Classifier-with-Deep-Learning-using-Keras-and-Tensorflow/blob/master/RGB%20Color%20Classifier%20ML%20Model/color_classifier_ml_model.py
