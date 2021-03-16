import keras
from keras import Model
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, BatchNormalization, Activation, Input, Add


def ResNet(input_shape):
    x_input = Input(input_shape)
    x = Conv2D(64, 7)(x_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)
    x_shortcut = x
    x = Conv2D(64, 3, padding="same")(x_shortcut)
    x = BatchNormalization(axis=3)(x_shortcut)
    x = Activation("relu")(x)
    x = Conv2D(64, 3,padding="same")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    x = Conv2D(64, 3, padding="same")(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    x = Add()([x, x_shortcut])        
    x = Activation("relu")(x)
    return Model(x_input, x, "resnet")
        
     
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
X = np.array([1,2,3,4,5,6,7,8,9,10])
y = X**2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)





    
model = keras.Sequential()
model.add(Dense(10))
model.compile(optimizer="sgd", 
              loss="mse")
