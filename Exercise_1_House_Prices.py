import tensorflow as tf
import numpy as np
from tensorflow import keras
# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,10.0], dtype=float)
    ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5,5.5], dtype=float)
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd',loss='mean_squared_error')
    model.fit(xs,ys, epochs=500)
    return model.predict(y_new)[0]
prediction = house_model([7.0])
print(prediction)
