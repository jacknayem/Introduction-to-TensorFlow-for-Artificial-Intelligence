# Exercise 1 (House_Prices)
In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.

So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.

How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.

Hint: Your network might work better if you scale the house price down. You don't have to give the answer 400...it might be better to create something that predicts the number 4, and then your answer is in the 'hundreds of thousands' etc.

```python
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
```
# Week 2 Quiz
**1. What’s the name of the dataset of Fashion images used in this week’s code?**
- Fashion MN
- Fashion Data
- Fashion MNIST **(Answer)**
- Fashion Tensors

**2. What do the above mentioned Images look like?**
- 28x28 Color
- 100x100 Color
- 82x82 Greyscale
- 28x28 Greyscale **(Answer)**

**3.How many images are in the Fashion MNIST dataset?**
- 70,000 **(Answer)**
- 10,000
- 42
- 60,000

**4. Why are there 10 output neurons?**
- To make it classify 10x faster
- Purely arbitrary
- There are 10 different labels **(Answer)**
- To make it train 10x faster

**5. What does Relu do?**
- It only returns x if x is less than zero
- For a value x, it returns 1/x
- It only returns x if x is greater than zero **(Answer)**
- It returns the negative of x

**6. Why do you split data into training and test sets?**
- To train a network with previously unseen data
- To test a network with previously unseen data **(Answer)**
- To make testing quicker
- To make training quicker

**7. What method gets called when an epoch finishes?**
- on_epoch_end **(Answer)**
- On_training_complete
- on_epoch_finished
- on_end

**8. What parameter to you set in your fit function to tell it to use callbacks?**
- callback=
- oncallback=
- callbacks= **(Answer)**
- oncallbacks=

# Exercise 2 (Number Identification)
In the course you learned how to do classificaiton using Fashion MNIST, a data set containing items of clothing. There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.

Some notes:

1. It should succeed in less than 10 epochs, so it is okay to change epochs= to 10, but nothing larger
2. When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
3. If you add any additional variables, make sure you use the same names as the ones used in the class
I've started the code for you below -- how would you finish it?
```python
import tensorflow as tf
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/mnist.npz"
# GRADED FUNCTION: train_mnist
def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if(logs.get('acc')>0.99):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True
    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    # YOUR CODE SHOULD START HERE
    x_train = x_train/255.0
    x_test = x_test/255.0
    callbacks = myCallback()
    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE
        tf.keras.layers.Flatten(input_shape = (28,28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        # YOUR CODE SHOULD END HERE
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(# YOUR CODE SHOULD START HERE
        x_train, y_train, epochs=10, callbacks=[callbacks]
              # YOUR CODE SHOULD END HERE
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]
    train_mnist()
```
# Week 3 Quiz
**1. What is a Convolution?**
- A technique to make images bigger
- A technique to isolate features in images **(Answer)**
- A technique to make images smaller
- A technique to filter out unwanted images

**2. What is a Pooling?**
- A technique to reduce the information in an image while maintaining features **(Answer)**
- A technique to isolate features in images
- A technique to combine pictures
- A technique to make images sharper

**3. How do Convolutions improve image recognition?**
- They isolate features in images **(Answer)**
- They make processing of images faster
- They make the image clearer
- They make the image smaller

**4. After passing a 3x3 filter over a 28x28 image, how big will the output be?**
- 31x31
- 25x25
- 28x28
- 26x26 **(Answer)**

**5. After max pooling a 26x26 image with a 2x2 filter, how big will the output be?**
- 56x56
- 26x26
- 28x28
- 13x13 **(Answer)**

**6. Applying Convolutions on top of our Deep neural network will make training:**
- Stay the same
- Faster
- Slower
- It depends on many factors. It might make your training faster or slower, and a poorly designed Convolutional layer may even be less efficient than a plain DNN! **(Answer)**

# Excersise 3 (Image Classification by Fasion MNIST)

```python
import tensorflow as tf
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
```
# Week 4 Quiz
**1. Using Image Generator, how do you label images?**

- You have to manually do it
- It’s based on the file name
- TensorFlow figures it out from the contents
- It’s based on the directory the image is contained in **(Answer)**

**2. What method on the Image Generator is used to normalize the image?**

- normalize_image
- rescale **(Answer)**
- Rescale_image
- normalize

**3. How did we specify the training size for the images?**

- The training_size parameter on the validation generator
- The target_size parameter on the training generator **(Answer)**
- The target_size parameter on the validation generator
- The training_size parameter on the training generator

**4. When we specify the input_shape to be (300, 300, 3), what does that mean?**

- Every Image will be 300x300 pixels, with 3 bytes to define color **(Answer)**
- There will be 300 images, each size 300, loaded in batches of 3
- There will be 300 horses and 300 humans, loaded in batches of 3
- Every Image will be 300x300 pixels, and there should be 3 Convolutional Layers

**5. If your training data is close to 1.000 accuracy, but your validation data isn’t, what’s the risk here?**

- No risk, that’s a great result
- You’re overfitting on your validation data
- You’re underfitting on your validation data
- You’re overfitting on your training data **(Answer)**

**6. Convolutional Neural Networks are better for classifying images like horses and humans because:**

- In these images, the features may be in different parts of the frame
- There’s a wide variety of horses
- There’s a wide variety of humans
- All of the above **(Answer)**

**7. After reducing the size of the images, the training results were different. Why?**

- We removed some convolutions to handle the smaller images **(Answer)**
- There was more condensed information in the images
- There was less information in the images
- The training was faster
