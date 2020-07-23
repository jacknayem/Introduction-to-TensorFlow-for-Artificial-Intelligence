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

