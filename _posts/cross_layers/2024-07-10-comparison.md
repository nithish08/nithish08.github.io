---
layout: post
title: "Cross Layers vs Fully Connected Layers"
author: Nithish Bolleddula
twitter_image: 
---


# Generate Dataset


```python
import numpy as np
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def get_random_X_y_data(data_size=100_000):
      # data_size 
      # number of features = 3 
      X = np.random.randint(200, size=[data_size, 3]) / 200.
      y = X[:,0]**2 + X[:,0]*X[:,1] + X[:,1]*X[:,2] + X[:,2]**2

      return X, y


x, y = get_random_X_y_data()
num_train = 90000
train_x = x[:num_train]
train_y = y[:num_train]
eval_x = x[num_train:]
eval_y = y[num_train:]

```

# Fully Connected Net


```python
deepnet = tf.keras.Sequential([
      tf.keras.layers.Dense(3, activation="relu"),
      tf.keras.layers.Dense(3, activation="relu"),
      tf.keras.layers.Dense(3, activation="relu"),
      tf.keras.layers.Dense(1)
    ])


train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(1000)
eval_data = tf.data.Dataset.from_tensor_slices((eval_x, eval_y)).batch(1000)

epochs = 100
learning_rate = 0.4

deepnet.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adagrad(learning_rate))
deepnet.fit(train_data, epochs=epochs, verbose=False)

deepnet_result = deepnet.evaluate(eval_data, return_dict=False, verbose=False)

```

# Network with Cross Layer


```python
class CrossLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CrossLayer, self).__init__()
        self.dense_layer = tf.keras.layers.Dense(units)
    
    def call(self, prev_x, init_x):
        # performs
        # init_x . (w @ prev_x + b) + prev_x
        # . -> element wise multiplication
        # @ -> matrix multiplication
        # b is united in the self.w layer
        return init_x * (self.dense_layer(prev_x)) + prev_x
        

class CrossLayersStackedModel(tf.keras.Model):
    def __init__(self, num_features=3, num_layers=1):
        super(CrossLayersStackedModel, self).__init__()
        self.num_features = num_features
        self.cross_layers = [CrossLayer(num_features) for _ in range(num_layers)]
        self.dense = tf.keras.layers.Dense(1)
    
    def call(self, x):
        init_x = x
        prev_x = x # does it make a copy?
        for cross_layer in self.cross_layers:
            prev_x = cross_layer(prev_x, init_x)
        return self.dense(prev_x)

crossnet = CrossLayersStackedModel()


epochs = 100
learning_rate = 0.4

crossnet.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adagrad(learning_rate))
crossnet.fit(train_data, epochs=epochs, verbose=False)
```




    <keras.src.callbacks.history.History at 0x17ca6f210>




```python
def calc_metrics(model):
    train_rmse = model.evaluate(train_data, return_dict=False, verbose=False)
    eval_rmse = model.evaluate(eval_data, return_dict=False, verbose=False)
    return train_rmse, eval_rmse


baseline_train_rmse = np.sqrt(np.mean((train_y - np.mean(train_y))**2))
baseline_eval_rmse = np.sqrt(np.mean((eval_y - np.mean(train_y))**2))
print('baseline', baseline_train_rmse, baseline_eval_rmse)
print(calc_metrics(deepnet))
print(calc_metrics(crossnet))
```

    baseline 0.689292448648356 0.688865395052829
    (0.4751493036746979, 0.4744740128517151)
    (3.974694706698756e-08, 4.0458154160205595e-08)


# Comparison 

| Metric | Mean Predictor |  Deepnet | Crossnet |
|----------|----------|----------|----------|
| Number of Parameters | 0| 40 |16 |
| Train RMSE | 6.9e-1 | 2.4e-3  | 5.5e-11 |
| Eval RMSE | 6.9e-1 |2.4e-3 | 5.3e-11 |

References
- https://www.tensorflow.org/recommenders/examples/dcn
