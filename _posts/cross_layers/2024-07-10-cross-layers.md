---
layout: post
title: "Cross Layers"
author: Nithish Bolleddula
tags: [dnn, feature interaction]
---


# Cross layers implementation


Cross layers help neural network learn explicit feature interactions. Compared to fully connected neural networks which models implicit feature interactions. We will program the cross layer using tensorflow and PyTorch in this blog and demonstrate it's capability.

Cross layer implementation - 

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
        # b is the bias unit
        return init_x * (self.dense_layer(prev_x)) + prev_x
```

Multiple cross layers can be stacked into a single model like this - 

```python
class CrossLayersStackedModel(tf.keras.Model):
    def __init__(self, num_features=3, num_layers=1):
        super(CrossModel1, self).__init__()
        self.num_features = num_features
        self.cross_layers = [CrossLayer(num_features) for _ in range(num_layers)]
        self.dense = tf.keras.layers.Dense(1) # output layer for regression problem
    
    def call(self, x):
        init_x = x
        prev_x = x # does it make a copy?
        for cross_layer in self.cross_layers:
            prev_x = cross_layer(prev_x, init_x)
        return self.dense(prev_x)
```




# Comparison with fully connected network


Let's say we are trying to model y, which depends on three features x_1, x_2 and x_3

$$
y = x_1^2 + x_1 \cdot x_2 + x_2 \cdot x_3 + x_3^2
$$


## Metric Comparison 

| Metric | Mean Predictor |  Deepnet | Crossnet |
|----------|----------|----------|----------|
| Number of Parameters | 0| 40 |16 |
| Train RMSE | 6.9e-1 | 2.4e-3  | 5.5e-11 |
| Eval RMSE | 6.9e-1 |2.4e-3 | 5.3e-11 |


Full notebook - [link](/2024/07/10/comparison.html)