#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


# In[3]:


def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))

def tanh(x):
    x = np.clip(x, -10, 10)
    return 2 / (1 + np.exp(-x)) - 1

def relu_grad_activation(x):
    return x > 0

def leaky_relu_grad_activation(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
    
def sigmoid_grad_activation(x):
    return x * (1 - x)

def tanh_grad_activation(x):
    return 1 - x**2

def softmax(x):
    x = np.array(x)
    x -= np.max(x, axis=1, keepdims=True)  # Stabilize
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def identity_activation(x):
    return x

def xavier_initialize(shape):
    return np.random.randn(*shape) * np.sqrt(2. / (shape[0] + shape[1]))

def he_initialize(shape):
    return np.random.randn(*shape) * np.sqrt(2. / shape[0])

def clip_gradients(grad, max_value=5.0):
    return np.clip(grad, -max_value, max_value)

def accuracy(y_hat, y):
    return np.mean(y_hat.reshape(1, -1) == y.reshape(1, -1))

def one_hot_encode_classes(y, num_classes):
    return np.eye(num_classes)[y]


# In[4]:


class InputLayer():
    def __init__(self, x=None):
        self.values = x
        self.n_samples, self.width = None, None
        
    def forward(self, training=True):
        if self.n_samples is None:
            self.n_samples, self.width = self.values.shape


# In[5]:


class BatchNormalization():
    def __init__(self, width, batch_momentum=0.9):
        self.gamma = np.ones((1, width))
        self.beta = np.zeros((1, width))
        self.running_mean = np.zeros((1, width))
        self.running_var = np.ones((1, width))
        self.batch_momentum = batch_momentum 

    def forward(self, x, training=True):
        if training:
            # Compute the mean and variance for the current batch
            self.mean = np.mean(x, axis=0, keepdims=True)
            self.var = np.var(x, axis=0, keepdims=True)

            # Normalize the input
            x_norm = (x - self.mean) / np.sqrt(self.var + 1e-8)

            # Update the running statistics
            self.running_mean = self.batch_momentum * self.running_mean + (1 - self.batch_momentum) * self.mean
            self.running_var = self.batch_momentum * self.running_var + (1 - self.batch_momentum) * self.var

        else:
            # Use running statistics during inference
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + 1e-8)

        # Scale and shift
        out = self.gamma * x_norm + self.beta
        return out, x_norm

    def backward(self, grad_output, x, x_norm):
        m = x.shape[0]  # Batch size

        # Gradients with respect to gamma and beta
        grad_gamma = np.sum(grad_output * x_norm, axis=0, keepdims=True)
        grad_beta = np.sum(grad_output, axis=0, keepdims=True)

        # Gradients with respect to x_norm
        grad_x_norm = grad_output * self.gamma

        # Gradients with respect to variance and mean
        grad_var = np.sum(grad_x_norm * (x - self.mean) * -0.5 * (self.var + 1e-8) ** (-3/2), axis=0, keepdims=True)
        grad_mean = np.sum(grad_x_norm * -1 / np.sqrt(self.var + 1e-8), axis=0, keepdims=True) + \
                    grad_var * np.sum(-2 * (x - self.mean), axis=0, keepdims=True) / m

        # Gradients with respect to x
        grad_x = grad_x_norm / np.sqrt(self.var + 1e-8) + grad_var * 2 * (x - self.mean) / m + grad_mean / m

        return grad_x, grad_gamma, grad_beta


# In[6]:


class CompLayer():
    def __init__(self, width, prev_layer=None, activation_function='relu', 
                 alpha=0.01, dropout_rate=0.2, batch_norm=True, 
                 batch_momentum=0.9, learning_rate=0.01):
        self.prev_layer = prev_layer
        self.width = width
        self.values = None
        self.grad = None
        self.weights = None
        self.grad_weights = None
        self.dropout_rate = dropout_rate
        self.dropout_mask = None
        self.velocity = None
        self.grad_squared_accum = None
        self.batch_norm = batch_norm
        self.batch_momentum = batch_momentum
        self.learning_rate = learning_rate

        activation_functions = {
            'relu': relu,
            'leaky_relu': lambda x: leaky_relu(x, alpha=alpha),
            'sigmoid': sigmoid,
            'tanh': tanh
        }
        grad_activations = {
            'relu': relu_grad_activation,
            'leaky_relu': lambda x: leaky_relu_grad_activation(x, alpha=alpha),
            'sigmoid': sigmoid_grad_activation,
            'tanh': tanh_grad_activation
        }
        
        self.activation_function = activation_functions[activation_function]
        self.grad_activation = grad_activations[activation_function]

        if self.batch_norm:
            # Instantiate BatchNormalization class
            self.batch_norm_layer = BatchNormalization(width=self.width, batch_momentum=self.batch_momentum)

    def forward(self, training=True):
        self.inputs = np.concatenate([self.prev_layer.values, 
                                      np.ones(shape=(self.prev_layer.values.shape[0], 1))], 
                                      axis=1)

        if self.weights is None:
            np.random.seed(0)
            if self.activation_function in [relu, leaky_relu]:
                self.weights = he_initialize((self.width, self.inputs.shape[1]))
            if self.activation_function in [sigmoid, tanh]:
                self.weights = xavier_initialize((self.width, self.inputs.shape[1]))

        self.z = self.inputs @ self.weights.T
        
        if self.batch_norm:
            self.z_, self.z_norm = self.batch_norm_layer.forward(self.z, training=training)
            
        else:
            self.z_ = self.z
            
        self.values = self.activation_function(self.z_)

        if training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.values.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.values *= self.dropout_mask        

    def backward(self):
        if self.dropout_rate > 0:
            self.grad *= self.dropout_mask
            
        grad_activation = self.grad * self.grad_activation(self.values)

        if self.batch_norm:
            # Backpropagate batch normalization
            grad_z, grad_gamma, grad_beta = self.batch_norm_layer.backward(grad_activation, self.z, self.z_norm)
            # Update scale and shift
            self.batch_norm_layer.gamma -= self.learning_rate * grad_gamma
            self.batch_norm_layer.beta -= self.learning_rate * grad_beta
        else:
            grad_z = grad_activation
        
        grad_input = clip_gradients(grad_activation @ self.weights[:, :-1])
        self.prev_layer.grad = grad_input

        self.grad_weights = clip_gradients(grad_activation.T @ self.inputs)        


# In[36]:


class LastLayer():
    def __init__(self, width=None, y=None, prev_layer=None, mode='classification'):
        self.prev_layer = prev_layer
        self.y = y
        np.random.seed(0)
        self.weights = None
        self.width=width
        self.grad_weights = 0
        self.velocity = None
        self.grad_squared_accum = None
        self.mode = mode

        output_activation_functions = {
            'regression': identity_activation,
            'classification': softmax
        }
        self.evaluation = output_activation_functions[mode]        

    def forward(self, training=True):
        if self.y.ndim == 1 or self.y.shape[1] == 1:  
            self.y = self.y.reshape(-1, 1)
            
        self.inputs = np.concatenate([self.prev_layer.values, 
                                      np.ones(shape=(self.prev_layer.values.shape[0], 1))], 
                                      axis=1)

        if self.weights is None:
            np.random.seed(0)
            if self.mode == 'classification':
                self.weights = xavier_initialize((self.width, self.inputs.shape[1]))
            if self.mode == 'regression':
                self.weights = he_initialize((self.width, self.inputs.shape[1]))
            
        self.values = self.evaluation(self.inputs @ self.weights.T)

    def backward(self):        
        self.grad = self.values - self.y
        grad_activation = self.grad

        grad_input =  clip_gradients(grad_activation @ self.weights[:, :-1])
        self.prev_layer.grad = grad_input

        self.grad_weights = clip_gradients(grad_activation.T @ self.inputs)

        

    def calculate_loss(self):
        if self.mode == 'classification':
            # Cross-entropy loss
            return -np.sum(np.sum(self.y * np.log(self.values + 1e-8), axis=1))
            
        if self.mode == 'regression':
            return np.sum((self.y - self.values) ** 2)


# In[37]:


def Forward(CompNodes, training=True):
    for comp_node in CompNodes:
        comp_node.forward(training=True)

def Backward(CompNodes):
    for comp_node in CompNodes:
        comp_node.backward()

def UpdateWeights(layers, batch_size, l1_lambda, l2_lambda, optimization, learning_rate, momentum, momentum2, iteration):
    for layer in layers:
        if layer.velocity is None:
            layer.velocity = np.zeros_like(layer.weights) 
            
        if layer.grad_squared_accum is None:
            layer.grad_squared_accum = np.zeros_like(layer.weights)
            
        l1_penalty = l1_lambda * np.sign(layer.weights)
        l2_penalty = l2_lambda * layer.weights

        total_grad = (layer.grad_weights + l1_penalty + l2_penalty) / batch_size

        if optimization == 'sgd':
            layer.weights -= learning_rate * total_grad

        if optimization == 'sgd_w_momentum':
            layer.velocity = momentum * layer.velocity + total_grad  # No (1 - momentum) term
            layer.weights -= learning_rate * layer.velocity 


        if optimization == 'RMSprop':
            layer.grad_squared_accum = momentum2 * layer.grad_squared_accum + (1 - momentum2) * (total_grad ** 2)
            v_hat = layer.grad_squared_accum / (1 - momentum2**iteration)
            layer.weights -= learning_rate * (total_grad / (np.sqrt(v_hat) + 1e-8))

        if optimization == 'adam':
            layer.velocity = momentum * layer.velocity + (1 - momentum) * total_grad
            layer.grad_squared_accum = momentum2 * layer.grad_squared_accum + (1 - momentum2) * (total_grad ** 2)
            # bias correction
            m_hat = layer.velocity / (1 - momentum**iteration)
            v_hat = layer.grad_squared_accum / (1 - momentum2**iteration) 
            layer.weights -= learning_rate * (m_hat / (np.sqrt(v_hat) + 1e-8))

        if optimization == 'nesterov':
            lookahead_weights = layer.weights - learning_rate * momentum * layer.velocity
            orig_weights = layer.weights
            layer.weights = lookahead_weights
            
            Forward(layers)
            Backward(reversed(layers[1:]))
            
            total_grad = (layer.grad_weights + l1_penalty + l2_penalty) / batch_size
            
            layer.weights = orig_weights
            
            layer.velocity = momentum * layer.velocity + total_grad
            layer.weights -= learning_rate * layer.velocity 
        
        layer.grad_weights = np.zeros_like(layer.weights)     


# In[38]:


class MultiLayerPerceptron():
    def __init__(self, width_list, 
                 n_batches, n_epochs,
                 batch_norm=True, batch_momentum=0.9, 
                 mode='classification', 
                 activation_function='relu', alpha=0.01,
                 early_stopping=False,
                 l1_lambda=0.0, l2_lambda=0.0,
                 dropout_rate=0.2,
                 optimization='sgd',
                 learning_rate=0.01,
                 momentum=0.9,
                 momentum2=0.999,
                 verbose=10):
        
        self.depth = len(width_list) + 1
        self.width_list = width_list
        self.batch_norm = batch_norm
        self.batch_momentum = batch_momentum
        self.n_batches = n_batches
        self.n_epochs = n_epochs
        self.best_n_epochs = None
        self.train_loss = None
        self.mode = mode
        self.activation_function = activation_function
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.optimization = optimization
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentum2 = momentum2
        self.verbose = verbose
        self.train_accuracy = []
        self.valid_accuracy = []

        self.layers = [None] * (self.depth + 1)
        self.layers[0] = InputLayer()
        
        for i in range(1, self.depth):
            self.layers[i] = CompLayer(width = self.width_list[i-1], 
                                       prev_layer=self.layers[i-1], 
                                       activation_function=self.activation_function,
                                       dropout_rate=self.dropout_rate,
                                       batch_norm=self.batch_norm, 
                                       batch_momentum=self.batch_momentum,
                                       learning_rate=self.learning_rate)
            
        self.layers[-1] = LastLayer(prev_layer=self.layers[-2], mode=self.mode)
        assert all(self.layers)
        

    def fit(self, X, y, X_valid, y_valid):
        num_classes = len(np.unique(y))
        y_ = one_hot_encode_classes(y, num_classes)
        y_valid_ = one_hot_encode_classes(y_valid, num_classes)
        
        self.layers[-1].width = num_classes
                
        batches = np.arange(0, len(X)+1, len(X) // self.n_batches)
        batches[-1] = len(X)
        
        for epoch in range(1, self.n_epochs+1):
            epoch_loss = 0
            for i in range(1, len(batches)):
                batch_size = batches[i] - batches[i-1]

                batch_data = (X[ batches[i-1]:batches[i] ], y_[ batches[i-1]:batches[i] ])
                batch_loss = self.train_batch(*batch_data)
            
                epoch_loss += batch_loss
                UpdateWeights(layers=self.layers[1:], 
                              batch_size=batch_size,
                              l1_lambda=self.l1_lambda, 
                              l2_lambda=self.l2_lambda,
                              optimization=self.optimization,
                              learning_rate=self.learning_rate,
                              momentum=self.momentum,
                              momentum2=self.momentum2,
                              iteration = epoch*self.n_batches + i)
                
            self.train_accuracy.append(self.score(X, y))
            self.valid_accuracy.append(self.score(X_valid, y_valid))

            if ( self.verbose != -1 ) & ( epoch % self.verbose == 0 ): 
                print(f'Epoch: {epoch}, Loss: {epoch_loss / self.n_batches:.3f}', flush=True)
                print(f'   Train Accuracy: {self.score(X, y)}', flush=True)               
                print(f'   Valid Accuracy: {self.score(X_valid, y_valid)}\n\n', flush=True)

    def train_batch(self, x_batch, y_batch):
        self.layers[0].values = x_batch
        self.layers[-1].y = y_batch
        Forward(self.layers, training=True)
        Backward(reversed(self.layers[1:]))
        loss = self.layers[-1].calculate_loss()
        l1_loss = self.l1_lambda * sum(np.sum(np.abs(layer.weights))for layer in self.layers[1:])
        l2_loss = self.l2_lambda * sum(np.sum(layer.weights**2)for layer in self.layers[1:])

        return (loss + l1_loss + l2_loss) / len(x_batch)
        
    def predict(self, X):
        self.layers[0].values = X
        Forward(self.layers, training=False)
        preds = self.layers[-1].values
        
        return np.argmax(preds, axis=1)

    def score(self, X, y):
        preds = self.predict(X)
        if self.mode == 'regression':
            return round(mean_absolute_error(preds, y), 3)
        
        if self.mode == 'classification':
            return round(accuracy(preds, y), 3)

    def get_train_accuracy(self):
        return self.train_accuracy

    def get_valid_accuracy(self):
        return self.valid_accuracy
 


# In[39]:


from sklearn import datasets

mnist = datasets.fetch_openml('mnist_784', version=1)

x_train = mnist.data[:60000].to_numpy()
y_train = mnist.target[:60000].astype(int).to_numpy()
x_test = mnist.data[60000:].to_numpy()
y_test = mnist.target[60000:].astype(int).to_numpy()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

x_train = x_train / 255.0
x_valid = x_valid / 255.0
x_test = x_test / 255.0

print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {x_train.shape}')

print(f'x_valid shape: {x_valid.shape}')
print(f'y_valid shape: {x_valid.shape}')

print(f'x_test shape: {x_test.shape}')
print(f'y_test shape: {x_test.shape}')


# In[42]:


mlp = MultiLayerPerceptron(
    width_list=[256],
    n_batches=16, 
    n_epochs=100, 
    batch_norm=True,
    batch_momentum=0.9,
    mode='classification', 
    activation_function='relu',
    alpha=1,
    l1_lambda=0.0001,
    l2_lambda=0.001,
    dropout_rate=0.2,
    optimization='adam',
    learning_rate=0.001,
    momentum=0.9,
    momentum2=0.999,
    verbose=10
)


# In[43]:


mlp.fit(x_train, y_train, x_valid, y_valid)


print(f'Final Train accuracy: {mlp.score(x_train, y_train)}')
print(f'Final Valid accuracy: {mlp.score(x_valid, y_valid)}')
print(f'Final Test accuracy: {mlp.score(x_test, y_test)}')


# In[ ]:


to

