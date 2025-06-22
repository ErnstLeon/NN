# Neural Network (NN)

This document provides a brief explanation of the **Neural Network** implemented here.
The model is designed for **classification** of input vectors into one of several classes (e.g., recognizing handwritten digits from 0 to 9).

## How to use

This program lets you train a fully connected neural network, allowing you to set the number of layers and neurons per layer. It also lets you store your trained model or load pretrained ones.

While you can choose an activation function for the hidden layers, the activation function for the last layer is set to be the **softmax** function, and the loss function is set to the **cross-entropy** loss.

### Initializing and Training a Neural Network

#### Initialize the Network

```cpp
NN::network<activation_function<datatype>, number_of_layers> network({neurons_per_layer}, random_initalitzation)
```

- `activation_function` can be either `NN::ReLU` or `NN::Sigmoid` for the hidden layers.
- `datatype` is usually `float` or `double` and used throughout the network as storrage type.
- `neurons_per_layer` is a `std::array<size_t, number_of_layers>`, defining the number of neurons in each layer.
- If `random_initalitzation == true` (default), the weights and biases are initialized randomly.

:warning: The first layer must match the number of input features, and the last layer must match the number of output classes.

#### Train the Network

```cpp
// training either with Gradient Descent (stepsize)
//NN::Optimizer::Gradient_Descent<T> optmizer{0.01};

// training either with Adam (stepsize, beta1, beta2, eplsilon)
NN::Optimizer::Adam_Optimizer<T> optmizer{0.01, 0.9, 0.999, 1e-8};

train_error = network.learn(dataset_train, optimizer, batch_size, number_of_epochs);
```

- `train_error`: the training error (cross-entropy loss) after the last optimization step.
- `dataset_train`: a `std::vector<std::pair<std::vector<T>, std::vector<T>>>` of inputs and one-hot encoded labels.
- `optimizer`:  An optimizer instance, either Gradient Descent or Adam.
- `Parameters`:
    - `batch_size`: number of training samples per batch
    - `number_of_epochs`: how many times the full dataset is passed through the network

#### Evaluate Performance

```cpp
test_error = network.assess(dataset_test);
```

- `test_error` the average error on the test dataset.
- `dataset_test`: a `std::vector<std::pair<std::vector<T>, std::vector<T>>>` of input-label pairs.

#### Make Predictions

```cpp
label = network.evaluate(sample);
```

- `label`: the predicted class probabilities as a vector (softmax output).
- `sample`: a single input as `std::vector<T>`.

#### Example (e.g. MNIST)
```cpp
NN::network<NN::Sigmoid<float>, 4> network({784, 128, 128, 10}, true);

NN::Optimizer::Adam_Optimizer<T> optmizer{0.001, 0.9, 0.999, 1e-8};

float train_error = network.learn(dataset_train, optmizer, 32, 20);
float test_error = network.assess(dataset_test);

std::vector<float> label = network.evaluate(sample);
```

#### Storing and Loading a trained model
```cpp
network.store("model.out");

NN::network<NN::Sigmoid<float>, 4> network{};
network.load("model.out");
```
These function store and load the biases and weight matrices for all layers.

### Full Example

A complete example of training a neural network using the MNIST dataset can be found in `example/main.cpp`.

This example demonstrates:

- Initializing the network
- Loading the dataset
- Training the model
- Evaluating its performance
- Making predictions
- Saving the trained model

The model is available in `models/MNIST_Sigmoid_4_Layers.out`

```cpp
g++ -I../include -std=c++20 -O3 -Wall -march=native -o main main.cpp
```
- Add `-fopenmp` for parallel training (only if supported)
### Requirements

- C++20 or later
- OpenMP (optional, for parallel training)

### Parallelization (optional, but highly recommended)

Training uses OpenMP to parallelize gradient computations per sample within each batch.
Compile with:

```cpp
g++ -fopenmp -I../include -std=c++20 -O3 -Wall -march=native -o main main.cpp
```

:warning: use `export OMP_NUM_THREADS=` to adjust the number of threads to your system.
:warning: using clang on MacOS, you need to manually install OpenMP, for example using homebrew:

```bash
brew install libomp
```

and then compile with 

```cpp
g++ -Xpreprocessor -fopenmp -I../include -L/opt/homebrew/opt/libomp/lib -lomp -I../include -std=c++20 -O3 -Wall -march=native -o main main.cpp
```

## Mathematical Details

### Forward Pass

For each layer $l$, the output is computed as:

```math
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} \\
a^{(l)} = \sigma(z^{(l)})
```

Where:

- $W^{(l)}$: weight matrix of layer $l$ 
- $b^{(l)}$: bias vector of layer $l$ 
- $\sigma$: activation function (ReLU or Sigmoid)  
- $a^{(l)}$: activation output of layer $l$  
- $a^{(0)}$: input vector to the network

The last layer uses the **softmax** function:

```math
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
```

### Loss Function

The network uses the **cross-entropy** loss:

```math
L = - \sum_i y_i \log(\hat{y}_i)
```

Where:

- $y_i$: true label (one-hot encoded)  
- $\hat{y}_i$: predicted probability from softmax

### Backpropagation

Gradients of the weights and biases are computed using the chain rule:

```math
\delta^{(l)} = \left(W^{(l+1)}\right)^T \delta^{(l+1)} \odot \sigma'(z^{(l)})
```

```math
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \left(a^{(l-1)}\right)^T
```

```math
\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}
```

Where:

- $\delta^{(l)}$: error term of layer $l$ 
- $\odot$: element-wise (Hadamard) product  
- $\sigma'(z^{(l)})$: derivative of activation function
