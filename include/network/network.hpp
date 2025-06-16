#ifndef NETWORK_H
#define NETWORK_H

#include <algorithm>
#include <array>
#include <cstring>
#include <execution>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "../learning/loss.hpp"
#include "../learning/optimization.hpp"

#include "../utility/custom_concepts.hpp"
#include "../utility/debug.hpp"

namespace NN
{
template<typename Activation, size_t NUM_LAYERS, typename T = typename Activation::type>
requires callable_with<Activation, T, T> && derivative_callable_with<Activation, T, T> && (NUM_LAYERS >= 3)
class network {
private:

    Debug_Mode debug_mode{Debug_Mode::Release};
    const Activation activation_func{};
    
    std::array<size_t, NUM_LAYERS> neurons_per_layer{};

    size_t num_input_neurons{};
    size_t num_output_neurons{};
    size_t max_neurons{};

    std::array<std::vector<T>, NUM_LAYERS - 1> biases{};
    std::array<std::vector<T>, NUM_LAYERS - 1> weights{};

    void forward_pass(const std::vector<T> &, std::array<std::vector<T>, NUM_LAYERS - 1> &, 
    std::array<std::vector<T>, NUM_LAYERS - 1> &) const;

    void backward_pass(const std::array<std::vector<T>, NUM_LAYERS - 1> &, 
    const std::array<std::vector<T>, NUM_LAYERS - 1> &, std::array<std::vector<T>, NUM_LAYERS - 1> &, 
    std::array<std::vector<T>, NUM_LAYERS - 1> &, const std::vector<T> &, const std::vector<T> &) const;

    T compute_loss(const std::vector<T> &, const std::vector<T> &) const;

public: 

    network() = default;

    network(std::initializer_list<size_t> neuron_list, bool rnd = false, Debug_Mode debug_mode = Debug_Mode::Release)
    : activation_func{activation_func}, debug_mode{debug_mode}
    {
        if(neuron_list.size() !=  NUM_LAYERS)
        {
            throw std::invalid_argument("The number of layers does not match with the provided.");
        }

        std::copy(neuron_list.begin(), neuron_list.end(), neurons_per_layer.begin());

        num_input_neurons = neurons_per_layer.front();
        num_output_neurons = neurons_per_layer.back();
        max_neurons = *std::max_element(neurons_per_layer.begin(), neurons_per_layer.end());

        for(size_t layer = 0; layer < NUM_LAYERS - 1; ++layer)
        {
            biases[layer].resize(neurons_per_layer[layer + 1]);
            weights[layer].resize(neurons_per_layer[layer] * neurons_per_layer[layer + 1]);
        }

        if(rnd){
            std::mt19937 gen(100);
            std::uniform_real_distribution<> dist(-1.0, 1.0);
            for(size_t layer = 0; layer < NUM_LAYERS - 1; ++layer)
            {
                std::generate(biases[layer].begin(), biases[layer].end(), [&](){return dist(gen);});
                std::generate(weights[layer].begin(), weights[layer].end(), [&](){return dist(gen);});
            }
        }
    };

    T learn(const std::vector<std::pair<std::vector<T>, std::vector<T>>> &, size_t, size_t num_epochs = 1000, T step_size = 0.01);
    T assess(const std::vector<std::pair<std::vector<T>, std::vector<T>>> &) const;
    
    std::vector<T> evaluate(const std::vector<T> & input) const;

    void store(const std::string &) const;
    void load(const std::string &);
};

template<typename Activation, size_t NUM_LAYERS, typename T>
requires callable_with<Activation, T, T> && derivative_callable_with<Activation, T, T> && (NUM_LAYERS >= 3)
T network<Activation, NUM_LAYERS, T>::learn(
    const std::vector<std::pair<std::vector<T>, std::vector<T>>> & dataset_orig, 
    size_t batch_size, size_t num_epochs, T step_size)
{
    for(const auto & [input, true_output] : dataset_orig)
    {
        if (input.size() != neurons_per_layer[0])
            throw std::invalid_argument{"Input does not match input layer size."};
        if (true_output.size() != neurons_per_layer.back())
            throw std::invalid_argument{"Output does not match output layer size."};
    }

    auto dataset = dataset_orig;

    std::mt19937 gen(100);
    size_t dataset_size = dataset.size();
    size_t num_batches = (dataset_size + batch_size - 1) / batch_size;

    std::array<std::vector<T>, NUM_LAYERS - 1> deriv_biases{};
    std::array<std::vector<T>, NUM_LAYERS - 1> deriv_weights{};

    for (size_t i = 0; i < NUM_LAYERS - 1; ++i) {
        deriv_biases[i].assign(biases[i].size(), static_cast<T>(0));
        deriv_weights[i].assign(weights[i].size(), static_cast<T>(0));
    }

    T train_error = static_cast<T>(0);

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) 
    {
        std::shuffle(dataset.begin(), dataset.end(), gen);

        for (size_t b = 0; b < num_batches; ++b) 
        {
            size_t current_batch_size = std::min(batch_size, dataset_size - b * batch_size);

            for (size_t i = 0; i < NUM_LAYERS - 1; ++i) {
                std::fill(deriv_biases[i].begin(), deriv_biases[i].end(), static_cast<T>(0));
                std::fill(deriv_weights[i].begin(), deriv_weights[i].end(), static_cast<T>(0));
            }

            #pragma omp parallel shared(deriv_biases, deriv_weights)
            {
                std::array<std::vector<T>, NUM_LAYERS - 1> thread_deriv_biases;
                std::array<std::vector<T>, NUM_LAYERS - 1> thread_deriv_weights;

                for (size_t i = 0; i < NUM_LAYERS - 1; ++i) {
                    thread_deriv_biases[i].assign(biases[i].size(), static_cast<T>(0));
                    thread_deriv_weights[i].assign(weights[i].size(), static_cast<T>(0));
                }

                std::array<std::vector<T>, NUM_LAYERS - 1> local_activations;
                std::array<std::vector<T>, NUM_LAYERS - 1> local_weighted_inputs;
                std::array<std::vector<T>, NUM_LAYERS - 1> local_deriv_biases;
                std::array<std::vector<T>, NUM_LAYERS - 1> local_deriv_weights;

                for (size_t i = 0; i < NUM_LAYERS - 1; ++i) {
                    local_activations[i].resize(biases[i].size());
                    local_weighted_inputs[i].resize(biases[i].size());
                    local_deriv_biases[i].resize(biases[i].size());
                    local_deriv_weights[i].resize(weights[i].size());
                }

                #pragma omp for nowait
                for (size_t sample = 0; sample < current_batch_size; ++sample) 
                {
                    size_t sample_index = b * batch_size + sample;

                    const auto & input = dataset[sample_index].first;
                    const auto & true_output = dataset[sample_index].second;

                    forward_pass(input, local_activations, local_weighted_inputs);
                    backward_pass(local_activations, local_weighted_inputs, local_deriv_biases, 
                                local_deriv_weights, input, true_output);

                    for (size_t layer = 0; layer < NUM_LAYERS - 1; ++layer) {
                        for (size_t i = 0; i < biases[layer].size(); ++i)
                            thread_deriv_biases[layer][i] += local_deriv_biases[layer][i];
                        for (size_t i = 0; i < weights[layer].size(); ++i)
                            thread_deriv_weights[layer][i] += local_deriv_weights[layer][i];
                    }
                }

                #pragma omp critical
                {
                    for (size_t layer = 0; layer < NUM_LAYERS - 1; ++layer) {
                        for (size_t i = 0; i < deriv_biases[layer].size(); ++i)
                            deriv_biases[layer][i] += thread_deriv_biases[layer][i];
                        for (size_t i = 0; i < deriv_weights[layer].size(); ++i)
                            deriv_weights[layer][i] += thread_deriv_weights[layer][i];
                    }
                }
            }

            T inv_sample_size = T{1} / static_cast<T>(current_batch_size);

            for (size_t layer = 0; layer < NUM_LAYERS - 1; ++layer) 
            {
                gradient_descent_step(biases[layer], deriv_biases[layer], inv_sample_size, step_size);
                gradient_descent_step(weights[layer], deriv_weights[layer], inv_sample_size, step_size);
            }
        }

        train_error = assess(dataset_orig);
        std::cout << "current epoch: " << epoch << " training error: " << train_error << std::endl;
    }

    return train_error;
}

template<typename Activation, size_t NUM_LAYERS, typename T>
requires callable_with<Activation, T, T> && derivative_callable_with<Activation, T, T> && (NUM_LAYERS >= 3)
T network<Activation, NUM_LAYERS, T>::assess(
    const std::vector<std::pair<std::vector<T>, std::vector<T>>> & dataset) const
{
    size_t dataset_size = dataset.size();
    T test_error = static_cast<T>(0);

    #pragma omp parallel for reduction(+:test_error)
    for (size_t i = 0; i < dataset.size(); ++i)
    {
        const auto & [sample, label] = dataset[i];
        auto model_out = evaluate(sample);
        test_error += compute_loss(model_out, label);
    }

    return test_error / static_cast<T>(dataset_size);
}

template<typename Activation, size_t NUM_LAYERS, typename T>
requires callable_with<Activation, T, T> && derivative_callable_with<Activation, T, T> && (NUM_LAYERS >= 3)
std::vector<T> network<Activation, NUM_LAYERS, T>::evaluate(const std::vector<T> & input) const
{
    if (input.size() != neurons_per_layer[0])
    {
        throw std::invalid_argument{"Input does not have the same size as input layer."};
    }

    std::vector<T> tmp_vec_1{input};
    std::vector<T> tmp_vec_2(max_neurons);
    tmp_vec_1.resize(max_neurons);

    std::vector<T> & tmp_input = tmp_vec_1;
    std::vector<T> & tmp_output = tmp_vec_2;

    for(size_t layer = 0; layer < NUM_LAYERS - 2; ++layer)
    {
        size_t input_size = neurons_per_layer[layer];
        size_t output_size = neurons_per_layer[layer + 1];

        for(size_t i = 0; i < output_size; ++i)
        {
            T sum = biases[layer][i];
            sum += std::transform_reduce(std::execution::par_unseq,
                        weights[layer].begin() + i * input_size,
                        weights[layer].begin() + (i + 1) * input_size,
                        tmp_input.begin(), T(0));
            tmp_output[i] = activation_func(sum);
        }

        std::swap(tmp_input, tmp_output);
    }

    size_t input_size = neurons_per_layer[NUM_LAYERS - 2];
    size_t output_size = neurons_per_layer[NUM_LAYERS - 1];

    for(size_t i = 0; i < output_size; ++i)
    {
        T sum = biases[NUM_LAYERS - 2][i];
        sum += std::transform_reduce(std::execution::par_unseq,
                        weights[NUM_LAYERS - 2].begin() + i * input_size,
                        weights[NUM_LAYERS - 2].begin() + (i + 1) * input_size,
                        tmp_input.begin(), T(0));
        tmp_output[i] = sum;
    }

    tmp_output.resize(output_size);
    softmax_inplace(tmp_output);

    return tmp_output;
}

template<typename Activation, size_t NUM_LAYERS, typename T>
requires callable_with<Activation, T, T> && derivative_callable_with<Activation, T, T> && (NUM_LAYERS >= 3)
void network<Activation, NUM_LAYERS, T>::forward_pass(
    const std::vector<T> & input, 
    std::array<std::vector<T>, NUM_LAYERS - 1> & activations, 
    std::array<std::vector<T>, NUM_LAYERS - 1> & weighted_inputs) const
{
    const std::vector<T>* tmp_input = &input;

    for(size_t layer = 0; layer < NUM_LAYERS - 2; ++layer)
    {
        size_t input_size = neurons_per_layer[layer];
        size_t output_size = neurons_per_layer[layer + 1];

        for(size_t i = 0; i < output_size; ++i)
        {
            T sum = biases[layer][i];
            sum += std::transform_reduce(std::execution::par_unseq,
                        weights[layer].begin() + i * input_size,
                        weights[layer].begin() + (i + 1) * input_size,
                        (*tmp_input).begin(), T(0));

            weighted_inputs[layer][i] = sum;
        }

        for(size_t i = 0; i < output_size; ++i)
        {
            activations[layer][i] = activation_func(weighted_inputs[layer][i]);
        }

        tmp_input = &activations[layer];
    }

    size_t input_size = neurons_per_layer[NUM_LAYERS - 2];
    size_t output_size = neurons_per_layer[NUM_LAYERS - 1];

    for(size_t i = 0; i < output_size; ++i)
    {
        T sum = biases[NUM_LAYERS - 2][i];
        sum += std::transform_reduce(std::execution::par_unseq,
                    weights[NUM_LAYERS - 2].begin() + i * input_size,
                    weights[NUM_LAYERS - 2].begin() + (i + 1) * input_size,
                    (*tmp_input).begin(), T(0));

        weighted_inputs[NUM_LAYERS - 2][i] = sum;
    }

    activations[NUM_LAYERS - 2] = softmax(weighted_inputs[NUM_LAYERS - 2]);
}

template<typename Activation, size_t NUM_LAYERS, typename T>
requires callable_with<Activation, T, T> && derivative_callable_with<Activation, T, T> && (NUM_LAYERS >= 3)
void network<Activation, NUM_LAYERS, T>::backward_pass(
    const std::array<std::vector<T>, NUM_LAYERS - 1> & activations, 
    const std::array<std::vector<T>, NUM_LAYERS - 1> & weighted_inputs,
    std::array<std::vector<T>, NUM_LAYERS - 1> & deriv_biases, 
    std::array<std::vector<T>, NUM_LAYERS - 1> & deriv_weights,
    const std::vector<T> & input, const std::vector<T> & true_output) const
{
    size_t input_size = neurons_per_layer[NUM_LAYERS - 2];
    size_t output_size = neurons_per_layer[NUM_LAYERS - 1];

    deriv_biases[NUM_LAYERS - 2] = activations[NUM_LAYERS - 2];

    for(size_t i = 0; i < output_size; ++i){

        deriv_biases[NUM_LAYERS - 2][i] -= true_output[i];

        for(size_t j = 0; j < input_size; ++j){
            deriv_weights[NUM_LAYERS - 2][i * input_size + j] = 
            deriv_biases[NUM_LAYERS - 2][i] * activations[NUM_LAYERS - 3][j];
        }
    }

    for(size_t layer = NUM_LAYERS - 3; layer > 0; --layer){

        input_size = neurons_per_layer[layer];
        output_size = neurons_per_layer[layer + 1];

        for(size_t i = 0; i < output_size; ++i){

            T tmp_value{};

            for(size_t k = 0; k < neurons_per_layer[layer + 2]; ++k){
                tmp_value += weights[layer + 1][k * output_size + i] * deriv_biases[layer + 1][k];
            }

            deriv_biases[layer][i] = tmp_value * activation_func.derivative(weighted_inputs[layer][i]);

            for(size_t j = 0; j < input_size; ++j){
                deriv_weights[layer][i * input_size + j] = 
                deriv_biases[layer][i] * activations[layer - 1][j];
            }
        }

    }

    input_size = neurons_per_layer[0];
    output_size = neurons_per_layer[1];

    for(size_t i = 0; i < output_size; ++i){

        T tmp_value{};

        for(size_t k = 0; k < neurons_per_layer[2]; ++k){
            tmp_value += weights[1][k * output_size + i] * deriv_biases[1][k];
        }

        deriv_biases[0][i] = tmp_value * activation_func.derivative(weighted_inputs[0][i]);

        for(size_t j = 0; j < input_size; ++j){
            deriv_weights[0][i * input_size + j] = deriv_biases[0][i] * input[j];
        }
    }
}

template<typename Activation, size_t NUM_LAYERS, typename T>
requires callable_with<Activation, T, T> && derivative_callable_with<Activation, T, T> && (NUM_LAYERS >= 3)
T network<Activation, NUM_LAYERS, T>::compute_loss(
    const std::vector<T> & input, const std::vector<T> & output) const
{
    return cross_entropy_loss(input, output);
}

template<typename Activation, size_t NUM_LAYERS, typename T>
requires callable_with<Activation, T, T> && derivative_callable_with<Activation, T, T> && (NUM_LAYERS >= 3)
void network<Activation, NUM_LAYERS, T>::store(const std::string & filename) const
{
    std::ofstream output(filename, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot open file " + filename);

    const char magic[4] = {'N', 'N', 'E', 'T'};
    uint8_t type_flag = std::is_same_v<T, float> ? 0 : 1;

    output.write(magic, 4);
    output.write(reinterpret_cast<const char*>(&type_flag), sizeof(type_flag));

    size_t num_layers = NUM_LAYERS;
    output.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    output.write(reinterpret_cast<const char*>(neurons_per_layer.data()), NUM_LAYERS * sizeof(size_t));

    for(size_t layer = 0; layer < NUM_LAYERS - 1; ++layer)
    {
        output.write(reinterpret_cast<const char*>(weights[layer].data()), weights[layer].size() * sizeof(T));
        output.write(reinterpret_cast<const char*>(biases[layer].data()), biases[layer].size() * sizeof(T));
    }
}

template<typename Activation, size_t NUM_LAYERS, typename T>
requires callable_with<Activation, T, T> && derivative_callable_with<Activation, T, T> && (NUM_LAYERS >= 3)
void network<Activation, NUM_LAYERS, T>::load(const std::string& filename)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open file " + filename);

    char magic[4];
    input.read(magic, 4);
    if (memcmp(magic, "NNET", 4) != 0)
        throw std::runtime_error("Invalid file format: missing magic number");

    uint8_t file_type_flag;
    input.read(reinterpret_cast<char*>(&file_type_flag), sizeof(file_type_flag));

    constexpr uint8_t this_type_flag = std::is_same_v<T, float> ? 0 : 1;
    if (file_type_flag != this_type_flag)
        throw std::runtime_error("Type mismatch: file stores a different floating point type.");

    size_t num_layers;
    input.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    if (num_layers != NUM_LAYERS)
        throw std::runtime_error("Network layer count mismatch");

    input.read(reinterpret_cast<char*>(neurons_per_layer.data()), NUM_LAYERS * sizeof(size_t));

    num_input_neurons = neurons_per_layer.front();
    num_output_neurons = neurons_per_layer.back();
    max_neurons = *std::max_element(neurons_per_layer.begin(), neurons_per_layer.end());

    for (size_t layer = 0; layer < NUM_LAYERS - 1; ++layer)
    {
        size_t weight_count = neurons_per_layer[layer + 1] * neurons_per_layer[layer];
        size_t bias_count = neurons_per_layer[layer + 1];

        weights[layer].resize(weight_count);
        biases[layer].resize(bias_count);

        input.read(reinterpret_cast<char*>(weights[layer].data()), weight_count * sizeof(T));
        input.read(reinterpret_cast<char*>(biases[layer].data()), bias_count * sizeof(T));

    }
}
}

#endif // NETWORK_H