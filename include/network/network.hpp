#ifndef NETWORK_H
#define NETWORK_H

#include <algorithm>
#include <array>
#include <execution>
#include <random>
#include <vector>

#include "utility/custom_concepts.hpp"
#include "learning/loss.hpp"

namespace NN{
template<typename Activation, size_t NUM_LAYERS, typename T = typename Activation::type>
requires callable_with<Activation, T, T> && derivative_callable_with<Activation, T, T> && (NUM_LAYERS >= 3)
class network {
private:

    const Activation activation_func{};
    
    std::array<size_t, NUM_LAYERS> neurons_per_layer{};

    size_t num_input_neurons{};
    size_t num_output_neurons{};

    std::array<std::vector<T>, NUM_LAYERS - 1> biases{};
    std::array<std::vector<T>, NUM_LAYERS - 1> weights{};

    void forward_pass(const std::vector<T> &, std::array<std::vector<T>, NUM_LAYERS - 1> &, 
    std::array<std::vector<T>, NUM_LAYERS - 1> &) const;

    void backward_pass(const std::array<std::vector<T>, NUM_LAYERS - 1> &, 
    const std::array<std::vector<T>, NUM_LAYERS - 1> &, std::array<std::vector<T>, NUM_LAYERS - 1> &, 
    std::array<std::vector<T>, NUM_LAYERS - 1> &, const std::vector<T> &, const std::vector<T> &) const;

    T compute_loss(const std::vector<T> &, const std::vector<T> &);

public: 

    network(std::initializer_list<size_t> neuron_list, bool rnd = false)
    : activation_func{activation_func}
    {
        if(neuron_list.size() !=  NUM_LAYERS)
        {
            throw std::invalid_argument("The number of layers does not match with the provided.");
        }

        std::copy(neuron_list.begin(), neuron_list.end(), neurons_per_layer.begin());

        num_input_neurons = neurons_per_layer.front();
        num_output_neurons = neurons_per_layer.back();

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
    T assess(const std::vector<std::pair<std::vector<T>, std::vector<T>>> &);
    
    std::vector<T> evaluate(const std::vector<T> & input);
};

template<typename Activation, size_t NUM_LAYERS, typename T>
requires callable_with<Activation, T, T> && derivative_callable_with<Activation, T, T> && (NUM_LAYERS >= 3)
T network<Activation, NUM_LAYERS, T>::learn(
    const std::vector<std::pair<std::vector<T>, std::vector<T>>> & dataset_orig, 
    size_t batch_size, size_t num_epochs, T step_size)
{
    auto dataset = dataset_orig;

    std::mt19937 gen(100);
    size_t dataset_size = dataset.size();
    size_t num_batches = (dataset_size + batch_size - 1) / batch_size;

    T train_error;

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) 
    {
        train_error = static_cast<T>(0);

        std::shuffle(dataset.begin(), dataset.end(), gen);

        for (size_t b = 0; b < num_batches; ++b) 
        {
            size_t current_batch_size = std::min(batch_size, dataset_size - b * batch_size);

            std::array<std::vector<T>, NUM_LAYERS - 1> deriv_biases{};
            std::array<std::vector<T>, NUM_LAYERS - 1> deriv_weights{};

            for (size_t i = 0; i < NUM_LAYERS - 1; ++i) {
                deriv_biases[i].assign(biases[i].size(), static_cast<T>(0));
                deriv_weights[i].assign(weights[i].size(), static_cast<T>(0));
            }

            #pragma omp parallel shared(deriv_biases, deriv_weights) reduction(+ : train_error)
            {
                std::array<std::vector<T>, NUM_LAYERS - 1> thread_deriv_biases;
                std::array<std::vector<T>, NUM_LAYERS - 1> thread_deriv_weights;

                for (size_t i = 0; i < NUM_LAYERS - 1; ++i) {
                    thread_deriv_biases[i].assign(biases[i].size(), static_cast<T>(0));
                    thread_deriv_weights[i].assign(weights[i].size(), static_cast<T>(0));
                }

                #pragma omp for nowait
                for (size_t sample = 0; sample < current_batch_size; ++sample) 
                {
                    size_t sample_index = b * batch_size + sample;

                    const auto & input = dataset[sample_index].first;
                    const auto & true_output = dataset[sample_index].second;

                    if (input.size() != neurons_per_layer[0])
                        throw std::invalid_argument{"Input does not match input layer size."};
                    if (true_output.size() != neurons_per_layer.back())
                        throw std::invalid_argument{"Output does not match output layer size."};

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

                    forward_pass(input, local_activations, local_weighted_inputs);
                    backward_pass(local_activations, local_weighted_inputs,
                                local_deriv_biases, local_deriv_weights,
                                input, true_output);

                    for (size_t layer = 0; layer < NUM_LAYERS - 1; ++layer) {
                        for (size_t i = 0; i < biases[layer].size(); ++i)
                            thread_deriv_biases[layer][i] += local_deriv_biases[layer][i];
                        for (size_t i = 0; i < weights[layer].size(); ++i)
                            thread_deriv_weights[layer][i] += local_deriv_weights[layer][i];
                    }

                    train_error += compute_loss(local_activations.back(), true_output);
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
                for (size_t i = 0; i < biases[layer].size(); ++i){
                    biases[layer][i] -= step_size * inv_sample_size * deriv_biases[layer][i];
                }

                for (size_t i = 0; i < weights[layer].size(); ++i){
                    weights[layer][i] -= step_size * inv_sample_size * deriv_weights[layer][i];
                }
            }
        }

        train_error /= static_cast<T>(dataset_size);
        std::cout << "current epoch: " << epoch << " training error: " << train_error << std::endl;
    }

    return train_error;
}

template<typename Activation, size_t NUM_LAYERS, typename T>
requires callable_with<Activation, T, T> && derivative_callable_with<Activation, T, T> && (NUM_LAYERS >= 3)
T network<Activation, NUM_LAYERS, T>::assess(
    const std::vector<std::pair<std::vector<T>, std::vector<T>>> & dataset)
{
    size_t dataset_size = dataset.size();
    T test_error = static_cast<T>(0);

    for (const auto & [sample, label] : dataset) 
    {
        auto model_out = evaluate(sample);
        test_error += compute_loss(model_out, label);
    }

    return test_error / static_cast<T>(dataset_size);
}

template<typename Activation, size_t NUM_LAYERS, typename T>
requires callable_with<Activation, T, T> && derivative_callable_with<Activation, T, T> && (NUM_LAYERS >= 3)
std::vector<T> network<Activation, NUM_LAYERS, T>::evaluate(const std::vector<T> & input)
{
    if (input.size() != neurons_per_layer[0])
    {
        throw std::invalid_argument{"Input does not have the same size as input layer."};
    }

    std::vector<T> tmp_vec = input;

    for(size_t layer = 0; layer < NUM_LAYERS - 2; ++layer)
    {
        size_t input_size = neurons_per_layer[layer];
        size_t output_size = neurons_per_layer[layer + 1];

        std::vector<T> tmp_out(output_size);

        for(size_t i = 0; i < output_size; ++i)
        {
            T sum = biases[layer][i];
            for(size_t j = 0; j < input_size; ++j)
            {
                sum += weights[layer][i * input_size + j] * tmp_vec[j];
            }
            tmp_out[i] = activation_func(sum);
        }

        tmp_vec = std::move(tmp_out);
    }

    size_t input_size = neurons_per_layer[NUM_LAYERS - 2];
    size_t output_size = neurons_per_layer[NUM_LAYERS - 1];
    
    std::vector<T> output(output_size);

    for(size_t i = 0; i < output_size; ++i)
    {
        T sum = biases[NUM_LAYERS - 2][i];
        for(size_t j = 0; j < input_size; ++j)
        {
            sum += weights[NUM_LAYERS - 2][i * input_size + j] * tmp_vec[j];
        }
        output[i] = sum;
    }

    softmax(output);

    return output;
}

template<typename Activation, size_t NUM_LAYERS, typename T>
requires callable_with<Activation, T, T> && derivative_callable_with<Activation, T, T> && (NUM_LAYERS >= 3)
void network<Activation, NUM_LAYERS, T>::forward_pass(
    const std::vector<T> & input, 
    std::array<std::vector<T>, NUM_LAYERS - 1> & activations, 
    std::array<std::vector<T>, NUM_LAYERS - 1> & weighted_inputs) const
{
    std::vector<T> tmp_input{input};
    std::vector<T> tmp_output{biases[0]};

    for(size_t layer = 0; layer < NUM_LAYERS - 2; ++layer)
    {
        size_t input_size = tmp_input.size();
        size_t output_size = tmp_output.size();

        for(size_t i = 0; i < output_size; ++i)
        {
            T sum = static_cast<T>(0);
            for(size_t j = 0; j < input_size; ++j)
            {
                sum += weights[layer][i * input_size + j] * tmp_input[j];
            } 
            tmp_output[i] += sum;
        }

        weighted_inputs[layer] = tmp_output;

        for(size_t i = 0; i < output_size; ++i)
        {
            tmp_output[i] = activation_func(tmp_output[i]);
        }

        activations[layer] = tmp_output;

        tmp_input = std::move(tmp_output);
        tmp_output = biases[layer + 1];

    }

    size_t input_size = tmp_input.size();
    size_t output_size = tmp_output.size();

    for(size_t i = 0; i < output_size; ++i)
    {
        T sum = static_cast<T>(0);
        for(size_t j = 0; j < input_size; ++j)
        {
            sum += weights[NUM_LAYERS - 2][i * input_size + j] * tmp_input[j];
        } 
        tmp_output[i] += sum;
    }

    weighted_inputs[NUM_LAYERS - 2] = tmp_output;
    softmax(tmp_output);
    activations[NUM_LAYERS - 2] = std::move(tmp_output);

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
    const std::vector<T> & input, const std::vector<T> & output)
{
    return cross_entropy_loss(input, output);
}
}

#endif // NETWORK_H