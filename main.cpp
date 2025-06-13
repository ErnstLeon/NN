#include<iostream>

#include <activation.hpp>
#include <custom_concepts.hpp>
#include <network.hpp>
#include <read_mnist.hpp>

int main (int argc, char ** argv)
{
    using T = float;

    size_t num_images_train, num_images_test, rows_train, cols_train, rows_test, cols_test;
    auto images_train = load_mnist_images<T>("./data/train-images.idx3-ubyte", 
                                            num_images_train, rows_train, cols_train);
    auto images_test = load_mnist_images<T>("./data/t10k-images.idx3-ubyte",
                                            num_images_test, rows_test, cols_test);

    size_t num_labels_train, num_labels_test;
    auto labels_train = load_mnist_labels<T>("./data/train-labels.idx1-ubyte", num_labels_train);
    auto labels_test = load_mnist_labels<T>("./data/t10k-labels.idx1-ubyte", num_labels_test);

    if (num_images_train != num_labels_train) {
        std::cerr << "Images and labels count mismatch in training data\n";
        return 1;
    }

    if (num_images_test != num_labels_test) {
        std::cerr << "Images and labels count mismatch in test data\n";
        return 1;
    }

    if (rows_train !=  rows_test || cols_train != cols_test) {
        std::cerr << "Mismatch between training and test data\n";
        return 1;
    }

    auto dataset_train = prepare_dataset<T>(images_train, labels_train);
    auto dataset_test = prepare_dataset<T>(images_test, labels_test);

    NN::network<NN::ReLU<float>, 4, float> net(NN::ReLU<float>(), {rows_train * cols_train, 256, 256, 10}, true);

    T train_error = net.learn(dataset_train, 64, 10, 0.01);
    T test_error = net.assess(dataset_test);

    std::cout << std::endl;
    std::cout << "training error: " << train_error << std::endl;
    std::cout << "test error: " << test_error << std::endl;
    std::cout << std::endl;

    auto label = net.evaluate(dataset_test[0].first);

    auto max_iter = std::max_element(label.begin(), label.end());
    size_t value = std::distance(label.begin(), max_iter);

    auto max_iter_ = std::max_element(dataset_test[0].second.begin(), dataset_test[0].second.end());
    size_t value_ = std::distance(dataset_test[0].second.begin(), max_iter_);

    std::cout << "modell output: " << value << std::endl;
    std::cout << "true category: " << value_ << std::endl;

    std::cout << std::endl;
    for(int i = 0; i < rows_test; ++i){
        for(int j = 0; j < cols_test; ++j){
            float val = dataset_test[0].first[i * cols_test + j];
            std::cout << (val > 0.5f ? '#' : '.') << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}