#ifndef READ_MNIST_H
#define READ_MNIST_H

#include <fstream>
#include <vector>
#include <stdexcept>
#include <iostream>

template<typename T>
std::vector<std::pair<std::vector<T>, std::vector<T>>> prepare_dataset(
    const std::vector<std::vector<T>> &images, 
    const std::vector<unsigned char> &labels)
{
    if (images.size() != labels.size())
        throw std::runtime_error("Images and labels count mismatch");

    size_t num_samples = images.size();
    size_t output_size = 10; // digits 0-9

    std::vector<std::pair<std::vector<T>, std::vector<T>>> dataset;
    dataset.reserve(num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
        std::vector<T> output(output_size, 0.0f);
        output[labels[i]] = 1.0f;  // One-hot

        dataset.emplace_back(images[i], output);
    }

    return dataset;
}

template<typename T>
std::vector<std::vector<T>> load_mnist_images(const std::string &filename, size_t &num_images, size_t &rows, size_t &cols) {
    
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file " + filename);

    // Header auslesen (big endian)
    uint32_t magic = 0, n_images = 0, n_rows = 0, n_cols = 0;

    auto read_uint32 = [&](uint32_t &val) {
        unsigned char bytes[4];
        file.read(reinterpret_cast<char*>(bytes), 4);
        val = (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]);
    };

    read_uint32(magic);
    if (magic != 2051) // magic number für images
        throw std::runtime_error("Invalid MNIST image file!");

    read_uint32(n_images);
    read_uint32(n_rows);
    read_uint32(n_cols);

    num_images = n_images;
    rows = n_rows;
    cols = n_cols;

    std::vector<std::vector<T>> images(n_images, std::vector<T>(n_rows * n_cols));

    for (size_t i = 0; i < n_images; ++i) {
        std::vector<unsigned char> buffer(n_rows * n_cols);
        file.read(reinterpret_cast<char*>(buffer.data()), n_rows * n_cols);
        for (size_t j = 0; j < n_rows * n_cols; ++j) {
            images[i][j] = static_cast<T>(buffer[j]) / static_cast<T>(255);
        }
    }

    return images;
}

template<typename T>
std::vector<unsigned char> load_mnist_labels(const std::string &filename, size_t &num_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file " + filename);

    uint32_t magic = 0, n_labels = 0;

    auto read_uint32 = [&](uint32_t &val) {
        unsigned char bytes[4];
        file.read(reinterpret_cast<char*>(bytes), 4);
        val = (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]);
    };

    read_uint32(magic);
    if (magic != 2049) // magic number für labels
        throw std::runtime_error("Invalid MNIST label file!");

    read_uint32(n_labels);
    num_labels = n_labels;

    std::vector<unsigned char> labels(n_labels);
    file.read(reinterpret_cast<char*>(labels.data()), n_labels);

    return labels;
}


#endif // READ_MNIST_H