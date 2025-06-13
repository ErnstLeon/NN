#include<iostream>

#include <activation.hpp>
#include <custom_concepts.hpp>
#include <network.hpp>
#include <read_mnist.hpp>

int main (int argc, char ** argv)
{
    using T = float;

    auto f = [](float a){return 1;};

    NN::network<NN::ReLU<T>, 5> m(NN::ReLU<T>(), {2,100,100,100,3}, true);

    size_t sample_size = 3;

    std::vector<std::pair<std::vector<T>, std::vector<T>>> dataset{};
    dataset.reserve(sample_size);

    dataset.emplace_back(std::vector<T>{0,1},std::vector<T>{0,0,1});
    dataset.emplace_back(std::vector<T>{1,1},std::vector<T>{0,1,0});
    dataset.emplace_back(std::vector<T>{1,0},std::vector<T>{1,0,0});

    m.learn(dataset, 3, 1000);

    auto test = m.evaluate(std::vector<float>{{1,0}});
    for(auto & i : test) std::cout << i << std::endl;
    std::cout << std::endl;

    auto test_ = m.evaluate(std::vector<float>{{0,1}});
    for(auto & i : test_) std::cout << i << std::endl;
    std::cout << std::endl;

    auto test__ = m.evaluate(std::vector<float>{{1,1}});
    for(auto & i : test__) std::cout << i << std::endl;
    std::cout << std::endl;

    auto test___ = m.evaluate(std::vector<float>{{100,0}});
    for(auto & i : test___) std::cout << i << std::endl;
    std::cout << std::endl;
}