#include<iostream>

#include <activation.hpp>
#include <custom_concepts.hpp>
#include <network.hpp>

int main (int argc, char ** argv)
{
    auto f = [](float a){return 1;};

    NN::network<NN::ReLU<float>, 5> m(NN::ReLU<float>(), {2,10,10,10,3}, true);

    m.learn(std::vector<float>{{2,2}},std::vector<float>{{0,1,0}}, 1);
    auto test = m.evaluate(std::vector<float>{{100,100}});

    for(auto & i : test) std::cout << i << std::endl;
}