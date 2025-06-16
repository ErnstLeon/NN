#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <cmath>
#include <execution>
#include <vector>

namespace NN
{
template<typename T>
inline void gradient_descent_step(std::vector<T> & value, 
    const std::vector<T> & gradient, T learning_rate = static_cast<T>(0.01), T scaling = static_cast<T>(1))
{
    if(value.size() != gradient.size())
    {
        throw std::invalid_argument("Gradient and parameters do not have same size.");
    }    
    
    std::transform(std::execution::par_unseq, value.begin(), value.end(),
                gradient.begin(), value.begin(),
                [&](T val, T grad) { return val - learning_rate * grad * scaling;});
}
}


#endif // OPTIMIZATION_H