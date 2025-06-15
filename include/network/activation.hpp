#ifndef ACTIVATION_FUNC_H
#define ACTIVATION_FUNC_H

#include <algorithm>
#include <cmath>
#include <vector>

namespace NN{
template<typename T>
class ReLU
{
public:
    
    using type = T;
    
    T operator()(T x) const
    {
        if(x > 0)
        {
            return x;
        }
        else{
            return static_cast<T>(0);
        }
    }

    T derivative(T x) const
    {
        if(x > 0)
        {
            return 1;
        }
        else{
            return static_cast<T>(0);
        }
    }
};

template<typename T>
inline std::vector<T> softmax(std::vector<T>& input) 
{    
    std::vector<T> output(input.size());

    T max_val = *std::max_element(input.begin(), input.end());

    T sum = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    for (T& val : output) {
        val /= sum;
    }

    return output;
}

template<typename T>
inline void softmax_inplace(std::vector<T>& input) {
    
    T max_val = *std::max_element(input.begin(), input.end());

    T sum = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        T tmp = input[i];
        input[i] = std::exp(tmp - max_val);
        sum += input[i];
    }

    for (T& val : input) {
        val /= sum;
    }
}

}

#endif // ACTIVATION_FUNC_H