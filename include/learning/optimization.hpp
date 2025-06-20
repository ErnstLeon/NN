#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <cmath>
#include <vector>

#include <Accelerate/Accelerate.h>

#include "../utility/debug.hpp"

namespace NN
{
template<typename T>
inline void gradient_descent_step(std::vector<T> & value, 
    const std::vector<T> & gradient, T learning_rate = static_cast<T>(0.01), 
    T scaling = static_cast<T>(1), Debug_Mode debug_mode = Debug_Mode::Release)
{
    if(debug_mode == Debug_Mode::Debug && value.size() != gradient.size())
    {
        throw std::invalid_argument("Gradient and parameters do not have same size.");
    }    
    
    T tmp_value = - learning_rate * scaling;
    T size = gradient.size();

    std::vector<T> tmp_vector(size, 0);

    if constexpr (std::is_same_v<T, double>) {
        vDSP_vsmulD(gradient.data(), 1, &tmp_value, tmp_vector.data(), 1, size);
        vDSP_vaddD(value.data(), 1, tmp_vector.data(), 1, value.data(), 1, size);
    }
    else if constexpr (std::is_same_v<T, float>) {
        vDSP_vsmul(gradient.data(), 1, &tmp_value, tmp_vector.data(), 1, size);
        vDSP_vadd(value.data(), 1, tmp_vector.data(), 1, value.data(), 1, size);
    }
    else {
        static_assert(std::is_same_v<T, void>, "Acceleration only for float or double type");
    }
}
}


#endif // OPTIMIZATION_H