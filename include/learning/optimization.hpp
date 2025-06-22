#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <algorithm>
#include <cmath>
#include <vector>

#include <Accelerate/Accelerate.h>

#include "../utility/debug.hpp"

namespace NN::Optimizer
{
template<typename T>
class Gradient_Descent{
private:
    T learning_rate = static_cast<T>(0.01);
    Debug_Mode debug_mode = Debug_Mode::Release;

public:
    Gradient_Descent() = default;

    Gradient_Descent(T learning_rate, Debug_Mode debug_mode = Debug_Mode::Release) 
    : learning_rate{learning_rate}, debug_mode{debug_mode} {};

    void update(std::vector<T> & current, const std::vector<T> & gradient)
    {
        if(debug_mode == Debug_Mode::Debug && current.size() != gradient.size())
        {
            throw std::invalid_argument("Gradient and parameters do not have same size.");
        }    

        T size = gradient.size();
        T tmp_value = - learning_rate;

        std::vector<T> tmp_vector(size, 0);

        if constexpr (std::is_same_v<T, double>) {
            vDSP_vsmulD(gradient.data(), 1, &tmp_value, tmp_vector.data(), 1, size);
            vDSP_vaddD(current.data(), 1, tmp_vector.data(), 1, current.data(), 1, size);
        }
        else if constexpr (std::is_same_v<T, float>) {
            vDSP_vsmul(gradient.data(), 1, &tmp_value, tmp_vector.data(), 1, size);
            vDSP_vadd(current.data(), 1, tmp_vector.data(), 1, current.data(), 1, size);
        }
        else {
            static_assert(std::is_same_v<T, void>, "Acceleration only for float or double type");
        }
        
    }
};

template<typename T>
class Adam_Optimizer {
private:
    T learning_rate = static_cast<T>(0.001);
    T beta1 = static_cast<T>(0.9);
    T beta2 = static_cast<T>(0.999);
    T epsilon = static_cast<T>(1e-8);
    int timestep = 0;
    Debug_Mode debug_mode = Debug_Mode::Release;

    std::vector<T> m;
    std::vector<T> v;

public:
    Adam_Optimizer() = default;

    Adam_Optimizer(T learning_rate, 
                T beta1 = static_cast<T>(0.9),   
                T beta2 = static_cast<T>(0.999),
                T epsilon = static_cast<T>(1e-8),
                Debug_Mode debug_mode = Debug_Mode::Release) 
    : learning_rate{learning_rate}, beta1{beta1}, beta2{beta2}, 
    epsilon{epsilon }, debug_mode{debug_mode} {}


    void update(std::vector<T>& current, const std::vector<T>& gradient) 
    {
        if (debug_mode == Debug_Mode::Debug && current.size() != gradient.size()) {
            throw std::invalid_argument("Gradient and parameters do not have same size.");
        }

        size_t size = current.size();
        int size_int = static_cast<int>(size);

        if (m.empty()) {
            m.resize(size, static_cast<T>(0));
            v.resize(size, static_cast<T>(0));
        }

        ++timestep;

        T inv_beta1_pow_t = - learning_rate / (static_cast<T>(1) - std::pow(beta1, timestep));
        T inv_beta2_pow_t = static_cast<T>(1) / (static_cast<T>(1) - std::pow(beta2, timestep));
        T one_minus_beta_1 = (static_cast<T>(1) - beta1);
        T one_minus_beta_2 = (static_cast<T>(1) - beta2);

        std::vector<T> m_hat(size, static_cast<T>(0));
        std::vector<T> v_hat(size, static_cast<T>(0));
        std::vector<T> grad = gradient;
        std::vector<T> grad_sq(size, static_cast<T>(0));

        if constexpr (std::is_same_v<T, double>) {

            vDSP_vsmulD(m.data(), 1, &beta1, m.data(), 1, size);
            vDSP_vsmulD(v.data(), 1, &beta2, v.data(), 1, size);

            vDSP_vmulD(grad.data(), 1, grad.data(), 1, grad_sq.data(), 1, size);

            vDSP_vsmulD(grad.data(), 1, &one_minus_beta_1, grad.data(), 1, size);
            vDSP_vsmulD(grad_sq.data(), 1, &one_minus_beta_2, grad_sq.data(), 1, size);

            vDSP_vaddD(grad.data(), 1, m.data(), 1, m.data(), 1, size);
            vDSP_vaddD(grad_sq.data(), 1, v.data(), 1, v.data(), 1, size);

            vDSP_vsmulD(m.data(), 1, &inv_beta1_pow_t, m_hat.data(), 1, size);
            vDSP_vsmulD(v.data(), 1, &inv_beta2_pow_t, v_hat.data(), 1, size);

            vvsqrt(v_hat.data(), v_hat.data(), &size_int);
            vDSP_vsaddD(v_hat.data(), 1, &epsilon, v_hat.data(), 1, size);
            vDSP_vdivD(v_hat.data(), 1, m_hat.data(), 1, m_hat.data(), 1, size);
            vDSP_vaddD(m_hat.data(), 1, current.data(), 1, current.data(), 1, size);
        }
        else if constexpr (std::is_same_v<T, float>) {

            vDSP_vsmul(m.data(), 1, &beta1, m.data(), 1, size);
            vDSP_vsmul(v.data(), 1, &beta2, v.data(), 1, size);

            vDSP_vmul(grad.data(), 1, grad.data(), 1, grad_sq.data(), 1, size);

            vDSP_vsmul(grad.data(), 1, &one_minus_beta_1, grad.data(), 1, size);
            vDSP_vsmul(grad_sq.data(), 1, &one_minus_beta_2, grad_sq.data(), 1, size);

            vDSP_vadd(grad.data(), 1, m.data(), 1, m.data(), 1, size);
            vDSP_vadd(grad_sq.data(), 1, v.data(), 1, v.data(), 1, size);

            vDSP_vsmul(m.data(), 1, &inv_beta1_pow_t, m_hat.data(), 1, size);
            vDSP_vsmul(v.data(), 1, &inv_beta2_pow_t, v_hat.data(), 1, size);

            vvsqrtf(v_hat.data(), v_hat.data(), &size_int);
            vDSP_vsadd(v_hat.data(), 1, &epsilon, v_hat.data(), 1, size);
            vDSP_vdiv(v_hat.data(), 1, m_hat.data(), 1, m_hat.data(), 1, size);
            vDSP_vadd(m_hat.data(), 1, current.data(), 1, current.data(), 1, size);
        }
        else {
            static_assert(std::is_same_v<T, void>, "Acceleration only for float or double type");
        }
    }

    void reset() {
        std::fill(m.begin(), m.end(), static_cast<T>(0));
        std::fill(v.begin(), v.end(), static_cast<T>(0));
        timestep = 0;
    }
};
}

#endif // OPTIMIZATION_H