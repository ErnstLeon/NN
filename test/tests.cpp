#include <array>
#include <gtest/gtest.h>
#include <vector>

#include "neural_network.hpp"

TEST(CrossEntropyLossTest, BasicTest) {
    std::vector<double> output = {0.7, 0.2, 0.1}; 
    std::vector<double> target = {1.0, 0.0, 0.0};

    double expected = - std::log(0.7 + 1e-12);
    double actual = NN::cross_entropy_loss(output, target);

    EXPECT_NEAR(actual, expected, 1e-6);
}

TEST(ReLUActivation, ForwardAndDerivative) {
    NN::ReLU<double> relu;

    EXPECT_EQ(relu(3.14), 3.14);
    EXPECT_EQ(relu(-2.0), 0.0);
    EXPECT_EQ(relu(0.0), 0.0);

    EXPECT_EQ(relu.derivative(5.0), 1.0);
    EXPECT_EQ(relu.derivative(-1.0), 0.0);
    EXPECT_EQ(relu.derivative(0.0), 0.0);
}

TEST(SigmoidActivation, ForwardAndDerivative) {
    NN::Sigmoid<double> sigmoid;

    EXPECT_NEAR(sigmoid(0.0), 0.5, 1e-6);
    EXPECT_NEAR(sigmoid(15.0), 1.0, 1e-4);
    EXPECT_NEAR(sigmoid(-15.0), 0.0, 1e-4);

    double s = sigmoid(0.0);
    EXPECT_NEAR(sigmoid.derivative(0.0), s * (1 - s), 1e-6);
}

TEST(SoftmaxFunction, BasicTest) {
    std::vector<double> input = {1.0, 2.0, 3.0};
    std::vector<double> output = NN::softmax(input);

    double sum = 0.0;
    for (double val : output) {
        EXPECT_GE(val, 0.0);
        EXPECT_LE(val, 1.0);
        sum += val;
    }
    EXPECT_NEAR(sum, 1.0, 1e-6);
    EXPECT_GT(output[2], output[1]);
    EXPECT_GT(output[1], output[0]);
}

TEST(SoftmaxInplaceFunction, BasicTest) {
    std::vector<double> input = {1.0, 2.0, 3.0};
    NN::softmax_inplace(input);

    double sum = 0.0;
    for (double val : input) {
        EXPECT_GE(val, 0.0);
        EXPECT_LE(val, 1.0);
        sum += val;
    }
    EXPECT_NEAR(sum, 1.0, 1e-6);
    EXPECT_GT(input[2], input[1]);
    EXPECT_GT(input[1], input[0]);
}

TEST(NetworkTest, EvaluateOutputShape) {

    NN::network<NN::Sigmoid<double>, 3> network({2, 3, 2}, true);

    std::vector<double> input = {1.0, 0.0};
    auto output = network.evaluate(input);

    EXPECT_EQ(output.size(), 2);
    for (double val : output) {
        EXPECT_GE(val, 0.0);
        EXPECT_LE(val, 1.0);
    }
}

TEST(GDOptimizerTest, ConvergesToMinimum) {
    
    NN::Optimizer::Gradient_Descent<double> optimizer{0.1};

    std::vector<double> value(1);
    std::vector<double> gradient(1);

    // function (x - 3)^2, derivate: 2(x - 3)
    value[0] = static_cast<double>(0.0);
    for (int step = 0; step < 1000; ++step) {
        gradient[0] = 2 * (value[0] - 3);
        optimizer.update(value, gradient);
    }

    EXPECT_NEAR(value[0], 3.0, 1e-4);
}

TEST(AdamOptimizerTest, ConvergesToMinimum) {
    
    NN::Optimizer::Adam_Optimizer<double> optimizer{0.1};

    std::vector<double> value(1);
    std::vector<double> gradient(1);

    // function (x - 3)^2, derivate: 2(x - 3)
    value[0] = static_cast<double>(0.0);
    for (int step = 0; step < 1000; ++step) {
        gradient[0] = 2 * (value[0] - 3);
        optimizer.update(value, gradient);
    }

    EXPECT_NEAR(value[0], 3.0, 1e-4);
}

TEST(NetworkTest, LearnXOR) {

    NN::Optimizer::Gradient_Descent<double> optimizer{0.5};
    NN::network<NN::Sigmoid<double>, 3> network({2, 10, 2}, true);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset = {
        {{0, 0}, {1, 0}},
        {{0, 1}, {0, 1}},
        {{1, 0}, {0, 1}},
        {{1, 1}, {1, 0}},
    };

    network.learn(dataset, optimizer, 2, 500);
    auto out_00 = network.evaluate({0, 0});
    auto out_01 = network.evaluate({0, 1});
    auto out_10 = network.evaluate({1, 0});
    auto out_11 = network.evaluate({1, 1});

    EXPECT_GT(out_01[1], 0.75);
    EXPECT_GT(out_10[1], 0.75);
    EXPECT_GT(out_00[0], 0.75);
    EXPECT_GT(out_11[0], 0.75);
}