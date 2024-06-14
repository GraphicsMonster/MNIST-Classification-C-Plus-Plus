#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <vector>
#include <stdexcept>
#include <string>

std::vector<std::vector<double>> RELU(const std::vector<std::vector<double>>& a);
std::vector<std::vector<double>> Sigmoid(const std::vector<std::vector<double>>& a);
std::vector<std::vector<double>> Tanh(const std::vector<std::vector<double>>& a);
std::vector<std::vector<double>> Softmax(const std::vector<std::vector<double>>& a);
std::vector<std::vector<double>> LeakyRELU(const std::vector<std::vector<double>>& a, double alpha);
std::vector<std::vector<double>> applyActivation(const std::vector<std::vector<double>>& data, std::string activation);

std::vector<std::vector<double>> RELU_derivative(const std::vector<std::vector<double>>& a);
std::vector<std::vector<double>> Sigmoid_derivative(const std::vector<std::vector<double>>& a);
std::vector<std::vector<double>> Tanh_derivative(const std::vector<std::vector<double>>& a);
std::vector<std::vector<double>> Softmax_derivative(const std::vector<std::vector<double>>& a);
std::vector<std::vector<double>> LeakyRELU_derivative(const std::vector<std::vector<double>>& a, double alpha);
std::vector<std::vector<double>> applyActivation_derivative(const std::vector<std::vector<double>>& data, std::string activation);

#endif
