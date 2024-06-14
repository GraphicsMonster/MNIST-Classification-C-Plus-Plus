#include "activations.hpp"
#include <cmath>
#include <vector>

std::vector<std::vector<double>> RELU(const std::vector<std::vector<double>>& a){
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
    for(int i = 0; i < a.size(); i++){
        for(int j = 0; j < a[0].size(); j++){
            result[i][j] = std::max(0.0, a[i][j]);
        }
    }

    return result;
}

std::vector<std::vector<double>> Sigmoid(const std::vector<std::vector<double>>& a){
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
    for(int i = 0; i < a.size(); i++){
        for(int j = 0; j < a[0].size(); j++){
            result[i][j] = 1/(1+exp(-a[i][j]));
        }
    }

    return result;
}

std::vector<std::vector<double>> Tanh(const std::vector<std::vector<double>>& a){
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
    for(int i = 0; i < a.size(); i++){
        for(int j = 0; j < a[0].size(); j++){
            result[i][j] = tanh(a[i][j]);
        }
    }

    return result;
}

std::vector<std::vector<double>> Softmax(const std::vector<std::vector<double>>& a){
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
    for(int i = 0; i < a.size(); i++){
        double sum = 0;
        for(int j = 0; j < a[0].size(); j++){
            result[i][j] = exp(a[i][j]);
            sum += result[i][j];
        }
        for(int j = 0; j < a[0].size(); j++){
            result[i][j] = result[i][j]/sum;
        }
    }

    return result;
}

std::vector<std::vector<double>> LeakyRELU(const std::vector<std::vector<double>>& a, double alpha){
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
    for(int i = 0; i < a.size(); i++){
        for(int j = 0; j < a[0].size(); j++){
            result[i][j] = std::max(alpha*a[i][j], a[i][j]);
        }
    }

    return result;
}

std::vector<std::vector<double>> applyActivation(const std::vector<std::vector<double>>& data, std::string activation)
{
    if(activation == "RELU")
    {
        return RELU(data);
    }
    else if(activation == "Sigmoid")
    {
        return Sigmoid(data);
    }
    else if(activation == "Tanh")
    {
        return Tanh(data);
    }
    else if(activation == "Softmax")
    {
        return Softmax(data);
    }
    else if(activation == "LeakyRELU")
    {
        return LeakyRELU(data, 0.01);
    }
    else if(activation == "None")
    {
        return data;
    }
    else
    {
        throw std::invalid_argument("Invalid activation function.");
    }
}

// Now let us write the derivatives of these activation functions

std::vector<std::vector<double>> RELU_derivative(const std::vector<std::vector<double>>& a){
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
    for(int i = 0; i < a.size(); i++){
        for(int j = 0; j < a[0].size(); j++){
            result[i][j] = (a[i][j] > 0) ? 1.0 : 0.0;
        }
    }

    return result;
}

std::vector<std::vector<double>> Sigmoid_derivative(const std::vector<std::vector<double>>& a){
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
    for(int i = 0; i < a.size(); i++){
        for(int j = 0; j < a[0].size(); j++){
            result[i][j] = a[i][j] * (1 - a[i][j]);
        }
    }

    return result;
}

std::vector<std::vector<double>> Tanh_derivative(const std::vector<std::vector<double>>& a){
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
    for(int i = 0; i < a.size(); i++){
        for(int j = 0; j < a[0].size(); j++){
            result[i][j] = 1 - pow(a[i][j], 2);
        }
    }

    return result;
}

std::vector<std::vector<double>> Softmax_derivative(const std::vector<std::vector<double>>& a){
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
    for(int i = 0; i < a.size(); i++){
        for(int j = 0; j < a[0].size(); j++){
            result[i][j] = a[i][j] * (1 - a[i][j]);
        }
    }

    return result;
}

std::vector<std::vector<double>> LeakyRELU_derivative(const std::vector<std::vector<double>>& a, double alpha){
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
    for(int i = 0; i < a.size(); i++){
        for(int j = 0; j < a[0].size(); j++){
            result[i][j] = a[i][j] > 0 ? 1 : alpha;
        }
    }

    return result;
}

std::vector<std::vector<double>> applyActivation_derivative(const std::vector<std::vector<double>>& data, std::string activation)
{
    if(activation == "RELU")
    {
        return RELU_derivative(data);
    }
    else if(activation == "Sigmoid")
    {
        return Sigmoid_derivative(data);
    }
    else if(activation == "Tanh")
    {
        return Tanh_derivative(data);
    }
    else if(activation == "Softmax")
    {
        return Softmax_derivative(data);
    }
    else if(activation == "LeakyRELU")
    {
        return LeakyRELU_derivative(data, 0.01);
    }
    else if(activation == "None")
    {
        return data;
    }
    else
    {
        throw std::invalid_argument("Invalid activation function.");
    }
}