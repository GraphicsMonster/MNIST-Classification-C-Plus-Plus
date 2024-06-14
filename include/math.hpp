#ifndef MATH_H
#define MATH_H

#include <vector>
#include <stdexcept>

std::vector<std::vector<double>> Multiply_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);
std::vector<double> Multiply_vector_matrix(std::vector<double>& data, std::vector<std::vector<double>>& weights);
std::vector<double> multiply_vectors(std::vector<double> a, std::vector<double> b);
std::vector<double> Add_vectors(std::vector<double> data, std::vector<double> biases);
std::vector<std::vector<double>> Add_biases(const std::vector<std::vector<double>>& matrix, const std::vector<double>& biases);
std::vector<std::vector<double>> Add_biases(const std::vector<std::vector<double>>& matrix, const std::vector<double>& biases);
double getRandomDouble(double min, double max);
std::vector<std::vector<double>> initializeMatrix(int rows, int cols);
std::vector<double> initializeBias(int size);
void printMatrix(const std::vector<std::vector<double>>& matrix);
std::vector<std::vector<double>> transposeMatrix(const std::vector<std::vector<double>>& matrix);
std::vector<std::vector<double>> hadamard(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2);
std::vector<double> calculate_bias_derivatives(const std::vector<std::vector<double>>& dldz);
double getXavierValue(int input_dim, int output_dim);
std::vector<int> argmax(const std::vector<std::vector<double>>& x);
void printvector(const std::vector<double>& vec);

#endif
