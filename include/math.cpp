#include "math.hpp"
#include <random>
#include <iostream>


// Global random number generator and distribution
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-0.1, 0.1);

std::vector<std::vector<double>> Multiply_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    if (A[0].size() != B.size()) {
        throw std::invalid_argument("The number of columns in the first matrix must be equal to the number of rows in the second matrix.");
    }

    // Dimensions of the resulting matrix
    size_t rows = A.size();
    size_t cols = B[0].size();
    size_t inner_dim = B.size();

    // Initialize the result matrix with zeros
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0.0));

    // Perform matrix multiplication
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < inner_dim; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}


std::vector<double> Multiply_vector_matrix(std::vector<double>& data, std::vector<std::vector<double>>& weights)
{
    if(data.size() != weights.size())
    {
        throw std::invalid_argument("The second matrix should have the same number of rows as that of columns in the first matrix.");
    }

    std::vector<double> output(weights[0].size(), 0.0);

    for(int i = 0; i<weights[0].size(); i++)
        {
            for(int j = 0; j<data.size(); j++)
            {
                output[i] += data[j] * weights[j][i];
            }
        }
    

    return output;
}

std::vector<double> multiply_vectors(std::vector<double> a, std::vector<double> b)
{
    if(a.size() != b.size())
    {
        throw std::invalid_argument("The two vectors should have the same size.");
    }

    std::vector<double> output(a.size(), 0.0);

    for(int i = 0; i<a.size(); i++)
    {
        output[i] = a[i] * b[i];
    }

    return output;
}

std::vector<double> Add_vectors(std::vector<double> data, std::vector<double> biases)
{
    for(int i = 0; i<biases.size(); i++){
        data[i] += biases[i];
    }

    return data;
}

std::vector<std::vector<double>> Add_biases(const std::vector<std::vector<double>>& matrix, const std::vector<double>& biases) {
    if (matrix.empty() || biases.empty() || matrix[0].size() != biases.size()) {
        throw std::invalid_argument("Matrix and biases dimensions do not match.");
    }

    std::vector<std::vector<double>> result = matrix;

    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = 0; j < result[0].size(); ++j) {
            result[i][j] += biases[j];
        }
    }

    return result;
}



double getRandomDouble() {
    return dis(gen);
}

double getXavierValue(int input_dim, int output_dim) {
    std::normal_distribution<> d(0, std::sqrt(2.0 / (input_dim + output_dim)));
    return d(gen);
}

// Function to initialize the matrix with Xavier initialization
std::vector<std::vector<double>> initializeMatrix(int rows, int cols) {
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            matrix[i][j] = getXavierValue(rows, cols);
        }
    }

    return matrix;
}

std::vector<double> initializeBias(int size) {
    std::vector<double> bias(size);
    
    for(int i = 0; i < size; ++i) {
        bias[i] = getXavierValue(1, size);
    }

    return bias;
}

// Function to print the matrix
void printMatrix(const std::vector<std::vector<double>>& matrix) {
    for(const auto& row : matrix) {
        for(const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n \n";
    }
}

void printvector(const std::vector<double>& vec) {
    for(const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << "\n";
}


std::vector<std::vector<double>> transposeMatrix(const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty()) return {}; // Return an empty matrix if input is empty

    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<double>> transposedMatrix(cols, std::vector<double>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposedMatrix[j][i] = matrix[i][j];
        }
    }

    return transposedMatrix;
}

std::vector<std::vector<double>> hadamard(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2) {
    std::vector<std::vector<double>> result(matrix1.size(), std::vector<double>(matrix2[0].size()));

    for(int i = 0; i<matrix1.size(); i++){
        for(int j = 0; j<matrix2[0].size(); j++){
            result[i][j] = matrix1[i][j] * matrix2[i][j];
        }  
    }

    return result;
}

std::vector<double> calculate_bias_derivatives(const std::vector<std::vector<double>>& dldz) {
    // Number of neurons (columns) in dldz
    int num_neurons = dldz[0].size();
    // Initialize bias_der vector with zeros
    std::vector<double> bias_der(num_neurons, 0.0);
    
    // Sum the gradients for each neuron across all samples in the batch
    for (int i = 0; i < dldz.size(); ++i) { // Iterate over rows (samples)
        for (int j = 0; j < num_neurons; ++j) { // Iterate over columns (neurons)
            bias_der[j] += dldz[i][j];
        }
    }
    
    return bias_der;
}

std::vector<int> argmax(const std::vector<std::vector<double>>& x){
    std::vector<int> result;
    for (const auto& row : x) {
        double maxVal = row[0];
        int maxIdx = 0;
        for (int i = 1; i < row.size(); ++i) {
            if (row[i] > maxVal) {
                maxVal = row[i];
                maxIdx = i;
            }
        }
        result.push_back(maxIdx);
    }

    return result;
}