#include <iostream>
#include <vector>
#include "../include/math.hpp"
#include "../include/activations.hpp"

using namespace std;

class FCLayer
{
    // A fully connected layer

    private:

        vector<vector<double>> weights;
        vector<double> biases;
        string activation;
        vector<vector<double>> raw;
        vector<vector<double>> weight_der;
        vector<double> bias_der;
    
    public:

        vector<vector<double>> input_data;

        FCLayer(){
            // Default constructor
        }

        FCLayer(int input_dim, int num_neurons, const string& activation){
            this->weights = initializeMatrix(input_dim, num_neurons);
            this->biases = initializeBias(num_neurons);
            this->activation = activation;
        }


        vector<vector<double>> forward(vector<vector<double>> data)
        {
            this->input_data = data;
            vector<vector<double>> output = Multiply_matrices(data, this->weights);
            output = Add_biases(output, this->biases);
            this->raw = output;
            output = applyActivation(output, this->activation);
            return output;
        }

        vector<vector<double>> backprop(vector<vector<double>> dlda){
            
            vector<vector<double>> activated_derivative = applyActivation_derivative(this->raw, this->activation);
            vector<vector<double>> dldz = hadamard(dlda, activated_derivative);

            this->weight_der = transposeMatrix(Multiply_matrices(transposeMatrix(dldz), this->input_data));
            this->bias_der = calculate_bias_derivatives(dldz);

            vector<vector<double>> dldx = Multiply_matrices(dldz, transposeMatrix(this->weights));
            return dldx;

        }

        vector<vector<double>> get_weights() {
            return this->weights;
        }

        void update_params(double lr){
            double clip_value = 5.0;
            for(int i = 0; i<this->weights.size(); i++){
                for(int j = 0; j<this->weights[0].size(); j++){
                    double gradient = this->weight_der[i][j];
                    gradient = max(min(gradient, clip_value), -clip_value);
                    this->weights[i][j] -= lr * gradient;
                }
            }

            for (size_t i = 0; i < biases.size(); ++i) {
                double gradient = this->bias_der[i];
                gradient = max(min(gradient, clip_value), -clip_value);
                biases[i] -= lr * gradient;
            }
        }

};

// TODO: My weight initialization sucks. NOt enough randomness. Fix it after lunch. -- Done
// TODO: Maybe use some standard initialization technique like xavier or whatever. -- Done