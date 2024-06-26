#include <iostream>
#include "layers.cpp"
#include <vector>
#include <cmath>

class MLP 
{
    private:
        int input_dim, num_neurons, output_dim;
        string activations;
        FCLayer fc1;
        FCLayer fc2;
        FCLayer fc3;
        double learning_rate;

    public:

        MLP(int input_dim, int num_neurons, int output_dim, double learning_rate, string activations)
        {
            this->input_dim = input_dim;
            this->num_neurons = num_neurons;
            this->output_dim = output_dim;
            this->learning_rate = learning_rate;
            this->activations = activations;

            this->fc1 = FCLayer(input_dim, num_neurons, activations);
            this->fc2 = FCLayer(num_neurons, num_neurons, activations);
            this->fc3 = FCLayer(num_neurons, output_dim, "Softmax"); // To generate a probability distribution across all the classes. Need to also update backprop to incorporate derivative of softmax.
        }

        vector<vector<double>> forward(vector<vector<double>> x)
        {
            // Forward pass
            vector<vector<double>> out = fc1.forward(x);
            out = fc2.forward(out);
            out = fc3.forward(out);
            return out;
        }

        double cross_entropy_loss(vector<vector<double>>& preds, vector<vector<double>>& true_vals) {

            double loss = 0.0;
            double epsilon = 1e-15;  // Small value to prevent log(0)

            for(size_t i = 0; i < preds.size(); i++) {
                for(size_t j = 0; j < preds[i].size(); j++) {
                loss -= true_vals[i][j] * log(max(preds[i][j], epsilon));
                }
            }

            return loss / preds.size();
        }

        double get_classification_loss(vector<vector<double>> x, vector<vector<double>> val){
            
        }

        void backpass(vector<vector<double>> true_vals){

            vector<vector<double>> preds = this->forward(this->fc1.input_data);

            vector<vector<double>> dlda(preds.size(), vector<double>(true_vals[0].size(), 0.0));

            for(int i = 0; i<preds.size(); i++){
                for(int j = 0; j<preds[0].size(); j++){
                    dlda[i][j] = -true_vals[i][j] / preds[i][j];
                }
            }

            vector<vector<double>> softmax_der = Softmax_derivative(preds);

            // dlda = loss_der * softmax_der (hadamard product in this case because every element individually results in same output through softmax regardless of order in the row)
            dlda = hadamard(dlda, softmax_der);

            // Initiating backpass
            vector<vector<double>>third_grad = this->fc3.backprop(dlda);
            this->fc3.update_params(this->learning_rate);

            vector<vector<double>>second_grad = this->fc2.backprop(third_grad); // Debugging step -- Removed the transpose operation third grad operation
            this->fc2.update_params(learning_rate);

            vector<vector<double>> first_grad = this->fc1.backprop(second_grad);
            this->fc1.update_params(learning_rate);
        }

        void log_max_values(const vector<vector<double>>& matrix, const string& name) {
            double max_val = 0;
            for (const auto& row : matrix) {
                for (const auto& val : row) {
                max_val = max(max_val, abs(val));
                }
            }
            cout << "Max " << name << ": " << max_val << endl;
        }

        void train(int num_epochs, double learning_rate, vector<vector<double>> x, vector<vector<double>> y){
            this->learning_rate = learning_rate;

            for(int epoch = 0; epoch<num_epochs; epoch++){
                vector<vector<double>> preds = this->forward(x);
                this->backpass(y);
                double loss = this->cross_entropy_loss(preds, y);;

                if((epoch+1) % 10 == 0){
                    cout << "Epoch: " << (epoch+1) << " || Loss: " << loss << "\n";
                    log_max_values(fc1.get_weights(), "fc1 weights");
                    log_max_values(fc2.get_weights(), "fc2 weights");
                    log_max_values(fc3.get_weights(), "fc3 weights");

                }
            }
        }

        double evaluate(vector<vector<double>> x, vector<vector<double>> y) {
            vector<vector<double>> results = this->forward(x);
            double accuracy = 0.0;
            vector<int> correct_args = argmax(y);
            vector<int> predicted_args = argmax(results);

            for(int i = 0; i < y.size(); i++){
            if(correct_args[i] == predicted_args[i]){
                accuracy = accuracy + 1;
            }
            }

            return accuracy/y.size();
        }
};

