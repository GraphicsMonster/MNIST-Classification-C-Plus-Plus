#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include "mlp.cpp"

using namespace std;

int main() {

    // Loading the data from mnist csv files
    ifstream file("./MNIST_dataset/mnist_train.csv");

    if(!file.is_open()){
        cout << "Error opening file\n";
        return 0;
    }

    vector<vector<double>> x;
    vector<double> y;

    string line;
    getline(file, line); // removing the first row


    while(getline(file, line)){
        vector<double> row;
        string val;
        stringstream s(line);
        while(getline(s, val, ',')){
            row.push_back(stod(val)/784.0);
        }
        y.push_back(row[0]);
        row.erase(row.begin());
        x.push_back(row);
    }

    // Normalized the lables too lol. Bugged me for hours. Fixed it here.
    for(int i = 0; i<y.size(); i++){
        y[i] *= 784.0;
    }

    // One hot encoding the labels
    vector<vector<double>> one_hot_labels(y.size(), vector<double>(10, 0));

    for(int i = 0; i<y.size(); i++){
        one_hot_labels[i][y[i-1]] = 1;
    }

    vector<vector<double>> train_set_x;
    vector<vector<double>> test_set_x;
    vector<vector<double>> train_set_y;
    vector<vector<double>> test_set_y;

    // Train test split
    int training_size = 400;

    for(int i = 0; i<training_size; i++){
        train_set_x.push_back(x[i]);
        train_set_y.push_back(one_hot_labels[i]);
    }

    // Test set split
    for(int i=1000; i<1100; i++){
        test_set_x.push_back(x[i]);
        test_set_y.push_back(one_hot_labels[i]);
    }

    // Hyperparams
    int input_features = x[0].size();
    int num_neurons = 24;
    int output_dim = 10;
    double learning_rate = 2e-4;
    string activations = "RELU";

    cout << "Training set size: (" << train_set_x.size() << ", " << train_set_x[0].size() << ") \n";

    MLP model = MLP(input_features, num_neurons, output_dim, learning_rate, activations);

    // Training the model
    model.train(100, learning_rate, train_set_x, train_set_y);

    // Evaluation
    double accuracy = model.evaluate(test_set_x, test_set_y) * 100;
    cout << "The model's accuracy on the test set is: " << accuracy << "%. \n";
    return 0;
}
