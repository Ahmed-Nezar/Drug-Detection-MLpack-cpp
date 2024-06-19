#include "helpers.hpp"

using namespace std;

int main()
{
    // Load the dataset
    string filename = "../data/drug200.csv"; 
    arma::mat train;
    data:: DatasetInfo info;
    mlpack::data::Load(filename, train, info); 

    // remove header
    train.shed_col(0);

    // Add x, y
    arma:: Row <size_t> y = arma::conv_to<arma::Row<size_t>>::from(train.row(5));
    train.shed_row(5);


    // Decision Tree Model
    DecisionTree dt;
    dt.Train(train,info, y,5);
    arma::Row<size_t> y_pred;
    dt.Classify(train, y_pred);
    double correct = arma::accu(y == y_pred);
    cout << "Accuracy of Decision Tree: " << correct / y.n_elem << endl;

    
}