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

    // Logistic Regression Model
    LogisticRegression lr;
    lr.Train(train, y);

    arma::Row<size_t> y_pred;
    lr.Classify(train, y_pred);

    double correct = arma::accu(y == y_pred);
    cout << "Accuracy of Logistic Regression: " << correct / y.n_elem << endl;

    // K Fold Cross Validation

    // KFoldCV<LogisticRegression<>, Accuracy> cv(10, train,info, y);
    // double accuracy = cv.Evaluate(8);

    // cout << "Accuracy of Logistic Regression: " << accuracy << endl;

    
}