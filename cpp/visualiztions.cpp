#include "helpers.hpp"

using namespace std;
using namespace mlpack;
using namespace arma;
using namespace mlpack::data;


int main()
{
    // Load the dataset
    string filename = "../../data/drug200.csv";  // Adjust the path as needed
    arma::mat train;
    data:: DatasetInfo info;
    mlpack::data::Load(filename, train, info); 

    // remove header
    train.shed_col(0);

    // Access individual columns
    arma::rowvec age = train.row(0);
    arma::rowvec sex = train.row(1);
    arma::rowvec bp = train.row(2);
    arma::rowvec cholesterol = train.row(3);
    arma::rowvec na_to_k = train.row(4);
    arma::rowvec drug = train.row(5);

    print_histogram(age, "Age Distribution", "age_distribution.png");
    print_histogram(sex, "Sex Distribution", "sex_distribution.png");
    print_histogram(bp, "Blood Pressure Distribution", "bp_distribution.png");
    print_histogram(cholesterol, "Cholesterol Distribution", "cholesterol_distribution.png");
    print_histogram(drug, "Drug Distribution", "drug_distribution.png");


    // Compute the correlation matrix
    arma::mat corr_matrix(train.n_rows + 1, train.n_rows + 1, arma::fill::zeros);

    calculate_correlation_matrix(train, corr_matrix, drug);


    // Save the correlation matrix
    save_correlation_matrix(corr_matrix, "correlation_matrix.dat");

    // Plot the heatmap using Gnuplot
    plot_correlation_map("correlation_matrix.dat");

    move_files();



    return 0;
}
