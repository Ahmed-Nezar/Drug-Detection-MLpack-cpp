#include <iostream>
#include <iostream>
#include <vector>
#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <gnuplot-iostream.h>
#include <filesystem>
#include <cstdlib>

using namespace std;
using namespace mlpack;
using namespace arma;
using namespace mlpack::data;

static int counter = 0;

bool check_Duplicates(arma::rowvec train_id, arma::mat train);
void check_Null(arma::mat train);
map<double, int> get_counts(arma::rowvec column);
void print_counts(map<double, int> counts);
void print_histogram(arma::rowvec column, const string& title, const string& filename);
void save_correlation_matrix(const arma::mat& corr, const string& filename);
// void save_correlation_matrix(const arma::rowvec& corr, const string& filename);
void plot_correlation_map(const string& filename);


void save_correlation_matrix(const arma::mat& corr, const string& filename)
{
    ofstream outputFile(filename);
    if (outputFile.is_open())
    {
        for (size_t i = 0; i < corr.n_rows; ++i)
        {
            for (size_t j = 0; j < corr.n_cols; ++j)
            {
                outputFile << corr(i, j) << (j == corr.n_cols - 1 ? "\n" : " ");
            }
        }
        outputFile.close();
        cout << "Correlation matrix saved to '" << filename << "'" << endl;
    }
    else
    {
        cerr << "Error opening file '" << filename << "' for writing" << endl;
    }
}

void plot_correlation_map(const string& filename)
{
    Gnuplot gp;
    gp << "set terminal png size 1024,1024\n";
    gp << "set output 'correlation_heatmap_matrix.png'\n";
    gp << "set autoscale yfix\n";
    gp << "set autoscale xfix\n";
    // gp << "set palette defined (0 0 0 0.5, 1 0 0 1, 2 0 0.5 1, 3 0 1 1, 4 0.5 1 0.5, 5 1 1 0, 6 1 0.5 0, 7 1 0 0, 8 0.5 0 0)\n";
    gp << "set palette defined (0 0.2 0.2 0.6,1 0.2 0.4 0.8,2 0.2 0.6 1, 3 0.2 0.8 1,4 0.5 1 0.5,5 1 1 0,6 1 0.8 0,7 1 0.6 0,8 1 0.4 0.4)\n";
    gp << "set pm3d map\n";
    gp << "splot 'correlation_matrix.dat' matrix with image notitle\n";
}

void print_histogram(arma::rowvec column, const string& title, const string& filename)
{
    // Get the unique values and their counts
    map<double, int> counts = get_counts(column);
    print_counts(counts);

    // Create a Gnuplot object
    Gnuplot gp;
    
    gp << "set terminal png\n";
    gp << "set output '" << filename << "'\n";
    // Set the plot title and labels
    gp << "set title '" << title << "'\n";
    gp << "set xlabel 'Value'\n";
    gp << "set ylabel 'Count'\n";

    // Plot the data as a histogram
    gp << "set boxwidth 0.9 relative\n";
    gp << "set style fill solid 0.5\n";
    gp << "set yrange [0:*]\n"; // Set y-axis range to start from 0
    gp << "plot 'counts.dat' using 1:2 with boxes\n";
}

map<double, int> get_counts(arma::rowvec column)
{
    map<double, int> counts;
    for (int i = 0; i < column.size(); i++)
    {
        double value = column(i);
        counts[value]++;
    }
    return counts;
}

void print_counts(map<double, int> counts)
{
    ostringstream filenameStream;
    filenameStream << "counts.dat";
    counter++;
    string filename = filenameStream.str();
    ofstream outputFile(filename);
    for (const auto& pair : counts)
    {
        outputFile << pair.first << " " << pair.second << endl;
        cout << pair.first << " " << pair.second << endl;
    }
    outputFile.close();
    cout << "Counts saved to 'counts.dat'" << endl;
}

void check_Null(mat train)
{
    if (train.has_nan() == true)
    {
        cout << "Null values found in the train data" << endl;
    }
    else
    {
        cout << "No null values in the train data" << endl;
    }
}

bool check_Duplicates(arma::rowvec train_id, mat train)
{
    bool mask = false;
    for (int i = 0; i < train_id.size(); i++)
    {
         for (int j = i+1; j < train_id.size(); j++)
         {
              if (train_id(i) == train_id(j))
              {
                bool mask = true;
                train.shed_col(j);
                cout << "Duplicate passenger ID found and removed" << endl;
              }
         }
    }
    return mask;
}

void move_files()
{
    if (!filesystem::exists("visualizations")) {
        filesystem::create_directory("visualizations");
    }

    system("mv *.png visualizations/");

    if (!filesystem::exists("data")) {
        filesystem::create_directory("data");
    }

    system("mv *.dat data/");
}

// Function to calculate the correlation matrix
void calculate_correlation_matrix(const arma::mat& train, arma::mat& corr_matrix, const arma::rowvec& target)
{
    // Compute correlations between each feature and the target variable
    for (size_t i = 0; i < train.n_rows - 1; ++i)
    {
        arma::rowvec feature = train.row(i);
        double correlation_with_target = arma::as_scalar(arma::cor(feature, target));
        corr_matrix(i + 1, 0) = corr_matrix(0, i + 1) = correlation_with_target;
    }

    // Compute correlations between each pair of features
    for (size_t i = 0; i < train.n_rows - 1; ++i)
    {
        for (size_t j = i; j < train.n_rows - 1; ++j)
        {
            arma::rowvec feature1 = train.row(i);
            arma::rowvec feature2 = train.row(j);
            double correlation = arma::as_scalar(arma::cor(feature1, feature2));
            corr_matrix(i + 1, j + 1) = corr_matrix(j + 1, i + 1) = correlation;
        }
    }
}
