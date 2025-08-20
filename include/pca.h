#pragma once
#include <fstream>
#include <string>
#include <utility>
#include <Eigen/Dense>
#include <random>

#include "utils.h"

void determine_pca_dim(std::vector<point_coord_type> &eigen_pairs, size_t &pca_dim, point_coord_type percent);
void determine_pca_dim_manual(std::vector<point_coord_type> &eigenvalues, size_t &pca_dim);
void performPCA(const std::string &filename, const Matrix<point_coord_type> &data,
                Matrix<point_coord_type> &pca_data,
                size_t &pca_dim, point_coord_type percent);

void random_orthogonal_transform(const Matrix<point_coord_type> &data,
                                 Matrix<point_coord_type> &transformed_data, int &rnd_seed);
std::vector<double> compute_dimension_variance(const Matrix<point_coord_type> &data);
double lower_bound_transform(std::vector<double> &a, std::vector<double> &b, size_t pca_dim);