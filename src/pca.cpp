#include "pca.h"
constexpr size_t DIM_THRESH = 13;

void determine_pca_dim(std::vector<point_coord_type> &eigenvalues, size_t &pca_dim, point_coord_type percent)
{
    // 确定pca_dim
    point_coord_type sum = 0.0;
    for (size_t i = 0; i < eigenvalues.size(); i++)
    {
        sum += eigenvalues[i];
    }
    point_coord_type thresh_variance = percent * sum;
    point_coord_type explained = 0.0;
    for (; pca_dim < eigenvalues.size(); ++pca_dim)
    {
        explained += eigenvalues[pca_dim];
        if (explained >= thresh_variance && pca_dim > DIM_THRESH)
            break;
    }
    pca_dim = pca_dim == eigenvalues.size() ? eigenvalues.size() : pca_dim + 1;
    std::cout << "Dimension reduction to " << pca_dim << " dimensions, explaining " << explained / sum * 100.0 << "% of the total variance" << std::endl;
}

void determine_pca_dim_manual(std::vector<point_coord_type> &eigenvalues, size_t &pca_dim)
{
    for (size_t i = 0; i < eigenvalues.size(); i++)
    {
        std::cout << eigenvalues[i] << " ";
    }
    std::cout << std::endl;
    std::cin >> pca_dim;
    if (pca_dim > eigenvalues.size())
    {
        std::cerr << "Input error, pca_dim cannot be greater than the number of eigenvalues.。\n";
        return;
    }
    point_coord_type total_variance = std::accumulate(eigenvalues.begin(), eigenvalues.end(), 0.0);
    point_coord_type retained_variance = std::accumulate(eigenvalues.begin(), eigenvalues.begin() + pca_dim, 0.0);
    std::cout << "Dimension reduction to " << pca_dim << " dimensions, explaining "
              << retained_variance / total_variance * 100.0
              << "% of the total variance" << std::endl;
}

void performPCA(const std::string &filename, const Matrix<point_coord_type> &data,
                Matrix<point_coord_type> &pca_data,
                size_t &pca_dim, point_coord_type percent)
{
    size_t n_samples = data.size(), n_features = data[0].size();
    Eigen::MatrixXd eigen_data(n_samples, n_features);
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < n_features; ++j)
            eigen_data(i, j) = data[i][j];
    Eigen::VectorXd mean = eigen_data.colwise().mean();
    eigen_data.rowwise() -= mean.transpose();
    Eigen::MatrixXd cov = (eigen_data.adjoint() * eigen_data) / (eigen_data.rows() - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
    Eigen::MatrixXd eigenvectors = eig.eigenvectors();
    Eigen::VectorXd eigenvalues = eig.eigenvalues();

    // 特征值从大到小排序
    std::vector<std::pair<point_coord_type, int>> eigen_pairs;
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i)
        eigen_pairs.push_back(std::make_pair(eigenvalues(i), i));
    std::sort(eigen_pairs.begin(), eigen_pairs.end(),
              [](const std::pair<point_coord_type, int> &a, const std::pair<point_coord_type, int> &b)
              {
                  return a.first > b.first;
              });
    std::vector<point_coord_type> eigenvalues_sorted;
    for (size_t i = 0; i < n_features; i++)
        eigenvalues_sorted.push_back(eigen_pairs[i].first);
    determine_pca_dim(eigenvalues_sorted, pca_dim, percent);

    // 构建投影矩阵
    Eigen::MatrixXd projection_matrix(eigenvectors.rows(), n_features);
    for (size_t i = 0; i < n_features; ++i)
    {
        size_t idx = eigen_pairs[i].second;
        Eigen::VectorXd vec = eigenvectors.col(idx);
        vec.normalize();
        projection_matrix.col(i) = vec;
    }

    Eigen::MatrixXd reduced_data = eigen_data * projection_matrix;
    pca_data.resize(n_samples, std::vector<point_coord_type>(n_features));
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < n_features; ++j)
            pca_data[i][j] = reduced_data(i, j);

    std::ofstream out(filename, std::ios::binary);
    if (!out)
        throw std::runtime_error("Unable to open file for writing: " + filename);
    int dim = static_cast<int>(projection_matrix.cols());
    out.write(reinterpret_cast<const char *>(&dim), sizeof(int));
    out.write(reinterpret_cast<const char *>(eigenvalues_sorted.data()), sizeof(double) * dim);
    out.close();
}

void random_orthogonal_transform(
    const Matrix<point_coord_type> &data,
    Matrix<point_coord_type> &transformed_data, int &rnd_seed)
{
    size_t dim = data[0].size();
    size_t n_samples = data.size();

    // 生成随机矩阵
    Eigen::MatrixXd random_matrix(dim, dim);
    static std::mt19937 gen(rnd_seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    random_matrix = Eigen::MatrixXd::NullaryExpr(dim, dim, [&]()
                                                 { return dist(gen); });

    // QR 分解得到正交矩阵 Q
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(random_matrix);
    Eigen::MatrixXd Q = qr.householderQ();

    // 转换输入数据到 Eigen
    Eigen::MatrixXd data_eigen(n_samples, dim);
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < dim; ++j)
            data_eigen(i, j) = data[i][j];

    // 取前 n_features 列进行变换
    Eigen::MatrixXd transformed = (data_eigen * Q);

    // 转回自定义 Matrix 类型
    transformed_data = data;
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < dim; ++j)
            transformed_data[i][j] = transformed(i, j);
}

std::vector<double> compute_dimension_variance(const Matrix<point_coord_type> &data)
{
    size_t n_samples = data.size();
    size_t dim = data[0].size();
    std::vector<double> variance(dim, 0.0);
    std::vector<double> mean(dim, 0.0);

    // 1) 计算均值
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < dim; ++j)
            mean[j] += data[i][j];
    for (size_t j = 0; j < dim; ++j)
        mean[j] /= n_samples;

    // 2) 计算方差
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < dim; ++j)
        {
            double diff = data[i][j] - mean[j];
            variance[j] += diff * diff;
        }
    for (size_t j = 0; j < dim; ++j)
        variance[j] /= (n_samples - 1);

    return variance;
}

double lower_bound_transform(std::vector<double> &a, std::vector<double> &b, size_t pca_dim)
{
    double sum = 0.0;
    for (size_t i = 0; i < pca_dim; ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    double norm_a = 0.0, norm_b = 0.0;
    for (size_t i = pca_dim; i < a.size(); i++)
    {
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return sum + norm_a + norm_b - 2 * std::sqrt(norm_a * norm_b);
}
