#pragma once
#include <vector>
#include <cstddef>
#include <cmath>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iomanip>

// 定义二维向量的别名
template <typename T>
using Matrix = std::vector<std::vector<T>>;

// 定义基本类型
typedef double point_coord_type;

// 定义点到中心距离的结构
struct Point
{
    point_coord_type distance;
    size_t index;
};

// 定义中心点结构
struct Center
{
    size_t cluster_count;
    size_t flag;
};

struct AdaptPoint
{
    point_coord_type distance;
    point_coord_type rest_norm;
    point_coord_type total_normSquare;
    point_coord_type total_norm;
    size_t init_clust;
    size_t group;
    size_t label;
};

struct CentroidNormSquare
{
    point_coord_type rest_norm;
    point_coord_type total_normSquare;
    point_coord_type total_norm;
};

struct AdaptPointV1
{
    point_coord_type distance;
    point_coord_type total_normSquare;
    size_t init_clust;
    size_t group;
    size_t label;
};

struct AdaptPointV2
{
    point_coord_type distance;
    point_coord_type rest_norm;
    point_coord_type total_normSquare;
    size_t init_clust;
    size_t group;
    size_t label;
};

struct CentroidNormSquareV2
{
    point_coord_type rest_norm;
    point_coord_type total_normSquare;
};

struct AdaptPointV4
{
    point_coord_type distance;
    size_t init_clust;
    size_t group;
    size_t label;
};

// 简单三元组结构存储稀疏矩阵非零元素
template <typename T>
struct SparseMatrix
{
    std::vector<size_t> rows;
    std::vector<size_t> cols;
    std::vector<T> values;
    size_t nrows = 0;
    size_t ncols = 0;
};

inline point_coord_type innerProduct(const std::vector<point_coord_type> &v1, const std::vector<point_coord_type> &v2)
{
    point_coord_type sum = 0.0;
    const size_t size = v1.size();

    for (size_t i = 0; i < size; ++i)
    {
        sum += v1[i] * v2[i];
    }

    return sum;
}

inline point_coord_type innerProduct(const std::vector<point_coord_type> &v1)
{
    point_coord_type sum = 0.0;
    const size_t size = v1.size();

    for (size_t i = 0; i < size; ++i)
    {
        sum += v1[i] * v1[i];
    }

    return sum;
}

// 计算欧氏距离平方（避免开方运算）
template <typename T>
inline T euclidean_dist_square(const std::vector<T> &a, const std::vector<T> &b)
{
    T sum = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        T diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum > 0 ? sum : 0.0;
}

// 计算欧氏距离
template <typename T>
inline T euclidean_dist(const std::vector<T> &a, const std::vector<T> &b)
{
    return std::sqrt(euclidean_dist_square(a, b));
}

// 计算欧氏距离
template <typename T>
inline T euclidean_dist(const std::vector<T> &a, const std::vector<T> &b, T a_normSquare, T b_normSquare)
{
    T temp = a_normSquare + b_normSquare - 2 * innerProduct(a, b);
    temp = temp <= 0 ? 0 : temp; // 确保非负
    return std::sqrt(temp);
}

template <typename T>
inline T euclidean_dist_square(const std::vector<T> &a, const std::vector<T> &b, T a_normSquare, T b_normSquare)
{
    T temp = a_normSquare + b_normSquare - 2 * innerProduct(a, b);
    return temp <= 0 ? 0 : temp; // 确保非负
}

template <typename T>
size_t getVectorMemoryBytes(const std::vector<T> &vec)
{
    return sizeof(std::vector<T>) + vec.capacity() * sizeof(T);
}

template <typename T>
size_t getMatrixMemoryBytes(const Matrix<T> &mat)
{
    size_t total = sizeof(Matrix<T>);
    total += mat.capacity() * sizeof(std::vector<T>);
    for (const auto &row : mat)
        total += row.capacity() * sizeof(T);
    return total;
}