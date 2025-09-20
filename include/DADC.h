// Data-aware distance comparison
#ifndef DADC_H
#define DADC_H
#include "utils.h"

constexpr size_t GROUP_SIZE[] = {1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584};

inline point_coord_type innerProduct(const std::vector<point_coord_type> &v1,
                                     const std::vector<point_coord_type> &v2,
                                     size_t pca_dim)
{
    point_coord_type sum = 0.0;
    size_t i = 0;
    size_t size = v1.size();

    for (; i < pca_dim; ++i)
    {
        sum += v1[i] * v2[i];
    }
    for (; i < size; ++i)
    {
        sum += v1[i] * v2[i];
    }
    return sum;
}

inline point_coord_type innerProduct(const std::vector<point_coord_type> &v1, point_coord_type &rest, size_t pca_dim)
{
    point_coord_type sum1 = 0.0;

    size_t i = 0;
    for (; i < pca_dim; ++i)
    {
        sum1 += v1[i] * v1[i];
    }
    point_coord_type part = sum1;

    size_t size = v1.size();
    for (; i < size; ++i)
    {
        sum1 += v1[i] * v1[i];
    }
    rest = std::sqrt(sum1 - part);
    return sum1;
}

/**
 * Principal-component-aware distance comparison.
 *
 * Computes squared Euclidean distance between two vectors
 * with early termination based on the most informative dimensions
 * (e.g., top PCA components explaining 90% of variance).
 *
 * This function avoids full-distance computation if the partial sum already
 * exceeds the specified upper bound.
 */
template <typename T>
inline T dist_comp(const std::vector<T> &a, const std::vector<T> &b,
                   AdaptPoint &p, CentroidNormSquare &c, T &bound, size_t &d1)
{
    T sum = 0;
    size_t i = 0;
    for (; i < d1; ++i)
    {
        T diff = a[i] * b[i];
        sum += diff;
    }

    // 距离估计
    T dist_square_hat = p.total_normSquare + c.total_normSquare - 2 * (sum + p.rest_norm * c.rest_norm);
    if (dist_square_hat > bound)
        return dist_square_hat;

    size_t dim = a.size();
    for (; i < dim; ++i)
    {
        T diff = a[i] * b[i];
        sum += diff;
    }
    sum = p.total_normSquare + c.total_normSquare - 2 * sum;
    return sum > 0 ? sum : 0.0;
}

template <typename T>
inline T dist_comp(const std::vector<T> &a, const std::vector<T> &b,
                   AdaptPointV2 &p, CentroidNormSquareV2 &c, T &bound, size_t &d1)
{
    T sum = 0;
    size_t i = 0;
    for (; i < d1; ++i)
    {
        T diff = a[i] * b[i];
        sum += diff;
    }

    // 距离估计
    T dist_square_hat = p.total_normSquare + c.total_normSquare - 2 * (sum + p.rest_norm * c.rest_norm);
    if (dist_square_hat >= bound)
        return dist_square_hat;

    size_t dim = a.size();
    for (; i < dim; ++i)
    {
        T diff = a[i] * b[i];
        sum += diff;
    }
    sum = p.total_normSquare + c.total_normSquare - 2 * sum;
    return sum > 0 ? sum : 0.0;
}

template <typename T>
inline T dist_comp(const std::vector<T> &a, const std::vector<T> &b,
                   AdaptPointV2 &p, CentroidNormSquareV2 &c, T &bound, size_t &d1, size_t &feature_cnt)
{
    T sum = 0;
    size_t i = 0;
    for (; i < d1; ++i)
    {
        T diff = a[i] * b[i];
        sum += diff;
    }
    feature_cnt += d1;
    // 距离估计
    T dist_square_hat = p.total_normSquare + c.total_normSquare - 2 * (sum + p.rest_norm * c.rest_norm);
    if (dist_square_hat >= bound)
        return dist_square_hat;

    size_t dim = a.size();
    feature_cnt += dim - d1;
    for (; i < dim; ++i)
    {
        T diff = a[i] * b[i];
        sum += diff;
    }
    sum = p.total_normSquare + c.total_normSquare - 2 * sum;
    return sum > 0 ? sum : 0.0;
}

template <typename T>
inline T dist_comp(const std::vector<T> &a, const std::vector<T> &b,
                   CentroidNormSquareV2 &p, CentroidNormSquareV2 &c, T &bound, size_t &d1, size_t &feature_cnt)
{
    T sum = 0;
    size_t i = 0;
    for (; i < d1; ++i)
    {
        T diff = a[i] * b[i];
        sum += diff;
    }
    feature_cnt += d1;
    // 距离估计
    T dist_square_hat = p.total_normSquare + c.total_normSquare - 2 * (sum + p.rest_norm * c.rest_norm);
    if (dist_square_hat >= bound)
        return dist_square_hat;

    size_t dim = a.size();
    feature_cnt += dim - d1;
    for (; i < dim; ++i)
    {
        T diff = a[i] * b[i];
        sum += diff;
    }
    sum = p.total_normSquare + c.total_normSquare - 2 * sum;
    return sum > 0 ? sum : 0.0;
}

template <typename T>
inline T dist_comp(const std::vector<T> &a, const std::vector<T> &b, T &bound, size_t &d1)
{
    T sum = 0;
    size_t i = 0;
    for (; i < d1; ++i)
    {
        T diff = a[i] - b[i];
        sum += diff * diff;
    }
    // 距离估计
    // point_coord_type dist_square_hat = sum;
    if (sum > bound)
        return sum;

    size_t dim = a.size();
    for (; i < dim; ++i)
    {
        T diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum > 0 ? sum : 0.0;
}

template <typename T>
inline T dist_comp(const std::vector<T> &a, const std::vector<T> &b, T &bound, size_t &d1, size_t &feature_cnt)
{
    T sum = 0;
    size_t i = 0;
    for (; i < d1; ++i)
    {
        T diff = a[i] - b[i];
        sum += diff * diff;
    }
    // 距离估计
    // point_coord_type dist_square_hat = sum;
    if (sum > bound)
    {
        feature_cnt += d1;
        return sum;
    }

    size_t dim = a.size();
    feature_cnt += dim;
    for (; i < dim; ++i)
    {
        T diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum > 0 ? sum : 0.0;
}

#endif // DADC_H