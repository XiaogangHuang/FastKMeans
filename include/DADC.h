// Data-aware distance comparison
#ifndef DADC_H
#define DADC_H
#include "utils.h"

constexpr size_t GROUP_SIZE[] = {1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584};
constexpr size_t GROUP_SIZE_EXP[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
constexpr size_t GROUP_SIZE_FIX_2[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35};
constexpr size_t GROUP_SIZE_FIX_4[] = {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69};
constexpr size_t GROUP_SIZE_FIX_8[] = {1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137};

inline std::vector<size_t> getGroupSizesById(int mode)
{
    switch (mode)
    {
    case 0: // Fibonacci
        return std::vector<size_t>(std::begin(GROUP_SIZE), std::end(GROUP_SIZE));
    case 1: // Exponential
        return std::vector<size_t>(std::begin(GROUP_SIZE_EXP), std::end(GROUP_SIZE_EXP));
    case 2: // Fixed step 2
        return std::vector<size_t>(std::begin(GROUP_SIZE_FIX_2), std::end(GROUP_SIZE_FIX_2));
    case 3: // Fixed step 4
        return std::vector<size_t>(std::begin(GROUP_SIZE_FIX_4), std::end(GROUP_SIZE_FIX_4));
    case 4: // Fixed step 6
        return std::vector<size_t>(std::begin(GROUP_SIZE_FIX_8), std::end(GROUP_SIZE_FIX_8));
    default:
        throw std::invalid_argument("Invalid group mode id. Use 0~4.");
    }
}

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

inline point_coord_type innerProduct(const std::vector<point_coord_type> &v1, point_coord_type &partial, point_coord_type &mu,
                                     point_coord_type &sigma, point_coord_type &inv_d, size_t pca_dim)
{
    point_coord_type sum1 = 0.0;

    size_t i = 0;
    for (; i < pca_dim; ++i)
    {
        sum1 += v1[i] * v1[i];
    }
    partial = sum1;

    size_t size = v1.size();
    point_coord_type mu1 = 0.0;
    for (; i < size; ++i)
    {
        point_coord_type temp = v1[i];
        mu1 += temp;
        sum1 += temp * temp;
    }
    mu = mu1 * inv_d;
    sigma = std::sqrt((sum1 - partial) * inv_d - mu * mu);
    return sum1;
}

inline point_coord_type innerProduct(const std::vector<point_coord_type> &v1, point_coord_type &mu,
                                     point_coord_type &sigma, point_coord_type &inv_d, size_t pca_dim)
{
    point_coord_type partial = 0.0;
    size_t i = 0;
    for (; i < pca_dim; ++i)
    {
        partial += v1[i] * v1[i];
    }

    point_coord_type sum1 = 0.0;
    point_coord_type mu1 = 0.0;
    size_t sz = v1.size();
    for (; i < sz; ++i)
    {
        point_coord_type temp = v1[i];
        mu1 += temp;
        sum1 += temp * temp;
    }
    mu = mu1 * inv_d;
    sigma = std::sqrt(sum1 * inv_d - mu * mu);
    return partial + sum1;
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
inline T dist_comp_ms(const std::vector<T> &a, const std::vector<T> &b,
                      NormSquareV5 &p, NormSquareV5 &c, T &bound, size_t &d1, size_t &feature_cnt)
{
    T sum = 0;
    size_t i = 0;
    for (; i < d1; ++i)
    {
        T diff = a[i] * b[i];
        sum += diff;
    }
    // 距离估计
    point_coord_type temp_val = p.mu * c.mu + p.sigma * c.sigma;
    size_t dim = a.size();
    T dist_square_hat = p.total_normSquare + c.total_normSquare - 2 * (sum + temp_val * (dim - d1));
    if (dist_square_hat >= bound)
    {
        feature_cnt += d1;
        return dist_square_hat;
    }

    feature_cnt += dim;
    for (; i < dim; ++i)
    {
        T diff = a[i] * b[i];
        sum += diff;
    }
    sum = p.total_normSquare + c.total_normSquare - 2 * sum;
    return sum > 0 ? sum : 0.0;
}

template <typename T>
inline T dist_comp_ms(const std::vector<T> &a, const std::vector<T> &b,
                      NormSquareV5 &p, NormSquareV5 &c, T &bound, size_t &d1, size_t &numDistances, size_t &feature_cnt)
{
    T sum = 0;
    size_t i = 0;
    for (; i < d1; ++i)
    {
        T diff = a[i] * b[i];
        sum += diff;
    }
    // 距离估计
    point_coord_type temp_val = p.mu * c.mu + p.sigma * c.sigma;
    size_t dim = a.size();
    T dist_square_hat = p.total_normSquare + c.total_normSquare - 2 * (sum + temp_val * (dim - d1));
    if (dist_square_hat >= bound)
    {
        feature_cnt += d1;
        return dist_square_hat;
    }

    numDistances++;
    feature_cnt += dim;
    for (; i < dim; ++i)
    {
        T diff = a[i] * b[i];
        sum += diff;
    }
    sum = p.total_normSquare + c.total_normSquare - 2 * sum;
    return sum > 0 ? sum : 0.0;
}

template <typename T>
inline T dist_comp_bv(const std::vector<T> &a, const std::vector<T> &b,
                      const std::vector<T> &p, const std::vector<T> &c, T &bound, size_t &d1, size_t &feature_cnt)
{
    T sum_temp = p[0] + c[0];
    const point_coord_type *a_ptr = a.data();
    const point_coord_type *b_ptr = b.data();

    point_coord_type sum = 0.0;
    const point_coord_type *a_end = a_ptr + d1;
    while (a_ptr < a_end)
        sum += (*a_ptr++) * (*b_ptr++);

    feature_cnt += d1;
    sum_temp -= 2 * sum;

    T dist_square_hat = sum_temp - 2 * p[1] * c[1];
    if (dist_square_hat >= bound)
        return dist_square_hat;

    size_t dim = a.size();
    size_t jt = 2;

    for (size_t idx = d1; idx < dim; idx += d1, ++jt)
    {
        size_t sz_block = std::min(d1, dim - idx);
        feature_cnt += sz_block;

        point_coord_type block_sum = 0.0;
        const point_coord_type *a_end_block = a_ptr + sz_block;
        while (a_ptr < a_end_block)
            block_sum += (*a_ptr++) * (*b_ptr++);

        sum_temp -= 2 * block_sum;
        dist_square_hat = sum_temp - 2 * p[jt] * c[jt];

        if (dist_square_hat >= bound)
            return dist_square_hat;
    }

    return dist_square_hat > 0 ? dist_square_hat : 0.0;
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