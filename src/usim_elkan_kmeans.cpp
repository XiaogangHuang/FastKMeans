#include "usim_elkan_kmeans.h"

UltraSimplifiedElkan::UltraSimplifiedElkan(size_t k) : k(k), iterations(1), numDistances(0), n(0), d(0), feature_cnt(0) {}

UltraSimplifiedElkan::~UltraSimplifiedElkan() {}

std::vector<size_t> UltraSimplifiedElkan::getLabels()
{
    std::vector<size_t> labels(n, 0);
    for (size_t i = 0; i < n; ++i)
    {
        labels[i] = points[i].index;
    }
    return labels;
}

void UltraSimplifiedElkan::setInitialCentroids(const Matrix<point_coord_type> &initial_centroids, size_t dim)
{
    centroids = initial_centroids;
    old_centroids = initial_centroids;
    d = centroids[0].size();
    pca_dim = dim;
}

void UltraSimplifiedElkan::init(const Matrix<point_coord_type> &data)
{
    n = data.size();
    d = data[0].size();

    // 初始化
    lower_bounds.resize(n, std::vector<point_coord_type>(k, 0.0));

    c_to_c.resize(k, std::vector<point_coord_type>(k, 0.0));
    points.resize(n, Point{std::numeric_limits<point_coord_type>::max(), 0});
    centers.resize(k, Center{0, 0});
    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    point_normSquares.resize(n, CentroidNormSquareV2{0.0, 0.0});
    for (size_t i = 0; i < n; i++)
    {
        size_t ind = points[i].index;
        centers[ind].cluster_count++;
        for (size_t j = 0; j < d; j++)
        {
            sums[ind][j] += data[i][j];
        }
        point_normSquares[i].total_normSquare = innerProduct(data[i]);
    }
    centroid_normSquares.resize(k, CentroidNormSquareV2{0.0, 0.0});
    for (size_t i = 0; i < k; i++)
    {
        centroid_normSquares[i].total_normSquare = innerProduct(centroids[i]);
    }
    near.resize(k, 0.0);
    div.resize(k, 0.0);
}

void UltraSimplifiedElkan::fit(const Matrix<point_coord_type> &data)
{
    init(data);
    while (!assignPoints(data))
    {
        recalculateCentroids();
        updateBounds();
        iterations++;
    }
}

bool UltraSimplifiedElkan::assignPoints(const Matrix<point_coord_type> &data)
{
    bool converged = true;
    // 重新分配点
    for (size_t i = 0; i < n; ++i)
    {
        size_t old_label = points[i].index;
        point_coord_type val = near[old_label];
        points[i].distance += div[old_label];
        if (points[i].distance > val)
        {
            size_t ci = 0;
            while (ci < k)
            {
                if (ci != old_label && points[i].distance > lower_bounds[i][ci] && points[i].distance > 0.5 * c_to_c[old_label][ci])
                {
                    numDistances++;
                    feature_cnt += d;
                    points[i].distance = euclidean_dist(data[i], centroids[old_label],
                                                        point_normSquares[i].total_normSquare, centroid_normSquares[old_label].total_normSquare);
                    lower_bounds[i][old_label] = points[i].distance;
                    if (lower_bounds[i][ci] < points[i].distance && points[i].distance > 0.5 * c_to_c[old_label][ci])
                    {
                        numDistances++;
                        feature_cnt += d;
                        lower_bounds[i][ci] = euclidean_dist(data[i], centroids[ci], point_normSquares[i].total_normSquare,
                                                             centroid_normSquares[ci].total_normSquare);
                        if (points[i].distance > lower_bounds[i][ci])
                        {
                            points[i].distance = lower_bounds[i][ci];
                            points[i].index = ci;
                        }
                    }
                    ci++;
                    break;
                }
                ci++;
            }
            while (ci < k)
            {
                if (points[i].distance > lower_bounds[i][ci] && points[i].distance > 0.5 * c_to_c[points[i].index][ci])
                {
                    numDistances++;
                    feature_cnt += d;
                    lower_bounds[i][ci] = euclidean_dist(data[i], centroids[ci], point_normSquares[i].total_normSquare,
                                                         centroid_normSquares[ci].total_normSquare);
                    if (points[i].distance > lower_bounds[i][ci])
                    {
                        points[i].distance = lower_bounds[i][ci];
                        points[i].index = ci;
                    }
                }
                ci++;
            }

            if (old_label != points[i].index)
            {
                converged = false;
                for (size_t j = 0; j < d; j++)
                {
                    sums[old_label][j] -= data[i][j];
                }
                centers[old_label].cluster_count--;

                for (size_t j = 0; j < d; j++)
                {
                    sums[points[i].index][j] += data[i][j];
                }
                centers[points[i].index].cluster_count++;
            }
        }
    }

    return converged;
}

void UltraSimplifiedElkan::recalculateCentroids()
{
    std::swap(old_centroids, centroids);

    for (size_t i = 0; i < k; ++i)
    {
        if (centers[i].cluster_count > 0)
        {
            point_coord_type scale = 1.0 / centers[i].cluster_count;
            point_coord_type sum = 0;
            for (size_t j = 0; j < d; ++j)
            {
                centroids[i][j] = sums[i][j] * scale;
                sum += centroids[i][j] * centroids[i][j];
            }
            numDistances++;
            feature_cnt += d;
            div[i] = euclidean_dist(centroids[i], old_centroids[i], sum, centroid_normSquares[i].total_normSquare);
            centroid_normSquares[i].total_normSquare = sum;
        }
        else
        {
            centroids[i] = old_centroids[i];
            div[i] = 0;
        }
    }
}

void UltraSimplifiedElkan::updateBounds()
{
    // 计算中心点之间的距离
    for (size_t i = 0; i < k; ++i)
    {
        c_to_c[i][i] = 0;

        for (size_t j = i + 1; j < k; ++j)
        {
            numDistances++;
            feature_cnt += d;
            point_coord_type dist = euclidean_dist(centroids[i], centroids[j]);
            c_to_c[i][j] = dist;
            c_to_c[j][i] = dist;
        }

        // 计算每个中心点的最近邻居距离
        point_coord_type smallest = std::numeric_limits<point_coord_type>::max();
        for (size_t j = 0; j < k; ++j)
        {
            if (i == j)
                continue;
            if (c_to_c[i][j] < smallest)
            {
                smallest = c_to_c[i][j];
                near[i] = 0.5 * smallest;
            }
        }
    }

    // 更新所有点的下界
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            lower_bounds[i][j] = std::max(lower_bounds[i][j] - div[j], point_coord_type(0));
        }
    }
}

void UltraSimplifiedElkan::fit_stepwise(const Matrix<point_coord_type> &data)
{
    init_stepwise(data);
    while (!assignPoints_stepwise(data))
    {
        recalculateCentroids_stepwise();
        updateBounds();
        iterations++;
    }
}

void UltraSimplifiedElkan::init_stepwise(const Matrix<point_coord_type> &data)
{
    n = data.size();
    d = data[0].size();

    // 初始化
    lower_bounds.resize(n, std::vector<point_coord_type>(k, 0.0));

    c_to_c.resize(k, std::vector<point_coord_type>(k, 0.0));
    points.resize(n, Point{std::numeric_limits<point_coord_type>::max(), 0});
    centers.resize(k, Center{0, 0});
    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    point_normSquares.resize(n, CentroidNormSquareV2{0.0, 0.0});
    for (size_t i = 0; i < n; i++)
    {
        size_t ind = points[i].index;
        centers[ind].cluster_count++;
        for (size_t j = 0; j < d; j++)
        {
            sums[ind][j] += data[i][j];
        }
        point_coord_type temp = innerProduct(data[i], point_normSquares[i].rest_norm, pca_dim);
        point_normSquares[i].total_normSquare = temp;
    }
    centroid_normSquares.resize(k, CentroidNormSquareV2{0.0, 0.0});
    for (size_t i = 0; i < k; i++)
    {
        point_coord_type temp = innerProduct(centroids[i],
                                             centroid_normSquares[i].rest_norm, pca_dim);
        centroid_normSquares[i].total_normSquare = temp;
    }
    near.resize(k, 0.0);
    div.resize(k, 0.0);
}

bool UltraSimplifiedElkan::assignPoints_stepwise(const Matrix<point_coord_type> &data)
{
    bool converged = true;
    // 重新分配点
    for (size_t i = 0; i < n; ++i)
    {
        size_t old_label = points[i].index;
        point_coord_type val = near[old_label];
        points[i].distance += div[old_label];
        if (points[i].distance > val)
        {
            size_t ci = 0;
            point_coord_type thresh;
            while (ci < k)
            {
                if (ci != old_label && points[i].distance > lower_bounds[i][ci] && points[i].distance > 0.5 * c_to_c[old_label][ci])
                {
                    numDistances++;
                    feature_cnt += d;
                    thresh = euclidean_dist_square(data[i], centroids[old_label], point_normSquares[i].total_normSquare,
                                                   centroid_normSquares[old_label].total_normSquare);
                    points[i].distance = std::sqrt(thresh);
                    lower_bounds[i][old_label] = points[i].distance;
                    if (lower_bounds[i][ci] < points[i].distance && points[i].distance > 0.5 * c_to_c[old_label][ci])
                    {
                        numDistances++;
                        point_coord_type adist = dist_comp(data[i], centroids[ci], point_normSquares[i],
                                                           centroid_normSquares[ci], thresh, pca_dim, feature_cnt);
                        lower_bounds[i][ci] = std::sqrt(adist);
                        if (thresh > adist)
                        {
                            thresh = adist;
                            points[i].distance = lower_bounds[i][ci];
                            points[i].index = ci;
                        }
                    }
                    ci++;
                    break;
                }
                ci++;
            }
            while (ci < k)
            {
                if (points[i].distance > lower_bounds[i][ci] && points[i].distance > 0.5 * c_to_c[points[i].index][ci])
                {
                    numDistances++;
                    point_coord_type adist = dist_comp(data[i], centroids[ci], point_normSquares[i],
                                                       centroid_normSquares[ci], thresh, pca_dim, feature_cnt);
                    lower_bounds[i][ci] = std::sqrt(adist);

                    if (thresh > adist)
                    {
                        thresh = adist;
                        points[i].distance = lower_bounds[i][ci];
                        points[i].index = ci;
                    }
                }
                ci++;
            }

            if (old_label != points[i].index)
            {
                converged = false;
                for (size_t j = 0; j < d; j++)
                {
                    sums[old_label][j] -= data[i][j];
                }
                centers[old_label].cluster_count--;

                for (size_t j = 0; j < d; j++)
                {
                    sums[points[i].index][j] += data[i][j];
                }
                centers[points[i].index].cluster_count++;
            }
        }
    }
    return converged;
}

void UltraSimplifiedElkan::recalculateCentroids_stepwise()
{
    std::swap(old_centroids, centroids);

    for (size_t i = 0; i < k; ++i)
    {
        if (centers[i].cluster_count > 0)
        {
            point_coord_type scale = 1.0 / centers[i].cluster_count;
            for (size_t j = 0; j < d; ++j)
            {
                centroids[i][j] = sums[i][j] * scale;
            }
            point_coord_type temp = centroid_normSquares[i].total_normSquare;
            centroid_normSquares[i].total_normSquare = innerProduct(centroids[i], centroid_normSquares[i].rest_norm, pca_dim);
            temp += centroid_normSquares[i].total_normSquare - 2 * innerProduct(centroids[i], old_centroids[i], pca_dim);
            feature_cnt += d;
            numDistances++;
            div[i] = temp > 0 ? std::sqrt(temp) : 0.0;
        }
        else
        {
            centroids[i] = old_centroids[i];
            div[i] = 0;
        }
    }
}

void UltraSimplifiedElkan::init_ns(const Matrix<point_coord_type> &data)
{
    n = data.size();
    d = data[0].size();

    // 初始化
    lower_bounds.resize(n, std::vector<point_coord_type>(k, 0.0));
    // angle_matrix.resize(n, std::vector<point_coord_type>(k, 0.0));
    points.resize(n, Point{std::numeric_limits<point_coord_type>::max(), 0});
    centers.resize(k, Center{0, 0});
    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    point_normSquares.resize(n, CentroidNormSquareV2{0.0, 0.0});
    for (size_t i = 0; i < n; i++)
    {
        size_t ind = points[i].index;
        lower_bounds[i][ind] = std::numeric_limits<point_coord_type>::max();
        centers[ind].cluster_count++;
        for (size_t j = 0; j < d; j++)
        {
            sums[ind][j] += data[i][j];
        }
        point_normSquares[i].total_normSquare = innerProduct(data[i]);
    }
    centroid_normSquares.resize(k, CentroidNormSquareV2{0.0, 0.0});
    for (size_t i = 0; i < k; i++)
    {
        centroid_normSquares[i].total_normSquare = innerProduct(centroids[i]);
    }
    div_ns_.resize(k, 0.0);
    div_ns.reserve(100);

    timestamp.resize(n, std::vector<uint32_t>(k, 1));
}

void UltraSimplifiedElkan::fit_ns(const Matrix<point_coord_type> &data)
{
    init_ns(data);
    while (!assignPoints_ns(data))
    {
        recalculateCentroids_ns();
        iterations++;
    }
}

bool UltraSimplifiedElkan::assignPoints_ns(const Matrix<point_coord_type> &data)
{
    for (size_t i = 0; i < k; i++)
    {
        centers[i].flag = 1;
    }

    bool converged = true;
    for (size_t i = 0; i < n; ++i)
    {
        size_t old_label = points[i].index;
        point_coord_type lowersx_li = lower_bounds[i][old_label];
        uint32_t timestamp_old_label = timestamp[i][old_label];
        point_coord_type upperbound = timestamp_old_label < div_ns.size() ? lower_bounds[i][old_label] + div_ns[timestamp_old_label][old_label]
                                                                          : lower_bounds[i][old_label];
        size_t ci = 0;
        while (ci < k)
        {
            uint32_t timestamp_ci = timestamp[i][ci];
            point_coord_type lowersx_ci = timestamp_ci < div_ns.size() ? lower_bounds[i][ci] - div_ns[timestamp_ci][ci]
                                                                       : lower_bounds[i][ci];

            if (ci != old_label && upperbound > lowersx_ci)
            {
                numDistances++;
                feature_cnt += d;
                lowersx_li = euclidean_dist(data[i], centroids[old_label], point_normSquares[i].total_normSquare,
                                            centroid_normSquares[old_label].total_normSquare);
                lower_bounds[i][old_label] = lowersx_li;
                timestamp[i][old_label] = iterations - 1;
                if (lowersx_li > lowersx_ci)
                {
                    numDistances++;
                    feature_cnt += d;
                    lowersx_ci = euclidean_dist(data[i], centroids[ci], point_normSquares[i].total_normSquare,
                                                centroid_normSquares[ci].total_normSquare);
                    lower_bounds[i][ci] = lowersx_ci;
                    timestamp[i][ci] = iterations - 1;
                    if (lowersx_li > lowersx_ci)
                    {
                        lowersx_li = lowersx_ci;
                        points[i].index = ci;
                    }
                }
                ci++;
                break;
            }
            ci++;
        }
        while (ci < k)
        {
            uint32_t timestamp_ci = timestamp[i][ci];
            point_coord_type lowersx_ci = timestamp_ci < div_ns.size() ? lower_bounds[i][ci] - div_ns[timestamp_ci][ci]
                                                                       : lower_bounds[i][ci];
            if (lowersx_li > lowersx_ci)
            {
                numDistances++;
                feature_cnt += d;
                lowersx_ci = euclidean_dist(data[i], centroids[ci], point_normSquares[i].total_normSquare,
                                            centroid_normSquares[ci].total_normSquare);
                lower_bounds[i][ci] = lowersx_ci;
                timestamp[i][ci] = iterations - 1;
                if (lowersx_li > lowersx_ci)
                {
                    lowersx_li = lowersx_ci;
                    points[i].index = ci;
                }
            }
            ci++;
        }

        if (old_label != points[i].index)
        {
            converged = false;
            for (size_t j = 0; j < d; j++)
            {
                sums[old_label][j] -= data[i][j];
            }
            for (size_t j = 0; j < d; j++)
            {
                sums[points[i].index][j] += data[i][j];
            }
            centers[old_label].cluster_count--;
            centers[old_label].flag = 0;
            centers[points[i].index].cluster_count++;
            centers[points[i].index].flag = 0;
        }
    }
    return converged;
}

void UltraSimplifiedElkan::recalculateCentroids_ns()
{
    // 保存旧的中心点
    centroids_history.push_back(centroids);

    auto &last_centroids = centroids_history.back();
    for (size_t i = 0; i < k; i++)
    {
        if (centers[i].flag == 0 && centers[i].cluster_count > 0)
        {
            point_coord_type scale = 1.0 / centers[i].cluster_count;
            point_coord_type sum = 0.0;
            for (size_t j = 0; j < d; j++)
            {
                centroids[i][j] = sums[i][j] * scale;
                sum += centroids[i][j] * centroids[i][j];
            }

            numDistances++;
            feature_cnt += d;
            div_ns_[i] = euclidean_dist(centroids[i], last_centroids[i], sum, centroid_normSquares[i].total_normSquare);
            centroid_normSquares[i].total_normSquare = sum;
        }
        else
        {
            centroids[i] = last_centroids[i];
            div_ns_[i] = 0.0;
        }
    }
    div_ns.push_back(div_ns_);

    // 计算中心点移动距离 (Norm of sums)
    for (size_t i = 0; i < iterations - 1; i++)
    {
        for (size_t j = 0; j < k; ++j)
        {
            if (div_ns_[j] > 0.0)
            {
                numDistances++;
                feature_cnt += d;
                div_ns[i][j] = euclidean_dist(centroids[j], centroids_history[i][j]);
            }
        }
    }
}

void UltraSimplifiedElkan::init_ns_stepwise(const Matrix<point_coord_type> &data)
{
    n = data.size();
    d = data[0].size();

    // 初始化
    lower_bounds.resize(n, std::vector<point_coord_type>(k, 0.0));
    // angle_matrix.resize(n, std::vector<point_coord_type>(k, 0.0));
    points.resize(n, Point{std::numeric_limits<point_coord_type>::max(), 0});
    centers.resize(k, Center{0, 0});
    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    point_normSquares.resize(n, CentroidNormSquareV2{0.0, 0.0});
    for (size_t i = 0; i < n; i++)
    {
        size_t ind = points[i].index;
        lower_bounds[i][ind] = std::numeric_limits<point_coord_type>::max();
        centers[ind].cluster_count++;
        for (size_t j = 0; j < d; j++)
        {
            sums[ind][j] += data[i][j];
        }
        point_coord_type temp = innerProduct(data[i], point_normSquares[i].rest_norm, pca_dim);
        point_normSquares[i].total_normSquare = temp;
    }
    centroid_normSquares.resize(k, CentroidNormSquareV2{0.0, 0.0});
    for (size_t i = 0; i < k; i++)
    {
        point_coord_type temp = innerProduct(centroids[i], centroid_normSquares[i].rest_norm, pca_dim);
        centroid_normSquares[i].total_normSquare = temp;
    }
    div_ns_.resize(k, 0.0);
    div_ns.reserve(100);

    timestamp.resize(n, std::vector<uint32_t>(k, 1));
}

void UltraSimplifiedElkan::fit_ns_stepwise(const Matrix<point_coord_type> &data)
{
    init_ns_stepwise(data);
    while (!assignPoints_ns_stepwise(data))
    {
        recalculateCentroids_ns_stepwise();
        iterations++;
    }
}

bool UltraSimplifiedElkan::assignPoints_ns_stepwise(const Matrix<point_coord_type> &data)
{
    for (size_t i = 0; i < k; i++)
    {
        centers[i].flag = 1;
    }

    bool converged = true;
    for (size_t i = 0; i < n; ++i)
    {
        size_t old_label = points[i].index;
        point_coord_type lowersx_li = lower_bounds[i][old_label];
        uint32_t timestamp_old_label = timestamp[i][old_label];
        point_coord_type upperbound = timestamp_old_label < div_ns.size() ? lower_bounds[i][old_label] + div_ns[timestamp_old_label][old_label]
                                                                          : lower_bounds[i][old_label];
        size_t ci = 0;
        point_coord_type thresh;
        while (ci < k)
        {
            uint32_t timestamp_ci = timestamp[i][ci];
            point_coord_type lowersx_ci = timestamp_ci < div_ns.size() ? lower_bounds[i][ci] - div_ns[timestamp_ci][ci]
                                                                       : lower_bounds[i][ci];
            if (ci != old_label && upperbound > lowersx_ci)
            {
                numDistances++;
                feature_cnt += d;
                thresh = euclidean_dist_square(data[i], centroids[old_label], point_normSquares[i].total_normSquare,
                                               centroid_normSquares[old_label].total_normSquare);
                lowersx_li = std::sqrt(thresh);
                lower_bounds[i][old_label] = lowersx_li;
                timestamp[i][old_label] = iterations - 1;
                if (lowersx_li > lowersx_ci)
                {
                    numDistances++;
                    point_coord_type adist = dist_comp(data[i], centroids[ci], point_normSquares[i],
                                                       centroid_normSquares[ci], thresh, pca_dim, feature_cnt);
                    lowersx_ci = std::sqrt(adist);
                    lower_bounds[i][ci] = lowersx_ci;
                    timestamp[i][ci] = iterations - 1;
                    if (thresh > adist)
                    {
                        thresh = adist;
                        lowersx_li = lowersx_ci;
                        points[i].index = ci;
                    }
                }
                ci++;
                break;
            }
            ci++;
        }
        while (ci < k)
        {
            uint32_t timestamp_ci = timestamp[i][ci];
            point_coord_type lowersx_ci = timestamp_ci < div_ns.size() ? lower_bounds[i][ci] - div_ns[timestamp_ci][ci]
                                                                       : lower_bounds[i][ci];
            if (lowersx_li > lowersx_ci)
            {
                numDistances++;
                point_coord_type adist = dist_comp(data[i], centroids[ci], point_normSquares[i],
                                                   centroid_normSquares[ci], thresh, pca_dim, feature_cnt);
                lowersx_ci = std::sqrt(adist);
                lower_bounds[i][ci] = lowersx_ci;
                timestamp[i][ci] = iterations - 1;
                if (thresh > adist)
                {
                    thresh = adist;
                    lowersx_li = lowersx_ci;
                    points[i].index = ci;
                }
            }
            ci++;
        }

        if (old_label != points[i].index)
        {
            converged = false;
            for (size_t j = 0; j < d; j++)
            {
                sums[old_label][j] -= data[i][j];
            }
            for (size_t j = 0; j < d; j++)
            {
                sums[points[i].index][j] += data[i][j];
            }
            centers[old_label].cluster_count--;
            centers[old_label].flag = 0;
            centers[points[i].index].cluster_count++;
            centers[points[i].index].flag = 0;
        }
    }
    return converged;
}

void UltraSimplifiedElkan::recalculateCentroids_ns_stepwise()
{
    // 保存旧的中心点
    centroids_history.push_back(centroids);

    auto &last_centroids = centroids_history.back();
    for (size_t i = 0; i < k; i++)
    {
        if (centers[i].flag == 0 && centers[i].cluster_count > 0)
        {
            point_coord_type scale = 1.0 / centers[i].cluster_count;
            for (size_t j = 0; j < d; j++)
            {
                centroids[i][j] = sums[i][j] * scale;
            }

            feature_cnt += d;
            numDistances++;
            point_coord_type temp = centroid_normSquares[i].total_normSquare;
            centroid_normSquares[i].total_normSquare = innerProduct(centroids[i],
                                                                    centroid_normSquares[i].rest_norm, pca_dim);
            temp += centroid_normSquares[i].total_normSquare - 2 * innerProduct(centroids[i], last_centroids[i], pca_dim);
            div_ns_[i] = temp > 0 ? std::sqrt(temp) : 0.0;
        }
        else
        {
            centroids[i] = last_centroids[i];
            div_ns_[i] = 0.0;
        }
    }
    div_ns.push_back(div_ns_);

    // 计算中心点移动距离 (Norm of sums)
    for (size_t i = 0; i < iterations - 1; i++)
    {
        for (size_t j = 0; j < k; ++j)
        {
            feature_cnt += d;
            numDistances++;
            div_ns[i][j] = euclidean_dist(centroids[j], centroids_history[i][j]);
        }
    }
}
