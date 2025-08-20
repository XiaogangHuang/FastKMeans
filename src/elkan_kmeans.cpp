#include "elkan_kmeans.h"

ElkanKmeans::ElkanKmeans(size_t k)
    : k(k), iterations(0), numDistances(0), n(0), d(0)
{
}

ElkanKmeans::~ElkanKmeans()
{
    clear();
}

void ElkanKmeans::clear()
{
    l_elkan.clear();
    u_elkan.clear();
    near.clear();
    div.clear();
    c_to_c.clear();
    labels.clear();
    cluster_count.clear();
    centroids.clear();
    old_centroids.clear();
}

void ElkanKmeans::init(const Matrix<point_coord_type> &data)
{
    n = data.size();
    d = data[0].size();

    // 初始化数据结构
    l_elkan.resize(n, std::vector<point_coord_type>(k, 0.0));
    u_elkan.resize(n, std::numeric_limits<point_coord_type>::max());
    near.resize(k, 0);
    div.resize(k, 0);
    c_to_c.resize(k, std::vector<point_coord_type>(k, 0.0));
    cluster_count.resize(k, 0);
    labels.resize(n, 0);

    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    point_normSquares.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        size_t ind = labels[i];
        cluster_count[ind]++;
        for (size_t j = 0; j < d; j++)
        {
            sums[ind][j] += data[i][j];
        }
        point_normSquares[i] = innerProduct(data[i]);
    }
    centroid_normSquares.resize(k);
    for (size_t i = 0; i < k; i++)
    {
        centroid_normSquares[i] = innerProduct(centroids[i]);
    }
}

void ElkanKmeans::setInitialCentroids(const Matrix<point_coord_type> &initial_centroids, size_t dim)
{
    pca_dim = dim;
    centroids = initial_centroids;
    old_centroids = initial_centroids;

    iterations = 0;
    numDistances = 0;
}

void ElkanKmeans::updateBounds()
{
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            l_elkan[i][j] -= div[j];
        }
    }

    // 计算中心点之间的距离
    for (size_t i = 0; i < k; ++i)
    {
        c_to_c[i][i] = 0;

        for (size_t j = 0; j < k; ++j)
        {
            numDistances++;
            point_coord_type dist = euclidean_dist(centroids[i], centroids[j], centroid_normSquares[i], centroid_normSquares[j]);
            c_to_c[i][j] = dist;
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
}

void ElkanKmeans::assignPoints(const Matrix<point_coord_type> &data)
{
    for (size_t i = 0; i < n; ++i)
    {
        size_t old_label = labels[i];
        point_coord_type val = near[old_label];
        u_elkan[i] += div[old_label];
        if (u_elkan[i] > val)
        {
            size_t ci = 0;
            while (ci < k)
            {
                if (ci != old_label && u_elkan[i] > l_elkan[i][ci] && u_elkan[i] > 0.5 * c_to_c[old_label][ci])
                {
                    numDistances++;
                    u_elkan[i] = euclidean_dist(data[i], centroids[old_label], point_normSquares[i], centroid_normSquares[old_label]);
                    l_elkan[i][old_label] = u_elkan[i];
                    if (u_elkan[i] > l_elkan[i][ci] && u_elkan[i] > 0.5 * c_to_c[old_label][ci])
                    {
                        numDistances++;
                        l_elkan[i][ci] = euclidean_dist(data[i], centroids[ci], point_normSquares[i], centroid_normSquares[ci]);
                        if (u_elkan[i] > l_elkan[i][ci])
                        {
                            u_elkan[i] = l_elkan[i][ci];
                            labels[i] = ci;
                        }
                    }
                    ci++;
                    break;
                }
                ci++;
            }
            while (ci < k)
            {
                if (u_elkan[i] > l_elkan[i][ci] && u_elkan[i] > 0.5 * c_to_c[labels[i]][ci])
                {
                    numDistances++;
                    l_elkan[i][ci] = euclidean_dist(data[i], centroids[ci], point_normSquares[i], centroid_normSquares[ci]);
                    if (u_elkan[i] > l_elkan[i][ci])
                    {
                        u_elkan[i] = l_elkan[i][ci];
                        labels[i] = ci;
                    }
                }
                ci++;
            }
            if (old_label != labels[i])
            {
                for (size_t j = 0; j < d; j++)
                {
                    sums[old_label][j] -= data[i][j];
                }
                cluster_count[old_label]--;

                for (size_t j = 0; j < d; j++)
                {
                    sums[labels[i]][j] += data[i][j];
                }
                cluster_count[labels[i]]++;
            }
        }
    }
}

bool ElkanKmeans::recalculateCentroids()
{
    bool converged = true;

    // 保存旧的中心点
    std::swap(centroids, old_centroids);

    for (size_t i = 0; i < k; i++)
    {
        if (cluster_count[i] > 0)
        {
            point_coord_type sum = 0;
            for (size_t j = 0; j < d; j++)
            {
                centroids[i][j] = sums[i][j] / cluster_count[i];
                sum += centroids[i][j] * centroids[i][j];
            }
            numDistances++;
            div[i] = euclidean_dist(centroids[i], old_centroids[i], centroid_normSquares[i], sum);
            centroid_normSquares[i] = sum;
            if (div[i] > 0)
            {
                converged = false;
            }
        }
        else
        {
            centroids[i] = old_centroids[i];
            div[i] = 0.0;
        }
    }

    return converged;
}

void ElkanKmeans::fit(const Matrix<point_coord_type> &data)
{
    init(data);

    bool converged = false;
    while (!converged)
    {
        assignPoints(data);
        converged = recalculateCentroids();
        if (!converged)
        {
            updateBounds();
        }
        iterations++;
    }
}

void ElkanKmeans::assignPoints_stepwise(const Matrix<point_coord_type> &data)
{
    for (size_t i = 0; i < n; ++i)
    {
        size_t old_label = labels[i];
        point_coord_type val = near[old_label], thresh;
        u_elkan[i] += div[old_label];
        if (u_elkan[i] > val)
        {
            size_t ci = 0;
            while (ci < k)
            {
                if (ci != old_label && u_elkan[i] > l_elkan[i][ci] && u_elkan[i] > 0.5 * c_to_c[old_label][ci])
                {
                    numDistances++;
                    thresh = euclidean_dist_square(data[i], centroids[old_label], point_normSquares[i],
                                                   centroid_normSquares[old_label]);
                    u_elkan[i] = std::sqrt(thresh);
                    l_elkan[i][old_label] = u_elkan[i];
                    if (u_elkan[i] > l_elkan[i][ci] && u_elkan[i] > 0.5 * c_to_c[old_label][ci])
                    {
                        numDistances++;
                        point_coord_type adist = dist_comp(data[i], centroids[ci], thresh, pca_dim, feature_cnt);
                        l_elkan[i][ci] = std::sqrt(adist);
                        if (thresh > adist)
                        {
                            thresh = adist;
                            u_elkan[i] = l_elkan[i][ci];
                            labels[i] = ci;
                        }
                    }
                    ci++;
                    break;
                }
                ci++;
            }
            while (ci < k)
            {
                if (u_elkan[i] > l_elkan[i][ci] && u_elkan[i] > 0.5 * c_to_c[labels[i]][ci])
                {
                    numDistances++;
                    point_coord_type adist = dist_comp(data[i], centroids[ci], thresh, pca_dim, feature_cnt);
                    l_elkan[i][ci] = std::sqrt(adist);
                    if (thresh > adist)
                    {
                        thresh = adist;
                        u_elkan[i] = l_elkan[i][ci];
                        labels[i] = ci;
                    }
                }
                ci++;
            }
            if (old_label != labels[i])
            {
                for (size_t j = 0; j < d; j++)
                {
                    sums[old_label][j] -= data[i][j];
                }
                cluster_count[old_label]--;

                for (size_t j = 0; j < d; j++)
                {
                    sums[labels[i]][j] += data[i][j];
                }
                cluster_count[labels[i]]++;
            }
        }
    }
}

void ElkanKmeans::fit_stepwise(const Matrix<point_coord_type> &data)
{
    init(data);

    bool converged = false;
    while (!converged)
    {
        assignPoints_stepwise(data);
        converged = recalculateCentroids();
        if (!converged)
        {
            updateBounds();
        }
        iterations++;
    }
}

void ElkanKmeans::init_simplified(const Matrix<point_coord_type> &data)
{
    n = data.size();
    d = data[0].size();

    // 初始化数据结构
    l_elkan.resize(n, std::vector<point_coord_type>(k, 0.0));
    u_elkan.resize(n, std::numeric_limits<point_coord_type>::max());
    div.resize(k, 0);
    cluster_count.resize(k, 0);
    labels.resize(n, 0);

    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    point_normSquares.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        size_t ind = labels[i];
        cluster_count[ind]++;
        for (size_t j = 0; j < d; j++)
        {
            sums[ind][j] += data[i][j];
        }
        point_normSquares[i] = innerProduct(data[i]);
    }
    centroid_normSquares.resize(k);
    for (size_t i = 0; i < k; i++)
    {
        centroid_normSquares[i] = innerProduct(centroids[i]);
    }
}

void ElkanKmeans::fit_simplified(const Matrix<point_coord_type> &data)
{
    init_simplified(data);

    bool converged = false;
    while (!converged)
    {
        assignPoints_simplified(data);
        converged = recalculateCentroids();

        if (!converged)
        {
            updateBounds_simplified();
        }
        iterations++;
    }
}

void ElkanKmeans::assignPoints_simplified(const Matrix<point_coord_type> &data)
{
    for (size_t i = 0; i < n; ++i)
    {
        size_t old_label = labels[i];
        u_elkan[i] += div[old_label];
        size_t ci = 0;
        while (ci < k)
        {
            if (ci != labels[i] && u_elkan[i] > l_elkan[i][ci])
            {
                numDistances++;
                u_elkan[i] = euclidean_dist(data[i], centroids[labels[i]], point_normSquares[i], centroid_normSquares[labels[i]]);
                l_elkan[i][labels[i]] = u_elkan[i];
                if (u_elkan[i] > l_elkan[i][ci])
                {
                    numDistances++;
                    l_elkan[i][ci] = euclidean_dist(data[i], centroids[ci], point_normSquares[i], centroid_normSquares[ci]);
                    if (u_elkan[i] > l_elkan[i][ci])
                    {
                        u_elkan[i] = l_elkan[i][ci];
                        labels[i] = ci;
                    }
                }
                ci++;
                break;
            }
            ci++;
        }
        while (ci < k)
        {
            if (u_elkan[i] > l_elkan[i][ci])
            {
                numDistances++;
                l_elkan[i][ci] = euclidean_dist(data[i], centroids[ci], point_normSquares[i], centroid_normSquares[ci]);
                if (u_elkan[i] > l_elkan[i][ci])
                {
                    u_elkan[i] = l_elkan[i][ci];
                    labels[i] = ci;
                }
            }
            ci++;
        }

        if (old_label != labels[i])
        {
            for (size_t j = 0; j < d; j++)
            {
                sums[old_label][j] -= data[i][j];
            }
            cluster_count[old_label]--;

            for (size_t j = 0; j < d; j++)
            {
                sums[labels[i]][j] += data[i][j];
            }
            cluster_count[labels[i]]++;
        }
    }
}

void ElkanKmeans::updateBounds_simplified()
{
    // 更新所有点的下界
    for (size_t i = 0; i < n; ++i)
    {
        // 更新每个点的下界
        for (size_t j = 0; j < k; ++j)
        {
            point_coord_type val = l_elkan[i][j] - div[j];
            l_elkan[i][j] = std::max(val, point_coord_type(0));
        }
    }
}

void ElkanKmeans::init_ns(const Matrix<point_coord_type> &data)
{
    n = data.size();
    d = data[0].size();

    // 初始化
    labels.resize(n, 0);
    cluster_count.resize(k, 0);
    timestamp.resize(n, std::vector<size_t>(k, 0));
    l_elkan.resize(n, std::vector<point_coord_type>(k, 0.0));
    u_elkan.resize(n, std::numeric_limits<point_coord_type>::max());
    div.resize(k, 0.0);

    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    point_normSquares.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        size_t ind = labels[i];
        cluster_count[ind]++;
        for (size_t j = 0; j < d; j++)
        {
            sums[ind][j] += data[i][j];
        }
        point_normSquares[i] = innerProduct(data[i]);
    }
    centroid_normSquares.resize(k);
    for (size_t i = 0; i < k; i++)
    {
        centroid_normSquares[i] = innerProduct(centroids[i]);
    }
}

void ElkanKmeans::fit_ns(const Matrix<point_coord_type> &data)
{
    init_ns(data);
    bool converged = false;
    while (!converged)
    {
        converged = assignPoints_ns(data);
        if (!converged)
        {
            recalculateCentroids_ns();
        }

        iterations++;
    }
}

bool ElkanKmeans::assignPoints_ns(const Matrix<point_coord_type> &data)
{
    bool converged = true;
    for (size_t i = 0; i < n; ++i)
    {
        size_t old_label = labels[i];
        point_coord_type upperbound = iterations > 0 ? u_elkan[i] + div_ns[timestamp[i][old_label]][old_label] : u_elkan[i];

        size_t ci = 0;
        while (ci < k)
        {
            point_coord_type lowersx_ci = iterations > 0 ? l_elkan[i][ci] - div_ns[timestamp[i][ci]][ci] : l_elkan[i][ci];
            if (ci != labels[i] && upperbound > lowersx_ci)
            {
                numDistances++;
                point_coord_type lowersx_li = euclidean_dist(data[i], centroids[labels[i]], point_normSquares[i], centroid_normSquares[labels[i]]);
                l_elkan[i][labels[i]] = lowersx_li;
                timestamp[i][labels[i]] = iterations;
                if (lowersx_li > lowersx_ci)
                {
                    numDistances++;
                    lowersx_ci = euclidean_dist(data[i], centroids[ci], point_normSquares[i], centroid_normSquares[ci]);
                    l_elkan[i][ci] = lowersx_ci;
                    timestamp[i][ci] = iterations;
                    if (lowersx_li > lowersx_ci)
                    {
                        lowersx_li = lowersx_ci;
                        labels[i] = ci;
                    }
                }
                u_elkan[i] = lowersx_li;
                ci++;
                break;
            }
            ci++;
        }
        while (ci < k)
        {
            point_coord_type lowersx_ci = timestamp[i][ci] < div_ns.size() ? l_elkan[i][ci] - div_ns[timestamp[i][ci]][ci] : l_elkan[i][ci];

            if (u_elkan[i] > lowersx_ci)
            {
                numDistances++;
                lowersx_ci = euclidean_dist(data[i], centroids[ci], point_normSquares[i], centroid_normSquares[ci]);
                l_elkan[i][ci] = lowersx_ci;
                timestamp[i][ci] = iterations;
                if (u_elkan[i] > lowersx_ci)
                {
                    u_elkan[i] = lowersx_ci;
                    labels[i] = ci;
                }
            }
            ci++;
        }

        if (old_label != labels[i])
        {
            converged = false;
            for (size_t j = 0; j < d; j++)
            {
                sums[old_label][j] -= data[i][j];
            }
            cluster_count[old_label]--;

            for (size_t j = 0; j < d; j++)
            {
                sums[labels[i]][j] += data[i][j];
            }
            cluster_count[labels[i]]++;
        }
    }
    return converged;
}

void ElkanKmeans::recalculateCentroids_ns()
{
    // 保存旧的中心点
    centroids_history.push_back(centroids);

    for (size_t i = 0; i < k; i++)
    {
        if (cluster_count[i] > 0)
        {
            point_coord_type sum = 0;
            for (size_t j = 0; j < d; j++)
            {
                centroids[i][j] = sums[i][j] / cluster_count[i];
                sum += centroids[i][j] * centroids[i][j];
            }
            numDistances++;
            div[i] = euclidean_dist(centroids[i], centroids_history.back()[i], sum, centroid_normSquares[i]);
            centroid_normSquares[i] = sum;
        }
        else
        {
            centroids[i] = centroids_history.back()[i];
        }
    }

    // 计算中心点移动距离 (Norm of sums)
    for (size_t i = 0; i < iterations; i++)
    {
        for (size_t j = 0; j < k; ++j)
        {
            numDistances++;
            div_ns[i][j] = euclidean_dist(centroids[j], centroids_history[i][j]);
        }
    }

    div_ns.push_back(div);
}