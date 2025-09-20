#include "../include/hamerly_kmeans.h"

HamerlyKmeans::HamerlyKmeans(size_t k)
    : k(k), iterations(0), numDistances(0), n(0), d(0)
{
    if (k == 0)
    {
        throw std::runtime_error("The number of clusters must be greater than0");
    }
}

HamerlyKmeans::~HamerlyKmeans()
{
    clear();
}

void HamerlyKmeans::clear()
{
    Matrix<point_coord_type>().swap(centroids);
    Matrix<point_coord_type>().swap(old_centroids);
    std::vector<point_coord_type>().swap(l_hamerly);
    std::vector<point_coord_type>().swap(u_elkan);
    std::vector<point_coord_type>().swap(near);
    std::vector<point_coord_type>().swap(div);
    std::vector<size_t>().swap(labels);
    std::vector<size_t>().swap(cluster_count);
}

void HamerlyKmeans::setInitialCentroids(const Matrix<point_coord_type> &initial_centroids)
{
    if (initial_centroids.size() != k)
    {
        throw std::runtime_error("The number of initial centroids does not match the number of clusters.");
    }

    // 初始化数据结构
    d = initial_centroids[0].size();

    centroids = initial_centroids;
    old_centroids = initial_centroids;

    // 重置状态
    iterations = 0;
    numDistances = 0;
}

void HamerlyKmeans::updateBounds()
{
    // 更新所有点的边界
    for (size_t i = 0; i < n; ++i)
    {
        point_coord_type max_dist = 0.0;

        // 找到第二近的中心点
        for (size_t j = 0; j < k; ++j)
        {
            if ((labels[i] != j) && (div[j] > max_dist))
            {
                max_dist = div[j];
            }
        }

        // 更新上界和下界
        u_elkan[i] += div[labels[i]];
        l_hamerly[i] -= max_dist;
    }

    // 计算每个中心点的最近邻居距离
    for (size_t i = 0; i < k; ++i)
    {
        point_coord_type smallest = std::numeric_limits<point_coord_type>::max();
        for (size_t j = 0; j < k; ++j)
        {
            if (i == j)
                continue;
            numDistances++;
            point_coord_type dist = euclidean_dist(centroids[i], centroids[j], centroid_normSquares[i], centroid_normSquares[j]);
            if (dist < smallest)
            {
                smallest = dist;
                near[i] = 0.5 * smallest;
            }
        }
    }
}

void HamerlyKmeans::assignPoints(const Matrix<point_coord_type> &data)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        point_coord_type val = std::max(near[labels[i]], l_hamerly[i]);
        if (u_elkan[i] <= val)
            continue;

        // 计算到当前中心点的距离
        numDistances++;
        point_coord_type u2 = euclidean_dist(data[i], centroids[labels[i]], point_normSquares[i], centroid_normSquares[labels[i]]);
        u_elkan[i] = u2;

        if (u_elkan[i] > val)
        {
            point_coord_type l2 = std::numeric_limits<point_coord_type>::max();
            size_t closest_center = labels[i];
            for (size_t j = 0; j < k; ++j)
            {
                if (j == labels[i])
                    continue;
                numDistances++;
                point_coord_type dist = euclidean_dist(data[i], centroids[j], point_normSquares[i], centroid_normSquares[j]);
                if (dist < u2)
                {
                    l2 = u2;
                    u2 = dist;
                    closest_center = j;
                }
                else if (dist < l2)
                {
                    l2 = dist;
                }
            }
            l_hamerly[i] = l2;
            if (closest_center != labels[i])
            {
                size_t old_label = labels[i];
                for (size_t j = 0; j < d; j++)
                {
                    sums[old_label][j] -= data[i][j];
                }
                cluster_count[old_label]--;

                for (size_t j = 0; j < d; j++)
                {
                    sums[closest_center][j] += data[i][j];
                }
                cluster_count[closest_center]++;

                labels[i] = closest_center;
                u_elkan[i] = u2;
            }
        }
    }
}

bool HamerlyKmeans::recalculateCentroids()
{
    bool converged = true;

    // 保存旧的中心点
    std::swap(centroids, old_centroids);

    for (size_t i = 0; i < k; ++i)
    {
        if (cluster_count[i] > 0)
        {
            point_coord_type scale = 1.0 / cluster_count[i];
            point_coord_type sum = 0;
            for (size_t j = 0; j < d; ++j)
            {
                centroids[i][j] = sums[i][j] * scale;
                sum += centroids[i][j] * centroids[i][j];
            }
            numDistances++;
            div[i] = euclidean_dist(centroids[i], old_centroids[i], sum, centroid_normSquares[i]);
            centroid_normSquares[i] = sum;
            if (div[i] > 0)
            {
                converged = false;
            }
        }
        else
        {
            centroids[i] = old_centroids[i];
            div[i] = 0;
        }
    }

    return converged;
}

void HamerlyKmeans::fit(const Matrix<point_coord_type> &data)
{
    // 更新数据集大小
    n = data.size();

    // 初始化数据结构
    l_hamerly.resize(n, 0);
    u_elkan.resize(n, std::numeric_limits<point_coord_type>::max());
    near.resize(k, 0);
    div.resize(k, 0);
    cluster_count.resize(k, 0);
    labels.resize(n, 0);

    point_normSquares.resize(n);
    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
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