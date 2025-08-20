#include "exp.h"

ExponionKmeans::ExponionKmeans(size_t k)
    : k(k), iterations(0), numDistances(0), n(0), d(0)
{
    npartitions = static_cast<size_t>(std::floor(std::log2(static_cast<point_coord_type>(k - 1))));
}

ExponionKmeans::~ExponionKmeans()
{
}

void ExponionKmeans::setInitialCentroids(const Matrix<point_coord_type> &initial_centroids)
{
    // 初始化数据结构
    d = initial_centroids[0].size();

    centroids = initial_centroids;
    old_centroids = initial_centroids;
    partitionvalues_halvies.resize(k, std::vector<point_coord_type>(npartitions - 1));
    geometricindices.resize(k, std::vector<size_t>(k - 1));
    c_to_c.resize(k, std::vector<point_coord_type>(k, 0.0));
    geometricpairs_halvies.resize(k); // 先设置行数
    for (size_t i = 0; i < k; ++i)
    {
        geometricpairs_halvies[i].resize(k - 1);
    }
    // 重置状态
    iterations = 0;
    numDistances = 0;
}

void ExponionKmeans::updateBounds()
{
    // 计算每个中心点的最近邻居距离
    for (size_t i = 0; i < k; ++i)
    {
        c_to_c[i][i] = std::numeric_limits<point_coord_type>::max();
        for (size_t j = i + 1; j < k; ++j)
        {
            if (i == j)
                continue;
            numDistances++;
            point_coord_type dist = euclidean_dist(centroids[i], centroids[j], centroid_normSquares[i], centroid_normSquares[j]);
            c_to_c[i][j] = dist;
            c_to_c[j][i] = dist;
        }
        near[i] = 0.5 * (*std::min_element(c_to_c[i].begin(), c_to_c[i].end()));
    }
}

void ExponionKmeans::assignPoints(const Matrix<point_coord_type> &data)
{
    point_coord_type m;
    size_t insertion_index;
    size_t delta_part;
    size_t n_distances_to_calculate;
    size_t old_label;
    point_coord_type max_div = *std::max_element(div.begin(), div.end());
    for (size_t i = 0; i < data.size(); ++i)
    {
        old_label = labels[i];
        l_hamerly[i] -= max_div;
        u_elkan[i] += div[old_label];
        m = std::max(near[old_label], l_hamerly[i]);
        if (u_elkan[i] <= m)
        {
            continue;
        }

        // 计算到当前中心点的距离
        numDistances++;
        point_coord_type u2 = euclidean_dist(data[i], centroids[old_label], point_normSquares[i], centroid_normSquares[old_label]);
        u_elkan[i] = u2;

        if (u_elkan[i] > m)
        {
            insertion_index = get_insertion_index(u_elkan[i] + near[old_label], npartitions - 1, partitionvalues_halvies[old_label]);
            if (insertion_index == npartitions - 1)
            {
                point_coord_type l2 = std::numeric_limits<point_coord_type>::max();
                size_t closest_center = old_label;
                for (size_t j = 0; j < k; ++j)
                {
                    if (j == old_label)
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
                u_elkan[i] = u2;
                l_hamerly[i] = l2;
                labels[i] = closest_center;
            }
            else
            {
                delta_part = 1;
                n_distances_to_calculate = 0;
                for (size_t psi = 0; psi < insertion_index; psi++)
                {
                    delta_part *= 2;
                    n_distances_to_calculate += delta_part;
                }
                point_coord_type l2 = std::numeric_limits<point_coord_type>::max();
                size_t closest_center = labels[i];
                for (size_t it = 0; it < n_distances_to_calculate; it++)
                {
                    size_t idx = geometricindices[labels[i]][it];
                    numDistances++;
                    point_coord_type dist = euclidean_dist(data[i], centroids[idx], point_normSquares[i], centroid_normSquares[idx]);
                    if (dist < u2)
                    {
                        l2 = u2;
                        u2 = dist;
                        closest_center = idx;
                    }
                    else if (dist < l2)
                    {
                        l2 = dist;
                    }
                }
                u_elkan[i] = u2;
                l_hamerly[i] = l2;
                labels[i] = closest_center;
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

bool ExponionKmeans::recalculateCentroids()
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

size_t ExponionKmeans::get_insertion_index(point_coord_type vin, size_t n_sortedv, std::vector<point_coord_type> &sortedv)
{
    size_t insertion_index = 0;
    while (insertion_index < n_sortedv)
    {
        if (vin > sortedv[insertion_index])
        {
            ++insertion_index;
        }
        else
        {
            break;
        }
    }
    return insertion_index;
}

void ExponionKmeans::update_pairs_parts_indices_halvies()
{
    size_t delta_part;
    size_t ncm1 = k - 1;
    for (size_t r = 0; r < k; r++)
    {
        size_t idx = 0;
        for (size_t c = 0; c < k; c++)
        {
            if (c != r)
            {
                geometricpairs_halvies[r][idx].first = 0.5 * c_to_c[r][c];
                geometricpairs_halvies[r][idx++].second = c;
            }
        }

        size_t middleindex = k - 1;
        size_t endindex;

        for (int pi = npartitions - 1; pi > 0; --pi)
        {
            endindex = middleindex;
            middleindex = std::pow(2, pi) - 1;
            std::nth_element(geometricpairs_halvies[r].begin(), geometricpairs_halvies[r].begin() + middleindex,
                             geometricpairs_halvies[r].begin() + endindex, std::less<std::pair<point_coord_type, int>>());
        }

        delta_part = 1;
        for (size_t i = 0; i < npartitions - 1; ++i)
        {
            delta_part *= 2;
            partitionvalues_halvies[r][i] = geometricpairs_halvies[r][delta_part - 2].first;
        }
        for (size_t c = 0; c < ncm1; ++c)
        {
            geometricindices[r][c] = geometricpairs_halvies[r][c].second;
        }
    }
}

void ExponionKmeans::fit(const Matrix<point_coord_type> &data)
{
    if (data.empty())
    {
        throw std::runtime_error("The dataset is empty.");
    }

    if (centroids.empty())
    {
        throw std::runtime_error("No initial centroids have been set.");
    }

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
            update_pairs_parts_indices_halvies();
        }

        iterations++;
    }
}

void ExponionKmeans::fit_ns(const Matrix<point_coord_type> &data)
{
    n = data.size();
    l_hamerly.resize(n, 0);
    u_elkan.resize(n, std::numeric_limits<point_coord_type>::max());
    near.resize(k, 0);
    div.resize(k, 0);
    cluster_count.resize(k, 0);
    labels.resize(n, 0);

    point_normSquares.resize(n);
    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    cumabs.resize(period + 1, std::vector<point_coord_type>(k, 0.0));
    max_deltaC_since.resize(period + 1, 0.0);
    tau_lower.resize(n, 0.0);

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
        assignPoints_ns(data);
        converged = recalculateCentroids();

        if (!converged)
        {
            updateBounds();
            update_pairs_parts_indices_halvies();

            for (size_t i = 0; i < 1 + iterations % period; i++)
            {
                for (size_t j = 0; j < k; j++)
                {
                    cumabs[i][j] += div[j];
                }
                max_deltaC_since[i] = *std::max_element(cumabs[i].begin(), cumabs[i].end());
            }

            if ((iterations + 1) % period == 0)
            {
                for (size_t i = 0; i < n; i++)
                {
                    l_hamerly[i] -= max_deltaC_since[tau_lower[i]];
                    tau_lower[i] = 0;
                }
                for (auto &row : cumabs)
                    std::fill(row.begin(), row.end(), 0.0);

                std::fill(max_deltaC_since.begin(), max_deltaC_since.end(), 0.0);
            }
        }

        iterations++;
    }
}

void ExponionKmeans::assignPoints_ns(const Matrix<point_coord_type> &data)
{
    for (size_t i = 0; i < data.size(); ++i)
    {
        size_t old_label = labels[i];
        point_coord_type lower = l_hamerly[i] - max_deltaC_since[tau_lower[i]];
        u_elkan[i] += div[old_label];
        point_coord_type m = std::max(near[old_label], lower);
        if (u_elkan[i] <= m)
        {
            continue;
        }

        // 计算到当前中心点的距离
        numDistances++;
        point_coord_type u2 = euclidean_dist(data[i], centroids[old_label], point_normSquares[i], centroid_normSquares[old_label]);
        u_elkan[i] = u2;

        if (u_elkan[i] > m)
        {
            size_t insertion_index = get_insertion_index(u_elkan[i] + near[old_label], npartitions - 1, partitionvalues_halvies[old_label]);
            tau_lower[i] = iterations % period;
            if (insertion_index == npartitions - 1)
            {
                point_coord_type l2 = std::numeric_limits<point_coord_type>::max();
                size_t closest_center = old_label;
                for (size_t j = 0; j < k; ++j)
                {
                    if (j == old_label)
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
                u_elkan[i] = u2;
                l_hamerly[i] = l2;
                labels[i] = closest_center;
            }
            else
            {
                size_t delta_part = 1;
                size_t n_distances_to_calculate = 0;
                for (size_t psi = 0; psi < insertion_index; psi++)
                {
                    delta_part *= 2;
                    n_distances_to_calculate += delta_part;
                }
                point_coord_type l2 = std::numeric_limits<point_coord_type>::max();
                size_t closest_center = labels[i];
                for (size_t it = 0; it < n_distances_to_calculate; it++)
                {
                    size_t idx = geometricindices[labels[i]][it];
                    numDistances++;
                    point_coord_type dist = euclidean_dist(data[i], centroids[idx], point_normSquares[i], centroid_normSquares[idx]);
                    if (dist < u2)
                    {
                        l2 = u2;
                        u2 = dist;
                        closest_center = idx;
                    }
                    else if (dist < l2)
                    {
                        l2 = dist;
                    }
                }
                u_elkan[i] = u2;
                l_hamerly[i] = l2;
                labels[i] = closest_center;
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
