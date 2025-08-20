#include "bv_kmeans_v1.h"
#include "kmeans.h"

BVKmeans::BVKmeans(size_t k)
    : k(k), iterations(0), numDistances(0), n(0), d(0), ngroups{1 + (k - 1) / 10}
{
}

BVKmeans::~BVKmeans()
{
}

void BVKmeans::setInitialCentroids(point_coord_type perc, const Matrix<point_coord_type> &initial_centroids)
{
    size_t dim = initial_centroids[0].size();
    block_size = std::max<size_t>(1, static_cast<size_t>(dim * perc));
    num_block = dim / block_size;
    if (dim % block_size != 0)
    {
        num_block++;
    }

    centroids = initial_centroids;
    old_centroids = initial_centroids;
    numDistances = 0;
}

void BVKmeans::fit(const Matrix<point_coord_type> &data)
{
    init(data);

    bool converged = false;
    while (!converged)
    {
        converged = assignPoints(data);
        recalculateCentroids();
        iterations++;
    }
}

void BVKmeans::init(const Matrix<point_coord_type> &data)
{
    n = data.size();
    d = data[0].size();
    feature_cnt = 0;

    points.resize(n, Point{std::numeric_limits<point_coord_type>::max(), 0});
    point_normSquares.resize(n);
    bv_data.resize(n, std::vector<point_coord_type>(num_block, 0.0));
    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    centers.resize(k, Center{0, 0});
    for (size_t i = 0; i < n; i++)
    {
        size_t ind = points[i].index;
        centers[ind].cluster_count++;
        for (size_t j = 0; j < d; j++)
        {
            sums[ind][j] += data[i][j];
        }
        point_coord_type temp = innerProduct(data[i]);
        point_normSquares[i] = temp;
        size_t idx = 0;
        size_t jt = 0;
        while (idx < d)
        {
            size_t sz = std::min(block_size, d - idx);

            temp = 0.0;
            for (size_t j = 0; j < sz; j++)
            {
                temp += data[i][idx] * data[i][idx];
                idx++;
            }
            bv_data[i][jt++] = std::sqrt(temp);
        }
    }
    // 初始化
    group_lowers.resize(n, std::vector<point_coord_type>(ngroups, 0.0));
    groupparts.resize(ngroups + 1, 0);
    group.resize(n, 0);
    div_center.resize(k, 0.0);
    div_group.resize(ngroups, 0.0);

    // 初始化组划分
    KMeans kmeans(ngroups);
    Matrix<point_coord_type> initial_centroids(centroids.begin(), centroids.begin() + ngroups);
    kmeans.setInitialCentroids(initial_centroids);
    kmeans.fit(centroids);
    const std::vector<size_t> &L = kmeans.getLabels();
    numDistances += kmeans.getNumDistances();
    for (size_t i = 0; i < k; i++)
    {
        groupparts[L[i] + 1]++;
    }
    for (size_t i = 0; i < ngroups; i++)
    {
        groupparts[i + 1] += groupparts[i];
    }

    std::vector<size_t> indices(k);
    std::iota(indices.begin(), indices.end(), 0); // [0, 1, 2, ..., N-1]
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b)
              { return L[a] < L[b]; });

    Matrix<point_coord_type> sorted_centroids;
    sorted_centroids.reserve(centroids.size());

    centroid_normSquares.reserve(k);
    for (size_t idx : indices)
    {
        sorted_centroids.push_back(centroids[idx]);
        centroid_normSquares.push_back(innerProduct(centroids[idx]));
    }
    std::swap(sorted_centroids, centroids);
    bv_centroids.resize(k, std::vector<point_coord_type>(num_block, 0.0));
    for (size_t i = 0; i < k; i++)
    {
        size_t idx = 0;
        size_t jt = 0;
        while (idx < d)
        {
            size_t sz = std::min(block_size, d - idx);

            point_coord_type temp = 0.0;
            for (size_t j = 0; j < sz; j++)
            {
                temp += centroids[i][idx] * centroids[i][idx];
                idx++;
            }
            bv_centroids[i][jt++] = std::sqrt(temp);
        }
    }
}

bool BVKmeans::assignPoints(const Matrix<point_coord_type> &data)
{
    for (size_t i = 0; i < k; i++)
    {
        centers[i].flag = 1;
    }
    std::vector<point_coord_type> glowers_previous(ngroups);
    point_coord_type adist;
    point_coord_type group_nearest, group_second_nearest;
    size_t group_nearest_index;
    point_coord_type globallower;
    bool converged = true;
    for (size_t i = 0; i < n; ++i)
    {
        size_t old_label = points[i].index;
        globallower = std::numeric_limits<point_coord_type>::max();
        for (size_t it = 0; it < ngroups; it++)
        {
            glowers_previous[it] = group_lowers[i][it];
            group_lowers[i][it] -= div_group[it];
            globallower = std::min(globallower, group_lowers[i][it]);
        }
        points[i].distance += div_center[old_label];
        if (globallower < points[i].distance)
        {
            numDistances++;
            feature_cnt += d;
            point_coord_type thresh = euclidean_dist_square(data[i], centroids[old_label], point_normSquares[i], centroid_normSquares[old_label]);
            points[i].distance = std::sqrt(thresh);
            if (globallower < points[i].distance)
            {
                for (size_t gi = 0; gi < ngroups; gi++)
                {
                    if (group_lowers[i][gi] < points[i].distance)
                    {
                        group_nearest = std::numeric_limits<point_coord_type>::max();
                        group_second_nearest = std::numeric_limits<point_coord_type>::max();
                        group_nearest_index = 0;
                        for (size_t ci = groupparts[gi]; ci < groupparts[gi + 1]; ci++)
                        {
                            if (old_label == ci || glowers_previous[gi] - div_center[ci] < group_second_nearest)
                            {
                                numDistances++;
                                adist = std::sqrt(distance_bv(i, ci, thresh, data));
                                if (adist < group_nearest)
                                {
                                    group_second_nearest = group_nearest;
                                    group_nearest = adist;
                                    group_nearest_index = ci;
                                }
                                else if (adist < group_second_nearest)
                                {
                                    group_second_nearest = adist;
                                }
                            }
                        }
                        if (group[i] != gi)
                        {
                            if (group_nearest < points[i].distance)
                            {
                                if (gi < group[i])
                                {
                                    group_lowers[i][group[i]] = std::min(points[i].distance, group_lowers[i][group[i]]);
                                }
                                else
                                {
                                    group_lowers[i][group[i]] = points[i].distance;
                                }
                                group_lowers[i][gi] = group_second_nearest;
                                thresh = group_nearest * group_nearest;
                                group[i] = gi;
                                points[i].index = group_nearest_index;
                                points[i].distance = group_nearest;
                            }
                            else
                            {
                                group_lowers[i][gi] = group_nearest;
                            }
                        }
                        else
                        {
                            group_lowers[i][gi] = group_second_nearest;
                            thresh = group_nearest * group_nearest;
                            points[i].distance = group_nearest;
                            points[i].index = group_nearest_index;
                        }
                    }
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
        }
    }
    return converged;
}

void BVKmeans::recalculateCentroids()
{
    std::swap(old_centroids, centroids);

    for (size_t i = 0; i < k; ++i)
    {
        if (centers[i].flag == 0 && centers[i].cluster_count > 0)
        {
            point_coord_type scale = 1.0 / centers[i].cluster_count;
            point_coord_type sum = 0.0;
            for (size_t j = 0; j < d; ++j)
            {
                centroids[i][j] = sums[i][j] * scale;
                sum += centroids[i][j] * centroids[i][j];
            }
            numDistances++;
            feature_cnt += d;
            div_center[i] = euclidean_dist(centroids[i], old_centroids[i], centroid_normSquares[i], sum);
            centroid_normSquares[i] = sum;
            size_t idx = 0;
            size_t jt = 0;
            while (idx < d)
            {
                size_t sz = std::min(block_size, d - idx);
                sum = 0.0;
                for (size_t j = 0; j < sz; j++)
                {
                    sum += centroids[i][idx] * centroids[i][idx];
                    idx++;
                }
                bv_centroids[i][jt++] = std::sqrt(sum);
            }
        }
        else
        {
            centroids[i] = old_centroids[i];
            div_center[i] = 0;
        }
    }

    for (size_t gi = 0; gi < ngroups; ++gi)
    {
        div_group[gi] = div_center[groupparts[gi]];
        for (size_t ci = groupparts[gi] + 1; ci < groupparts[gi + 1]; ci++)
        {
            if (div_center[ci] > div_group[gi])
            {
                div_group[gi] = div_center[ci];
            }
        }
    }
}

point_coord_type BVKmeans::distance_bv(size_t x, size_t y, point_coord_type &thresh, const Matrix<point_coord_type> &data)
{
    point_coord_type temp = point_normSquares[x] + centroid_normSquares[y];
    point_coord_type dist = temp - 2 * std::sqrt(point_normSquares[x] * centroid_normSquares[y]);
    if (dist > thresh)
    {
        return dist;
    }
    dist = 0.0;
    for (size_t i = 0; i < num_block; i++)
    {
        feature_cnt++;
        dist += bv_data[x][i] * bv_centroids[y][i];
    }
    dist = temp - 2 * dist;
    if (dist > thresh)
    {
        return dist;
    }
    dist = 0.0;
    for (size_t i = 0; i < d; i++)
    {
        feature_cnt++;
        dist += data[x][i] * centroids[y][i];
    }
    dist = temp - 2 * dist;
    return dist > 0 ? dist : 0.0;
}
