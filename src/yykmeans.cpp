#include "yykmeans.h"

YinYangKmeans::YinYangKmeans(size_t k) : k(k), ngroups{1 + (k - 1) / 10}, iterations(0), numDistances(0), n(0), d(0),
                                         assign_time{0}, update_time{0} {}

YinYangKmeans::~YinYangKmeans() {}

std::vector<size_t> YinYangKmeans::getLabels()
{
    std::vector<size_t> labels(n, 0);
    for (size_t i = 0; i < n; ++i)
    {
        labels[i] = points[i].index;
    }
    return labels;
}

void YinYangKmeans::setInitialCentroids(const Matrix<point_coord_type> &initial_centroids)
{
    centroids = initial_centroids;
    old_centroids = initial_centroids;
    d = centroids[0].size();
}

void YinYangKmeans::fit(const Matrix<point_coord_type> &data)
{
    auto start = std::chrono::high_resolution_clock::now();
    init(data);
    auto end = std::chrono::high_resolution_clock::now();
    assign_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    bool converged = false;
    while (!converged)
    {
        auto assign_start = std::chrono::high_resolution_clock::now();
        converged = assignPoints(data);
        end = std::chrono::high_resolution_clock::now();
        assign_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - assign_start);
        auto update_start = std::chrono::high_resolution_clock::now();
        recalculateCentroids();
        iterations++;
        end = std::chrono::high_resolution_clock::now();
        update_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - update_start);
    }
}

void YinYangKmeans::init(const Matrix<point_coord_type> &data)
{
    n = data.size();
    d = data[0].size();
    feature_cnt = 0;

    // 初始化
    group_lowers.resize(n, std::vector<point_coord_type>(ngroups, 0.0));
    groupparts.resize(ngroups + 1, 0);

    points.resize(n, Point{std::numeric_limits<point_coord_type>::max(), 0});
    group.resize(n, 0);
    centers.resize(k, Center{0, 0});
    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    point_normSquares.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        size_t ind = points[i].index;
        centers[ind].cluster_count++;
        for (size_t j = 0; j < d; j++)
        {
            sums[ind][j] += data[i][j];
        }
        point_normSquares[i] = innerProduct(data[i]);
    }

    div_center.resize(k, 0.0);
    div_group.resize(ngroups, 0.0);

    // 初始化组划分
    KMeans kmeans(ngroups);
    Matrix<point_coord_type> initial_centroids(centroids.begin(), centroids.begin() + ngroups);
    kmeans.setInitialCentroids(initial_centroids);
    kmeans.fit(centroids);
    const std::vector<size_t> &L = kmeans.getLabels();
    numDistances += kmeans.getNumDistances();
    feature_cnt += numDistances * d;
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
    centroids = std::move(sorted_centroids);
}

bool YinYangKmeans::assignPoints(const Matrix<point_coord_type> &data)
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
            points[i].distance = euclidean_dist(data[i], centroids[old_label], point_normSquares[i], centroid_normSquares[old_label]);
            feature_cnt += d;
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
                                adist = euclidean_dist(data[i], centroids[ci], point_normSquares[i], centroid_normSquares[ci]);
                                feature_cnt += d;
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

void YinYangKmeans::recalculateCentroids()
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
            div_center[i] = euclidean_dist(centroids[i], old_centroids[i], centroid_normSquares[i], sum);
            feature_cnt += d;
            centroid_normSquares[i] = sum;
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

void YinYangKmeans::fit_stepwise(const Matrix<point_coord_type> &data, size_t &dim)
{
    pca_dim = dim;
    auto start = std::chrono::high_resolution_clock::now();
    init(data);
    auto end = std::chrono::high_resolution_clock::now();
    assign_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    bool converged = false;
    while (!converged)
    {
        auto assign_start = std::chrono::high_resolution_clock::now();
        converged = assignPoints_stepwise(data);
        end = std::chrono::high_resolution_clock::now();
        assign_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - assign_start);

        auto update_start = std::chrono::high_resolution_clock::now();
        recalculateCentroids();
        iterations++;
        end = std::chrono::high_resolution_clock::now();
        update_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - update_start);
    }
}

bool YinYangKmeans::assignPoints_stepwise(const Matrix<point_coord_type> &data)
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
            points[i].distance = euclidean_dist(data[i], centroids[old_label], point_normSquares[i], centroid_normSquares[old_label]);
            feature_cnt += d;
            if (globallower < points[i].distance)
            {
                point_coord_type thresh = points[i].distance * points[i].distance;
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
                                adist = dist_comp(data[i], centroids[ci], thresh, pca_dim, feature_cnt);
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
                        group_nearest = std::sqrt(group_nearest);
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
                                group_lowers[i][gi] = std::sqrt(group_second_nearest);;
                                group[i] = gi;
                                points[i].index = group_nearest_index;
                                points[i].distance = group_nearest;
                                thresh = group_nearest * group_nearest;
                            }
                            else
                            {
                                group_lowers[i][gi] = group_nearest;
                            }
                        }
                        else
                        {
                            group_lowers[i][gi] = std::sqrt(group_second_nearest);;
                            thresh = group_nearest * group_nearest;
                            points[i].distance = group_nearest;
                            points[i].index = group_nearest_index;
                        }
                    }
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
    return converged;
}
