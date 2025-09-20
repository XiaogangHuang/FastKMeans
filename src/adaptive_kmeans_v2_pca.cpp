#include "adaptive_kmeans_v2_pca.h"

AdaptiveKmeansV2PCA::AdaptiveKmeansV2PCA(size_t k, size_t ub)
    : k(k), iterations(1), numDistances(0), n(0), d(0)
{
    size_t cnt = 0;
    numGroups = 0;
    while (cnt < k)
    {
        size_t sz;
        if (numGroups < 17)
            sz = std::min(ub, GROUP_SIZE[numGroups++]);
        else
        {
            sz = ub;
            numGroups++;
        }
        group_size.push_back(sz);
        cnt += sz;
        // if (numGroups < ub)
        // {
        //     size_t sz = GROUP_SIZE[numGroups++];
        //     group_size.push_back(sz);
        //     cnt += sz;
        // }
        // else
        // {
        //     numGroups++;
        //     group_size.push_back(GROUP_SIZE[ub]);
        //     cnt += GROUP_SIZE[ub];
        // }
    }
    group_size.back() -= cnt - k;
}

AdaptiveKmeansV2PCA::~AdaptiveKmeansV2PCA()
{
}

void AdaptiveKmeansV2PCA::rearrange_centroids()
{
    Matrix<point_coord_type> dists(k, std::vector<point_coord_type>(k, 0.0));
    for (size_t i = 0; i < k; i++)
    {
        for (size_t j = i + 1; j < k; j++)
        {
            numDistances++;
            feature_cnt += d;
            point_coord_type dist = euclidean_dist_square(centroids[i], centroids[j], centroid_normSquares[i].total_normSquare, centroid_normSquares[j].total_normSquare);
            dists[i][j] = dist;
            dists[j][i] = dist;
        }
    }
    group_index.resize(k, std::vector<std::vector<size_t>>(numGroups));
    std::vector<size_t> indices(k);
    std::iota(indices.begin(), indices.end(), 0);
    for (size_t i = 0; i < k; i++)
    {
        auto begin_it = indices.begin();
        auto end_it = indices.end();
        std::sort(begin_it, end_it, [&](size_t a, size_t b)
                  { return dists[i][a] < dists[i][b]; });
        for (size_t f = 0; f < numGroups; ++f)
        {
            size_t count = group_size[f];
            end_it = begin_it + count;
            group_index[i][f].assign(begin_it, end_it);
            begin_it = end_it;
        }
    }

    // std::vector<size_t> lpow(numGroups + 1, 0);
    // for (size_t f = 0; f < numGroups; ++f)
    // {
    //     lpow[f + 1] = lpow[f] + group_size[f];
    // }
    // for (size_t i = 0; i < k; i++)
    // {
    //     auto begin_it = indices.begin();
    //     for (int f = numGroups; f > 0; --f)
    //     {
    //         std::nth_element(begin_it, begin_it + lpow[f - 1],
    //                          begin_it + lpow[f], [&](size_t a, size_t b)
    //                          { return dists[i][a] < dists[i][b]; });
    //         group_index[i][f - 1].assign(begin_it + lpow[f - 1], begin_it + lpow[f]);
    //     }
    // }

    size_t cnt = 0;
    for (size_t i = 0; i < numGroups; i++)
    {
        for (size_t j = 0; j < group_index[0][i].size(); j++)
        {
            size_t idx = group_index[0][i][j];
            indices[idx] = cnt++;
        }
    }
    for (size_t it = 0; it < k; it++)
    {
        auto &group_it = group_index[it];
        for (size_t i = 0; i < group_it.size(); ++i)
        {
            auto &vec = group_it[i];
            for (size_t j = 0; j < vec.size(); ++j)
            {
                vec[j] = indices[vec[j]];
            }
        }
    }
    std::sort(group_index.begin(), group_index.end(),
              [&](const std::vector<std::vector<size_t>> &a, const std::vector<std::vector<size_t>> &b)
              {
                  return a[0][0] < b[0][0];
              });
    std::vector<CentroidNormSquareV2> centroid_normSquares_new(k);
    for (size_t i = 0; i < k; i++)
    {
        old_centroids[indices[i]] = centroids[i];
        centroid_normSquares_new[indices[i]] = centroid_normSquares[i];
    }
    std::swap(centroids, old_centroids);
    std::swap(centroid_normSquares, centroid_normSquares_new);
}

void AdaptiveKmeansV2PCA::setInitialCentroids(const Matrix<point_coord_type> &initial_centroids, const size_t dim)
{
    pca_dim = dim;
    centroids = initial_centroids;
    centroid_normSquares.resize(k, CentroidNormSquareV2{0.0, 0.0});
    for (size_t i = 0; i < k; i++)
    {
        point_coord_type temp = innerProduct(initial_centroids[i],
                                             centroid_normSquares[i].rest_norm, pca_dim);
        centroid_normSquares[i].total_normSquare = temp;
    }
    old_centroids = initial_centroids;
    numDistances = 0;
}

void AdaptiveKmeansV2PCA::fit(const Matrix<point_coord_type> &data)
{
    init(data);
    while (!recalculateCentroids())
    {
        assignPoints(data);
        iterations++;
    }
    // auto start = std::chrono::high_resolution_clock::now();
    // init(data);
    // auto end = std::chrono::high_resolution_clock::now();
    // assign_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // // verifyCentroids(data);
    // while (true)
    // {
    //     start = std::chrono::high_resolution_clock::now();
    //     bool converged = recalculateCentroids();
    //     end = std::chrono::high_resolution_clock::now();
    //     update_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    //     if (converged)
    //     {
    //         break;
    //     }
    //     start = std::chrono::high_resolution_clock::now();
    //     assignPoints(data);
    //     end = std::chrono::high_resolution_clock::now();
    //     assign_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    //     iterations++;
    // }
}

void AdaptiveKmeansV2PCA::init(const Matrix<point_coord_type> &data)
{
    n = data.size();
    d = data[0].size();
    feature_cnt = 0;

    // 初始化数据结构
    div_group.resize(k, std::vector<point_coord_type>(numGroups, 0.0));
    div.resize(k, 0);
    cluster_count.resize(k, 0);
    points.resize(n, AdaptPointV2{std::numeric_limits<point_coord_type>::max(), 0.0, 0.0, 0, 0, 0});
    for (size_t i = 0; i < n; i++)
    {
        point_coord_type temp = innerProduct(data[i], points[i].rest_norm, pca_dim);
        points[i].total_normSquare = temp;
    }
    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    rearrange_centroids();
    init_group_generation(data);
}

void AdaptiveKmeansV2PCA::init_group_generation(const Matrix<point_coord_type> &data)
{
    group_lowers.resize(n, std::vector<point_coord_type>(numGroups, 0.0));
    std::vector<point_coord_type> dist_vec(k);
    for (size_t i = 0; i < n; ++i)
    {
        point_coord_type min_dist = std::numeric_limits<point_coord_type>::max();
        size_t labels_i = 0;
        for (size_t j = 0; j < k; j++)
        {
            numDistances++;
            point_coord_type dist = dist_comp(data[i], centroids[j], points[i],
                                              centroid_normSquares[j], min_dist, pca_dim, feature_cnt);

            dist_vec[j] = dist;
            if (dist < min_dist)
            {
                min_dist = dist;
                labels_i = j;
            }
        }
        points[i].distance = std::sqrt(min_dist);
        points[i].init_clust = labels_i;
        points[i].label = labels_i;
        cluster_count[labels_i]++;
        for (size_t j = 0; j < d; j++)
        {
            sums[labels_i][j] += data[i][j];
        }

        group_lowers[i][0] = std::numeric_limits<point_coord_type>::max();
        for (size_t j = 1; j < numGroups; j++)
        {
            auto &vec = group_index[labels_i][j];
            min_dist = std::numeric_limits<point_coord_type>::max();
            for (size_t gi = 0; gi < vec.size(); gi++)
            {
                size_t clust = vec[gi];
                if (dist_vec[clust] < min_dist)
                {
                    min_dist = dist_vec[clust];
                }
            }
            group_lowers[i][j] = std::sqrt(min_dist);
        }
    }
}

void AdaptiveKmeansV2PCA::assignPoints(const Matrix<point_coord_type> &data)
{
    for (size_t i = 0; i < n; ++i)
    {
        auto &point = points[i];
        auto &group_lower_row = group_lowers[i];
        const auto &temp = div_group[point.init_clust];
        point_coord_type globallower = std::numeric_limits<point_coord_type>::max();
        for (size_t it = 0; it < numGroups; ++it)
        {
            point_coord_type val = std::max(0.0, group_lower_row[it] - temp[it]);
            group_lower_row[it] = val;

            if (val < globallower)
                globallower = val;
        }
        size_t old_label = point.label;
        point.distance += div[old_label];
        if (globallower < point.distance)
        {
            numDistances++;
            feature_cnt += d;
            point_coord_type thresh = point.total_normSquare + centroid_normSquares[old_label].total_normSquare - 2 * innerProduct(data[i], centroids[old_label], pca_dim);
            point.distance = thresh > 0 ? std::sqrt(thresh) : 0.0;
            for (size_t gi = 0; gi < numGroups; gi++)
            {
                if (group_lower_row[gi] >= point.distance)
                    continue;

                point_coord_type group_nearest = std::numeric_limits<point_coord_type>::max();
                point_coord_type group_second_nearest = std::numeric_limits<point_coord_type>::max();
                size_t group_nearest_index = 0;
                for (size_t clust : group_index[point.init_clust][gi])
                {
                    numDistances++;
                    point_coord_type adist = dist_comp(data[i], centroids[clust], point,
                                                       centroid_normSquares[clust], thresh, pca_dim, feature_cnt);

                    if (adist < group_nearest)
                    {
                        group_second_nearest = group_nearest;
                        group_nearest = adist;
                        group_nearest_index = clust;
                    }
                    else if (adist < group_second_nearest)
                    {
                        group_second_nearest = adist;
                    }
                }
                group_nearest = std::sqrt(group_nearest);
                // if (group_nearest <= point.distance)
                // {
                //     if (point.group != gi)
                //     {
                //         group_lower_row[point.group] = std::min(group_lower_row[point.group], point.distance);
                //     }
                //     group_lower_row[gi] = std::sqrt(group_second_nearest);
                //     thresh = group_nearest * group_nearest;
                //     point.distance = group_nearest;
                //     point.group = gi;
                //     point.label = group_nearest_index;
                // }
                // else
                // {
                //     group_lower_row[gi] = group_nearest;
                // }
                if (point.group != gi)
                {
                    if (group_nearest < point.distance)
                    {
                        if (point.distance < group_lower_row[point.group])
                        {
                            group_lower_row[point.group] = point.distance;
                        }
                        group_lower_row[gi] = std::sqrt(group_second_nearest);
                        thresh = group_nearest * group_nearest;
                        point.distance = group_nearest;
                        point.group = gi;
                        point.label = group_nearest_index;
                    }
                    else
                    {
                        group_lower_row[gi] = group_nearest;
                    }
                }
                else
                {
                    group_lower_row[gi] = std::sqrt(group_second_nearest);
                    thresh = group_nearest * group_nearest;
                    point.distance = group_nearest;
                    point.label = group_nearest_index;
                }
            }
            if (old_label != point.label)
            {
                for (size_t j = 0; j < d; j++)
                {
                    sums[old_label][j] -= data[i][j];
                }
                for (size_t j = 0; j < d; j++)
                {
                    sums[point.label][j] += data[i][j];
                }
                cluster_count[old_label]--;
                cluster_count[point.label]++;
            }
        }
    }
}

bool AdaptiveKmeansV2PCA::recalculateCentroids()
{
    std::swap(centroids, old_centroids);

    point_coord_type sum_div = 0.0;
    for (size_t i = 0; i < k; i++)
    {
        if (cluster_count[i] > 0)
        {
            point_coord_type scale = 1.0 / cluster_count[i];
            for (size_t j = 0; j < d; j++)
            {
                point_coord_type temp = sums[i][j] * scale;
                centroids[i][j] = temp;
            }
            numDistances++;
            feature_cnt += d;
            point_coord_type temp = centroid_normSquares[i].total_normSquare;
            centroid_normSquares[i].total_normSquare = innerProduct(centroids[i],
                                                                    centroid_normSquares[i].rest_norm, pca_dim);
            temp += centroid_normSquares[i].total_normSquare - 2 * innerProduct(centroids[i], old_centroids[i], pca_dim);
            div[i] = temp > 0 ? std::sqrt(temp) : 0.0;
            sum_div += div[i];
        }
        else
        {
            centroids[i] = old_centroids[i];
            div[i] = 0;
        }
    }

    for (size_t i = 0; i < k; i++)
    {
        for (size_t gi = 0; gi < numGroups; gi++)
        {
            const auto &group = group_index[i][gi];
            point_coord_type group_max = div[group[0]];
            for (size_t it = 1; it < group.size(); ++it)
            {
                size_t nei = group[it];
                if (group_max < div[nei])
                {
                    group_max = div[nei];
                }
            }
            div_group[i][gi] = group_max;
        }
    }

    return sum_div == 0.0;
}

void AdaptiveKmeansV2PCA::verifyCentroids(const Matrix<point_coord_type> &data)
{
    for (size_t i = 0; i < n; i++)
    {
        point_coord_type temp = euclidean_dist(data[i], centroids[points[i].label],
                                               points[i].total_normSquare, centroid_normSquares[points[i].label].total_normSquare);
        for (size_t j = 0; j < k; j++)
        {
            if (i != j && temp > euclidean_dist(data[i], centroids[j], points[i].total_normSquare, centroid_normSquares[j].total_normSquare))
            {
                std::cout << i << " " << points[i].label << " " << j << " " << std::setprecision(20) << temp << std::endl;
                break;
            }
        }
    }
}
