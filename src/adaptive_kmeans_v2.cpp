#include "adaptive_kmeans_v2.h"

AdaptiveKmeansV2::AdaptiveKmeansV2(size_t k, size_t ub)
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

AdaptiveKmeansV2::~AdaptiveKmeansV2()
{
}

void AdaptiveKmeansV2::setInitialCentroids(const Matrix<point_coord_type> &initial_centroids)
{
    centroids = initial_centroids;
    centroid_normSquares.resize(k, 0.0);
    for (size_t i = 0; i < k; i++)
    {
        point_coord_type temp = innerProduct(initial_centroids[i]);
        centroid_normSquares[i] = temp;
    }
    old_centroids = initial_centroids;
    numDistances = 0;
}

void AdaptiveKmeansV2::fit(const Matrix<point_coord_type> &data)
{
    init(data);
    while (!recalculateCentroids())
    {
        assignPoints(data);
        iterations++;
    }
}

void AdaptiveKmeansV2::init(const Matrix<point_coord_type> &data)
{
    n = data.size();
    d = data[0].size();
    feature_cnt = 0;

    // 初始化数据结构
    div.resize(k, 0);
    cluster_count.resize(k, 0);
    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    div_group.resize(k, std::vector<point_coord_type>(numGroups, 0.0));
    points.resize(n, AdaptPointV1{std::numeric_limits<point_coord_type>::max(), 0.0, 0, 0, 0});
    for (size_t i = 0; i < n; i++)
    {
        point_coord_type temp = innerProduct(data[i]);
        points[i].total_normSquare = temp;
    }
    rearrange_centroids();
    init_group_generation(data);
}

void AdaptiveKmeansV2::rearrange_centroids()
{
    Matrix<point_coord_type> dists(k, std::vector<point_coord_type>(k, 0.0));
    for (size_t i = 0; i < k; i++)
    {
        for (size_t j = i + 1; j < k; j++)
        {
            numDistances++;
            feature_cnt += d;
            point_coord_type dist = euclidean_dist_square(centroids[i], centroids[j], centroid_normSquares[i], centroid_normSquares[j]);
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
    std::vector<point_coord_type> centroid_normSquares_new(k);
    for (size_t i = 0; i < k; i++)
    {
        old_centroids[indices[i]] = centroids[i];
        centroid_normSquares_new[indices[i]] = centroid_normSquares[i];
    }
    std::swap(centroids, old_centroids);
    std::swap(centroid_normSquares, centroid_normSquares_new);
}

void AdaptiveKmeansV2::init_group_generation(const Matrix<point_coord_type> &data)
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
            feature_cnt += d;
            point_coord_type dist = euclidean_dist_square(data[i], centroids[j], points[i].total_normSquare, centroid_normSquares[j]);
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
            min_dist = std::numeric_limits<point_coord_type>::max();
            for (size_t gi = 0; gi < group_index[labels_i][j].size(); gi++)
            {
                size_t clust = group_index[labels_i][j][gi];
                if (dist_vec[clust] < min_dist)
                {
                    min_dist = dist_vec[clust];
                }
            }
            group_lowers[i][j] = std::sqrt(min_dist);
        }
    }
}

void AdaptiveKmeansV2::assignPoints(const Matrix<point_coord_type> &data)
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
            point.distance = euclidean_dist(data[i], centroids[old_label], point.total_normSquare, centroid_normSquares[old_label]);
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
                    feature_cnt += d;
                    point_coord_type adist = euclidean_dist_square(data[i], centroids[clust], point.total_normSquare, centroid_normSquares[clust]);
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
                if (point.group != gi)
                {
                    if (group_nearest < point.distance)
                    {
                        if (point.distance < group_lower_row[point.group])
                        {
                            group_lower_row[point.group] = point.distance;
                        }
                        group_lower_row[gi] = std::sqrt(group_second_nearest);
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

bool AdaptiveKmeansV2::recalculateCentroids()
{
    std::swap(centroids, old_centroids);

    point_coord_type sum_div = 0.0;
    for (size_t i = 0; i < k; i++)
    {
        if (cluster_count[i] > 0)
        {
            point_coord_type scale = 1.0 / cluster_count[i];
            point_coord_type normSquare = 0.0;
            for (size_t j = 0; j < d; j++)
            {
                point_coord_type temp = sums[i][j] * scale;
                centroids[i][j] = temp;
                normSquare += temp * temp;
            }
            numDistances++;
            feature_cnt += d;
            point_coord_type temp = normSquare + centroid_normSquares[i] - 2 * innerProduct(centroids[i], old_centroids[i]);
            div[i] = temp > 0 ? std::sqrt(temp) : 0.0;
            centroid_normSquares[i] = normSquare;
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

/*
void AdaptiveKmeansV2::fit_ns(const Matrix<point_coord_type> &data)
{
    init_ns(data);
    bool converged = false;
    while (!converged)
    {
        assignPoints_ns(data);
        converged = recalculateCentroids_ns();
        iterations++;
    }
}

void AdaptiveKmeansV2::init_ns(const Matrix<point_coord_type> &data)
{
    n = data.size();
    d = data[0].size();
    feature_cnt = 0;

    // 初始化数据结构
    div_group.resize(k, std::vector<point_coord_type>(numGroups, 0.0));
    div.resize(k, 0);
    cluster_count.resize(k, 0);
    points.resize(n, AdaptPointV2{std::numeric_limits<point_coord_type>::max(), 0.0, 0.0, 0.0, 0, 0, 0});
    for (size_t i = 0; i < n; i++)
    {
        point_coord_type temp = innerProduct(data[i]);
        points[i].total_normSquare = temp;
    }
    sums.resize(k, std::vector<point_coord_type>(d, 0.0));

    timestamp.resize(n, std::vector<size_t>(numGroups, 0));
    globallowers_at_last.resize(n, 0.0);
    tau_globallowers.resize(n, 0);
    centroids_history.reserve(50);
    div_ns.reserve(50);
    div_ns_g.reserve(50);
    rearrange_centroids();
    init_group_generation(data);
    for (size_t i = 0; i < n; i++)
    {
        globallowers_at_last[i] = *std::min_element(group_lowers[i].begin(), group_lowers[i].end());
    }

    centroids_history.emplace_back(centroids);
    auto &last_centroids = centroids_history.back();
    for (size_t i = 0; i < k; i++)
    {
        if (cluster_count[i] > 0)
        {
            point_coord_type scale = 1.0 / cluster_count[i];
            point_coord_type normSquare = 0.0;
            for (size_t j = 0; j < d; j++)
            {
                point_coord_type temp = sums[i][j] * scale;
                centroids[i][j] = temp;
                normSquare += temp * temp;
            }
            numDistances++;
            feature_cnt += d;
            div[i] = std::sqrt(normSquare + centroid_normSquares[i].total_normSquare - 2 * innerProduct(centroids[i], last_centroids[i]));
            centroid_normSquares[i].total_normSquare = normSquare;
        }
        else
        {
            centroids[i] = last_centroids[i];
            div[i] = 0;
        }
    }
    div_ns.emplace_back(div);
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
    div_ns_g.emplace_back(div_group);
    div_global.emplace_back(*std::max_element(div.begin(), div.end()));
}

void AdaptiveKmeansV2::assignPoints_ns(const Matrix<point_coord_type> &data)
{
    for (size_t i = 0; i < n; ++i)
    {
        auto &point = points[i];
        size_t old_label = point.label;
        point_coord_type globallower = std::fmax(globallowers_at_last[i] - div_global[tau_globallowers[i]], 0.0);
        point.distance += div[old_label];
        if (globallower < point.distance)
        {
            numDistances++;
            feature_cnt += d;
            point.distance = euclidean_dist(data[i], centroids[old_label], point.total_normSquare, centroid_normSquares[old_label].total_normSquare);
            auto &group_lower_row = group_lowers[i];
            globallowers_at_last[i] = std::numeric_limits<point_coord_type>::max();
            for (size_t gi = 0; gi < numGroups; gi++)
            {
                point_coord_type temp = timestamp[i][gi] < iterations ? group_lower_row[gi] - div_ns_g[timestamp[i][gi]][point.init_clust][gi] : group_lower_row[gi];
                if (temp >= point.distance)
                {
                    point_coord_type lower = group_lower_row[gi];
                    lower -= timestamp[i][gi] < div_ns_g.size() ? div_ns_g[timestamp[i][gi]][point.init_clust][gi] : 0.0;
                    globallowers_at_last[i] = std::min(globallowers_at_last[i], lower);
                    continue;
                }

                timestamp[i][gi] = iterations;
                point_coord_type group_nearest = std::numeric_limits<point_coord_type>::max();
                point_coord_type group_second_nearest = std::numeric_limits<point_coord_type>::max();
                size_t group_nearest_index = 0;
                for (size_t clust : group_index[point.init_clust][gi])
                {
                    numDistances++;
                    feature_cnt += d;
                    point_coord_type adist = euclidean_dist_square(data[i], centroids[clust], point.total_normSquare, centroid_normSquares[clust].total_normSquare);
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
                if (point.group != gi)
                {
                    if (group_nearest < point.distance)
                    {
                        if (gi < point.group)
                        {
                            group_lower_row[point.group] = std::min(point.distance, group_lower_row[point.group] -
                                                                                        div_ns_g[timestamp[i][point.group]][point.init_clust][point.group]);
                        }
                        else
                        {
                            group_lower_row[point.group] = point.distance;
                            globallowers_at_last[i] = point.distance;
                        }
                        timestamp[i][point.group] = iterations;
                        group_lower_row[gi] = std::sqrt(group_second_nearest);
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
                    point.distance = group_nearest;
                    point.label = group_nearest_index;
                }
                point_coord_type lower = group_lower_row[gi];
                point_coord_type offset = 0.0;
                if (timestamp[i][gi] < div_ns_g.size())
                {
                    offset = div_ns_g[timestamp[i][gi]][point.init_clust][gi];
                }
                globallowers_at_last[i] = std::min(globallowers_at_last[i], lower - offset);
            }
            tau_globallowers[i] = iterations;
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

bool AdaptiveKmeansV2::recalculateCentroids_ns()
{
    centroids_history.emplace_back(centroids);
    auto &last_centroids = centroids_history.back();

    point_coord_type sum_div = 0.0;
    for (size_t i = 0; i < k; i++)
    {
        if (cluster_count[i] > 0)
        {
            point_coord_type scale = 1.0 / cluster_count[i];
            point_coord_type normSquare = 0.0;
            for (size_t j = 0; j < d; j++)
            {
                point_coord_type temp = sums[i][j] * scale;
                centroids[i][j] = temp;
                normSquare += temp * temp;
            }
            numDistances++;
            feature_cnt += d;
            div[i] = std::sqrt(normSquare + centroid_normSquares[i].total_normSquare - 2 * innerProduct(centroids[i], last_centroids[i]));
            centroid_normSquares[i].total_normSquare = normSquare;
            sum_div += div[i];
        }
        else
        {
            centroids[i] = last_centroids[i];
            div[i] = 0;
        }
    }

    div_ns.emplace_back(div);
    for (size_t i = 0; i < iterations; i++)
    {
        for (size_t j = 0; j < k; ++j)
        {
            if (div[j] > 0.0)
            {
                numDistances++;
                feature_cnt += d;
                div_ns[i][j] = euclidean_dist(centroids[j], centroids_history[i][j]);
            }
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
    div_ns_g.emplace_back(div_group);

    div_global.emplace_back(*std::max_element(div.begin(), div.end()));
    for (size_t i = 0; i < iterations; i++)
    {
        for (size_t j = 0; j < k; j++)
        {
            for (size_t gi = 0; gi < numGroups; ++gi)
            {
                const auto &group = group_index[j][gi];
                div_ns_g[i][j][gi] = div_ns[i][group[0]];
                for (size_t it = 1; it < group.size(); ++it)
                {
                    size_t nei = group[it];
                    if (div_ns[i][nei] > div_ns_g[i][j][gi])
                    {
                        div_ns_g[i][j][gi] = div_ns[i][nei];
                    }
                }
            }
        }
        div_global[i] = *std::max_element(div_ns[i].begin(), div_ns[i].end());
    }

    return sum_div == 0.0;
}

void AdaptiveKmeansV2::fit_2lowers(const Matrix<point_coord_type> &data)
{
    init_2lowers(data);

    while (!recalculateCentroids())
    {
        assignPoints_2lowers(data);
        iterations++;
    }
}

void AdaptiveKmeansV2::init_2lowers(const Matrix<point_coord_type> &data)
{
    n = data.size();
    d = data[0].size();

    // 初始化数据结构
    div_group.resize(k, std::vector<point_coord_type>(numGroups, 0.0));
    div.resize(k, 0);
    cluster_count.resize(k, 0);
    points.resize(n, AdaptPointV2{std::numeric_limits<point_coord_type>::max(), 0.0, 0.0, 0.0, 0, 0, 0});
    for (size_t i = 0; i < n; i++)
    {
        point_coord_type temp = innerProduct(data[i]);
        points[i].total_normSquare = temp;
    }
    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    rearrange_centroids();
    init_group_generation_2lowers(data);
}

void AdaptiveKmeansV2::init_group_generation_2lowers(const Matrix<point_coord_type> &data)
{
    group_lowers_new.resize(n, std::vector<LowerBound>(numGroups, LowerBound{0.0, 0.0, 0}));
    std::vector<point_coord_type> dist_vec(k);
    for (size_t i = 0; i < n; ++i)
    {
        point_coord_type min_dist = std::numeric_limits<point_coord_type>::max();
        size_t labels_i = 0;
        for (size_t j = 0; j < k; j++)
        {
            numDistances++;
            point_coord_type dist = euclidean_dist_square(data[i], centroids[j], points[i].total_normSquare, centroid_normSquares[j].total_normSquare);
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
        const auto &label_groups = group_index[labels_i];
        group_lowers_new[i][0].first_lower = std::numeric_limits<point_coord_type>::max();
        group_lowers_new[i][0].second_lower = std::numeric_limits<point_coord_type>::max();
        group_lowers_new[i][0].first_index = label_groups[0][0];
        for (size_t j = 1; j < numGroups; j++)
        {
            const auto &group_clusts = label_groups[j];
            min_dist = std::numeric_limits<point_coord_type>::max();
            point_coord_type min2 = std::numeric_limits<point_coord_type>::max();
            size_t min_idx = static_cast<size_t>(-1);
            for (size_t gi = 0; gi < group_clusts.size(); gi++)
            {
                size_t clust = group_clusts[gi];
                point_coord_type dist = dist_vec[clust];
                if (dist < min_dist)
                {
                    min2 = min_dist;
                    min_dist = dist;
                    min_idx = clust;
                }
                else if (dist < min2)
                {
                    min2 = dist;
                }
            }
            group_lowers_new[i][j].first_lower = std::sqrt(min_dist);
            group_lowers_new[i][j].second_lower = std::sqrt(min2);
            group_lowers_new[i][j].first_index = min_idx;
        }
    }
}

void AdaptiveKmeansV2::assignPoints_2lowers(const Matrix<point_coord_type> &data)
{
    point_coord_type adist;
    point_coord_type group_nearest, group_second_nearest;
    size_t group_nearest_index;

    for (size_t i = 0; i < n; ++i)
    {
        auto &point = points[i];
        auto &group_lower_row = group_lowers_new[i];
        const auto &temp = div_group[point.init_clust];
        point_coord_type globallower = std::numeric_limits<point_coord_type>::max();
        size_t old_label = point.label;
        for (size_t it = 0; it < numGroups; ++it)
        {
            auto &lower = group_lower_row[it];
            point_coord_type val2 = std::max(0.0, lower.second_lower - temp[it]);
            lower.second_lower = val2;

            point_coord_type val1 = std::numeric_limits<point_coord_type>::max();
            if (lower.first_index != old_label)
            {
                val1 = std::max(0.0, lower.first_lower - div[lower.first_index]);
                lower.first_lower = val1;
            }
            globallower = std::min(globallower, std::min(val1, val2));
        }

        point.distance += div[old_label];
        if (globallower < point.distance)
        {
            numDistances++;
            adist = euclidean_dist(data[i], centroids[old_label], point.total_normSquare, centroid_normSquares[old_label].total_normSquare);
            point.distance = adist;
            group_lower_row[point.group].first_lower = adist;
            for (size_t gi = 0; gi < numGroups; gi++)
            {
                if (group_lower_row[gi].second_lower < point.distance)
                {
                    group_nearest = std::numeric_limits<point_coord_type>::max();
                    group_second_nearest = std::numeric_limits<point_coord_type>::max();
                    group_nearest_index = 0;
                    for (size_t clust : group_index[point.init_clust][gi])
                    {
                        numDistances++;
                        adist = euclidean_dist_square(data[i], centroids[clust], point.total_normSquare, centroid_normSquares[clust].total_normSquare);
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
                    group_lower_row[gi].first_lower = group_nearest;
                    group_lower_row[gi].first_index = group_nearest_index;
                    group_lower_row[gi].second_lower = std::sqrt(group_second_nearest);
                    if (group_nearest < point.distance)
                    {
                        point.distance = group_nearest;
                        point.group = gi;
                        point.label = group_nearest_index;
                    }
                }
                else if (group_lower_row[gi].first_lower < point.distance)
                {
                    numDistances++;
                    adist = euclidean_dist(data[i], centroids[group_lower_row[gi].first_index], point.total_normSquare, centroid_normSquares[group_lower_row[gi].first_index].total_normSquare);
                    group_lower_row[gi].first_lower = adist;
                    if (adist < point.distance)
                    {
                        point.distance = adist;
                        point.group = gi;
                        point.label = group_lower_row[gi].first_index;
                    }
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
*/