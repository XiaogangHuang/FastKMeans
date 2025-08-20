#include "../include/ball_kmeans.h"

// Ball k-means主函数
void BallKmeans::fit()
{
    bool judge = true;

    Matrix<size_t> temp_cluster_point_index;
    Matrix<size_t> cluster_point_index;
    Matrix<size_t> clusters_neighbors_index;
    Matrix<size_t> now_data_index;
    Matrix<point_coord_type> temp_dis;

    Matrix<point_coord_type> new_centroids(k, std::vector<point_coord_type>(dim, 0));
    Matrix<point_coord_type> centers_dis(k, std::vector<point_coord_type>(k));

    std::vector<size_t> flag(k, 0);
    std::vector<size_t> old_flag(k, 0);

    delta.resize(k);
    point_normSquares.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        point_normSquares[i] = innerProduct(dataset[i]);
    }

    centroid_normSquares.resize(k);
    for (size_t i = 0; i < k; i++)
    {
        centroid_normSquares[i] = innerProduct(centers[i]);
    }

    std::vector<size_t> old_now_index;
    std::vector<point_coord_type> distance_arr;

    std::vector<point_coord_type> the_rs(k);

    size_t now_centers_rows;
    size_t num_of_neighbour;
    size_t neighbour_num;
    size_t data_num;
    size_t now_num;

    size_t minCol;
    num_of_neighbour = 0;
    iterations = 0; // 重置迭代次数

    // 初始化
    initialize(cluster_point_index, clusters_neighbors_index, temp_dis);
    temp_cluster_point_index.assign(cluster_point_index.begin(), cluster_point_index.end());

    while (true)
    {
        old_flag = flag;
        cluster_point_index.assign(temp_cluster_point_index.begin(), temp_cluster_point_index.end());
        iterations += 1;

        // 更新类中心
        if (!update_centroids(cluster_point_index, flag, new_centroids))
        {
            std::swap(centers, new_centroids);
            // 更新类半径
            update_radius(cluster_point_index, temp_dis, the_rs, flag, k);

            // 计算类中心间距离
            calculateCentersDistance(the_rs, centers_dis);

            std::fill(flag.begin(), flag.end(), 0);

            // 处理每个类
            for (size_t now_ball = 0; now_ball < k; now_ball++)
            {
                sortedNeighborClusters neighbors;
                getSortedNeighborsRing(the_rs, centers_dis, now_ball,
                                       clusters_neighbors_index[now_ball], neighbors);

                now_num = temp_dis[now_ball].size();
                if (the_rs[now_ball] == 0)
                    continue;

                old_now_index.clear();
                old_now_index.assign(clusters_neighbors_index[now_ball].begin(),
                                     clusters_neighbors_index[now_ball].end());
                clusters_neighbors_index[now_ball].clear();
                neighbour_num = neighbors.size();
                Matrix<point_coord_type> now_centers(neighbour_num, std::vector<point_coord_type>(dim));

                for (size_t i = 0; i < neighbour_num; i++)
                {
                    clusters_neighbors_index[now_ball].push_back(neighbors[i].index);
                    now_centers[i] = centers[neighbors[i].index];
                }
                num_of_neighbour += neighbour_num;

                now_centers_rows = now_centers.size();

                judge = true;

                if (clusters_neighbors_index[now_ball] != old_now_index)
                    judge = false;
                else
                {
                    for (size_t i : clusters_neighbors_index[now_ball])
                    {
                        if (old_flag[i] != false)
                        {
                            judge = false;
                            break;
                        }
                    }
                }

                if (judge)
                    continue;

                now_data_index.clear();
                distance_arr.clear();

                for (size_t j = 1; j < neighbour_num; j++)
                {
                    distance_arr.push_back(centers_dis[clusters_neighbors_index[now_ball][j]][now_ball] / 2);
                    now_data_index.push_back(std::vector<size_t>());
                }

                for (size_t i = 0; i < now_num; i++)
                {
                    for (size_t j = 1; j < neighbour_num; j++)
                    {
                        if (j == now_centers_rows - 1 && temp_dis[now_ball][i] > distance_arr[j - 1])
                        {
                            now_data_index[j - 1].push_back(cluster_point_index[now_ball][i]);
                            break;
                        }
                        if (j != now_centers_rows - 1 && temp_dis[now_ball][i] > distance_arr[j - 1] &&
                            temp_dis[now_ball][i] <= distance_arr[j])
                        {
                            now_data_index[j - 1].push_back(cluster_point_index[now_ball][i]);
                            break;
                        }
                    }
                }
                judge = false;

                // 处理每个区域
                for (size_t j = 1; j < neighbour_num; j++)
                {
                    data_num = now_data_index[j - 1].size();

                    if (data_num == 0)
                        continue;

                    Matrix<point_coord_type> temp_distance(data_num, std::vector<point_coord_type>(j + 1));
                    calculateRingDistance(j, data_num, now_centers, now_data_index, temp_distance);

                    size_t new_label;
                    for (size_t i = 0; i < data_num; i++)
                    {
                        minCol = std::min_element(temp_distance[i].begin(), temp_distance[i].end()) - temp_distance[i].begin();
                        new_label = clusters_neighbors_index[now_ball][minCol];
                        if (assignment[now_data_index[j - 1][i]] != new_label)
                        {
                            flag[now_ball] = true;
                            flag[new_label] = true;

                            auto it = (temp_cluster_point_index[assignment[now_data_index[j - 1][i]]]).begin();
                            while ((it) != (temp_cluster_point_index[assignment[now_data_index[j - 1][i]]]).end())
                            {
                                if (*it == now_data_index[j - 1][i])
                                {
                                    it = (temp_cluster_point_index[assignment[now_data_index[j - 1][i]]]).erase(it);
                                    break;
                                }
                                else
                                    ++it;
                            }
                            temp_cluster_point_index[new_label].push_back(now_data_index[j - 1][i]);
                            assignment[now_data_index[j - 1][i]] = new_label;
                        }
                    }
                }
            }
        }
        else
        {
            converged = true;
            break;
        }
    }
}

// 初始化，将点分配给最近的类中心
void BallKmeans::initialize(Matrix<size_t> &cluster_point_index,
                            Matrix<size_t> &clusters_neighbors_index,
                            Matrix<point_coord_type> &temp_dis)
{
    // 初始化数据结构
    cluster_point_index.resize(k);
    clusters_neighbors_index.resize(k);
    temp_dis.resize(k);

    // 计算距离矩阵
    for (size_t i = 0; i < dataset.size(); ++i)
    {
        point_coord_type min_dist = std::numeric_limits<point_coord_type>::max();
        size_t minCol = 0;
        for (size_t j = 0; j < centers.size(); ++j)
        {
            numDistances_assign++;
            point_coord_type adist = euclidean_dist_square(dataset[i], centers[j], point_normSquares[i], centroid_normSquares[j]);
            if (adist < min_dist)
            {
                min_dist = adist;
                minCol = j;
            }
        }
        assignment[i] = minCol;
        cluster_point_index[minCol].push_back(i);
    }
}

// 更新类中心
bool BallKmeans::update_centroids(Matrix<size_t> &cluster_point_index,
                                  std::vector<size_t> &flag,
                                  Matrix<point_coord_type> &new_c)
{
    bool isequal = true;
    new_c.assign(k, std::vector<point_coord_type>(dim, 0));

    for (size_t i = 0; i < k; i++)
    {
        if (flag[i] != 0 || iterations == 1)
        {
            const size_t cluster_size = cluster_point_index[i].size();
            if (cluster_size > 0)
            {
                // 累加簇中所有点的坐标
                for (size_t j = 0; j < cluster_size; j++)
                {
                    const auto &point = dataset[cluster_point_index[i][j]];
                    for (size_t d = 0; d < dim; d++)
                    {
                        new_c[i][d] += point[d];
                    }
                }
                // 计算平均值
                const point_coord_type inv_size = 1.0 / cluster_size;
                for (size_t d = 0; d < dim; d++)
                {
                    new_c[i][d] *= inv_size;
                }
                point_coord_type sum = innerProduct(new_c[i]);
                numDistances_other++;
                delta[i] = euclidean_dist(new_c[i], centers[i], centroid_normSquares[i], sum);
                centroid_normSquares[i] = sum;
                if (delta[i] > 0)
                {
                    isequal = false;
                }
            }
            else
            {
                new_c[i] = centers[i];
            }
        }
        else
        {
            new_c[i] = centers[i];
        }
    }
    return isequal;
}

// 更新类半径
void BallKmeans::update_radius(Matrix<size_t> &cluster_point_index,
                               Matrix<point_coord_type> &temp_dis,
                               std::vector<point_coord_type> &the_rs,
                               std::vector<size_t> &flag,
                               size_t the_rs_size)
{
    for (size_t i = 0; i < the_rs_size; i++)
    {
        if (flag[i] != 0 || iterations == 1)
        {
            const size_t cluster_size = cluster_point_index[i].size();
            the_rs[i] = 0;
            temp_dis[i].clear();
            temp_dis[i].reserve(cluster_size);

            for (size_t j = 0; j < cluster_size; j++)
            {
                numDistances_radius++;
                point_coord_type dist = euclidean_dist(centers[i], dataset[cluster_point_index[i][j]],
                                                       point_normSquares[cluster_point_index[i][j]], centroid_normSquares[i]);
                temp_dis[i].push_back(dist);
                the_rs[i] = std::max(the_rs[i], dist);
            }
        }
    }
}

// 计算类中心间的距离
void BallKmeans::calculateCentersDistance(std::vector<point_coord_type> &the_rs,
                                          Matrix<point_coord_type> &centers_dis)
{
    if (iterations == 1)
    {
        for (size_t i = 0; i < k; i++)
        {
            for (size_t j = i + 1; j < k; j++)
            {
                centers_dis[i][j] = euclidean_dist(centers[i], centers[j], centroid_normSquares[i], centroid_normSquares[j]);
                centers_dis[j][i] = centers_dis[i][j];
                numDistances_other++;
            }
        }
    }
    else
    {
        for (size_t i = 0; i < k; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                if (centers_dis[i][j] >= 2 * the_rs[i] + delta[i] + delta[j])
                {
                    centers_dis[i][j] -= (delta[i] + delta[j]);
                }
                else
                {
                    centers_dis[i][j] = euclidean_dist(centers[i], centers[j], centroid_normSquares[i], centroid_normSquares[j]);
                    numDistances_other++;
                }
            }
        }
    }
}

// 获取排序后的邻居类
void BallKmeans::getSortedNeighborsRing(std::vector<point_coord_type> &the_Rs,
                                        Matrix<point_coord_type> &centers_dis,
                                        size_t now_ball,
                                        std::vector<size_t> &now_center_index,
                                        sortedNeighborClusters &neighbors) const
{
    std::vector<bool> flag(k, false);
    neighbors.clear();
    neighbors.reserve(k);

    // 添加当前球
    NeighborCluster currentBall;
    currentBall.distance = 0;
    currentBall.index = static_cast<size_t>(now_ball);
    neighbors.push_back(currentBall);
    flag[now_ball] = true;

    // 添加已知的邻居
    for (size_t j = 1; j < now_center_index.size(); j++)
    {
        const size_t idx = now_center_index[j];
        if (centers_dis[now_ball][idx] == 0 ||
            2 * the_Rs[now_ball] - centers_dis[now_ball][idx] < 0)
        {
            flag[idx] = true;
        }
        else
        {
            NeighborCluster neighbor;
            neighbor.distance = centers_dis[now_ball][idx];
            neighbor.index = idx;
            neighbors.push_back(neighbor);
            flag[idx] = true;
        }
    }

    // 添加其他潜在的邻居
    for (size_t j = 0; j < k; j++)
    {
        if (!flag[j] && centers_dis[now_ball][j] != 0 &&
            2 * the_Rs[now_ball] - centers_dis[now_ball][j] >= 0)
        {
            NeighborCluster neighbor;
            neighbor.distance = centers_dis[now_ball][j];
            neighbor.index = j;
            neighbors.push_back(neighbor);
        }
    }

    // 按距离排序
    std::sort(neighbors.begin(), neighbors.end(),
              [](const NeighborCluster &a, const NeighborCluster &b)
              {
                  return a.distance < b.distance;
              });
}

// 计算环内距离
void BallKmeans::calculateRingDistance(size_t j, size_t data_num,
                                       Matrix<point_coord_type> &now_centers,
                                       Matrix<size_t> &now_data_index,
                                       Matrix<point_coord_type> &temp_distance)
{
    if (data_num == 0)
        return;

    for (size_t i = 0; i < data_num; ++i)
    {
        size_t idx = now_data_index[j - 1][i];
        for (size_t jt = 0; jt < j + 1; ++jt)
        {
            temp_distance[i][jt] = euclidean_dist_square(dataset[idx], now_centers[jt]);
        }
    }
    numDistances_assign += data_num * (j + 1);
}
