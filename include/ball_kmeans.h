#ifndef BALL_KMEANS_H
#define BALL_KMEANS_H

#include "utils.h"

// 定义邻居结构
struct NeighborCluster
{
    point_coord_type distance;
    size_t index;
};

typedef std::vector<NeighborCluster> sortedNeighborClusters;

class BallKmeans
{
public:
    BallKmeans(const Matrix<point_coord_type> &dataset, size_t k)
        : dataset(dataset), centers(), dim(dataset[0].size()), n(dataset.size()), k(k),
          numDistances_assign(0), numDistances_radius(0), numDistances_other(0), converged(false), assignment(n, 0), iterations(0)
    {
        if (dataset.empty() || dataset[0].empty())
        {
            throw std::runtime_error("数据集为空");
        }
        if (k > n)
        {
            throw std::runtime_error("聚类数无效");
        }
    }

    // 设置初始聚类中心
    void setInitialCentroids(const Matrix<point_coord_type> &initialCentroids)
    {
        if (initialCentroids.size() != k)
        {
            throw std::runtime_error("初始聚类中心数量不匹配");
        }
        centers = initialCentroids;
        // 重置相关变量
        numDistances_assign = 0;
        numDistances_radius = 0;
        numDistances_other = 0;
        iterations = 0;
        std::fill(assignment.begin(), assignment.end(), 0);
        converged = false;
    }

    // 获取聚类标签
    [[nodiscard]] const std::vector<size_t> &getLabels() const
    {
        return assignment;
    }

    // 获取距离计算次数
    [[nodiscard]] size_t getNumDistances() const
    {
        return numDistances_assign + numDistances_radius + numDistances_other;
    }

    // 获取迭代次数
    [[nodiscard]] size_t getIterations() const
    {
        return iterations;
    }

    // 打印距离计算次数
    void printDistCalDetail() const
    {
        std::cout << "numDistances_assign: " << numDistances_assign << std::endl;
        std::cout << "numDistances_radius: " << numDistances_radius << std::endl;
        std::cout << "numDistances_other: " << numDistances_other << std::endl;
    }

    // Ball k-means主函数
    void fit();

    // 初始化函数
    void initialize(Matrix<size_t> &cluster_point_index,
                    Matrix<size_t> &clusters_neighbors_index,
                    Matrix<point_coord_type> &temp_dis);

    // 更新类中心
    bool update_centroids(Matrix<size_t> &cluster_point_index,
                          std::vector<size_t> &flag,
                          Matrix<point_coord_type> &new_c);

    // 更新类半径
    void update_radius(Matrix<size_t> &cluster_point_index,
                       Matrix<point_coord_type> &temp_dis,
                       std::vector<point_coord_type> &the_rs,
                       std::vector<size_t> &flag,
                       size_t the_rs_size);

    // 计算类中心间的距离
    void calculateCentersDistance(std::vector<point_coord_type> &the_rs,
                                  Matrix<point_coord_type> &centers_dis);

    // 获取排序后的邻居类
    void getSortedNeighborsRing(std::vector<point_coord_type> &the_Rs,
                                Matrix<point_coord_type> &centers_dis,
                                size_t now_ball,
                                std::vector<size_t> &now_center_index,
                                sortedNeighborClusters &neighbors) const;

    // 计算环内距离
    void calculateRingDistance(size_t j, size_t data_num,
                               Matrix<point_coord_type> &now_centers,
                               Matrix<size_t> &now_data_index,
                               Matrix<point_coord_type> &temp_distance);

    point_coord_type getMemoryUsage() const
    {
        size_t bytes = sizeof(*this);
        bytes += getVectorMemoryBytes(assignment);
        bytes += getVectorMemoryBytes(point_normSquares) +
                 getVectorMemoryBytes(centroid_normSquares) + getVectorMemoryBytes(delta);

        bytes += getMatrixMemoryBytes(centers);

        return static_cast<point_coord_type>(bytes) / (1024.0 * 1024.0); // 返回 MB 单位
    }

protected:
    const Matrix<point_coord_type> &dataset;
    Matrix<point_coord_type> centers;
    std::vector<point_coord_type> point_normSquares;
    std::vector<point_coord_type> centroid_normSquares;
    std::vector<point_coord_type> delta;
    size_t dim;
    size_t n;
    size_t k;
    size_t numDistances_assign;
    size_t numDistances_radius;
    size_t numDistances_other;
    bool converged;
    std::vector<size_t> assignment;
    size_t iterations = 0;
};

#endif // BALL_KMEANS_H