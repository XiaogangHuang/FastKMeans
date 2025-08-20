#pragma once
#include "DADC.h"

class BVKmeans
{
public:
    BVKmeans(size_t k);
    ~BVKmeans();

    void setInitialCentroids(point_coord_type perc, const Matrix<point_coord_type> &initial_centroids);
    void fit(const Matrix<point_coord_type> &data);

    size_t getFeatureCnt() const { return feature_cnt; }
    [[nodiscard]] const std::vector<size_t> getLabels() const
    {
        std::vector<size_t> labels(n, 0);
        for (size_t i = 0; i < n; ++i)
        {
            labels[i] = points[i].index;
        }
        return labels;
    }
    [[nodiscard]] size_t getIterations() const { return iterations; }
    [[nodiscard]] size_t getNumDistances() const { return numDistances; }
    [[nodiscard]] const Matrix<point_coord_type> &getCentroids() const { return centroids; }
    point_coord_type getMemoryUsage() const
    {
        size_t bytes = sizeof(*this);

        bytes += getVectorMemoryBytes(div_group) + getVectorMemoryBytes(div_center) +
                 getVectorMemoryBytes(point_normSquares) + getVectorMemoryBytes(centroid_normSquares) +
                 getVectorMemoryBytes(groupparts) + getVectorMemoryBytes(group);

        bytes += sizeof(Point) * points.capacity();
        bytes += sizeof(Center) * centers.capacity();
        bytes += getMatrixMemoryBytes(centroids);
        bytes += getMatrixMemoryBytes(old_centroids);
        bytes += getMatrixMemoryBytes(bv_data);
        bytes += getMatrixMemoryBytes(bv_centroids);
        bytes += getMatrixMemoryBytes(sums);
        bytes += getMatrixMemoryBytes(group_lowers);

        return static_cast<point_coord_type>(bytes) / (1024.0 * 1024.0);
    }

private:
    void init(const Matrix<point_coord_type> &data);
    bool assignPoints(const Matrix<point_coord_type> &data);
    void recalculateCentroids();

    point_coord_type distance_bv(size_t x, size_t y, point_coord_type &thresh, const Matrix<point_coord_type> &data);

    // 基本参数
    size_t k;            // 聚类数
    size_t iterations;   // 当前迭代次数
    size_t numDistances; // 距离计算次数
    size_t feature_cnt;
    size_t n;          // 数据点数
    size_t d;          // 特征维度
    size_t block_size; // 块大小
    size_t num_block;  // 块数
    size_t ngroups;

    // 核心数据
    Matrix<point_coord_type> centroids;     // 当前中心点
    Matrix<point_coord_type> old_centroids; // 上一轮中心点
    Matrix<point_coord_type> bv_data;
    Matrix<point_coord_type> bv_centroids;
    Matrix<point_coord_type> sums;

    Matrix<point_coord_type> group_lowers; // 下界矩阵
    std::vector<point_coord_type> div_group;
    std::vector<point_coord_type> div_center; // 中心点移动距离[k]

    std::vector<size_t> groupparts;
    std::vector<Point> points;   // 点[n]
    std::vector<size_t> group;   // 组标记[n]
    std::vector<Center> centers; // 中心点向量[k]
    std::vector<point_coord_type> point_normSquares;
    std::vector<point_coord_type> centroid_normSquares;
};