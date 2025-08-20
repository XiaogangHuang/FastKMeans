#pragma once
#include "DADC.h"
#include "kmeans.h"

class AdaptiveKmeansV2Lloyd
{
public:
    AdaptiveKmeansV2Lloyd(size_t k, size_t ub);
    ~AdaptiveKmeansV2Lloyd();

    void setInitialCentroids(const Matrix<point_coord_type> &initial_centroids);
    void fit(const Matrix<point_coord_type> &data);

    std::vector<size_t> getLabels();
    [[nodiscard]] size_t getIterations() const { return iterations; }
    [[nodiscard]] size_t getNumDistances() const { return numDistances; }
    size_t getFeatureCnt() const { return feature_cnt; }
    [[nodiscard]] size_t getNumGroups() const { return ngroups; }
    [[nodiscard]] size_t getGroupSize(size_t i) const { return groupparts[i + 1] - groupparts[i]; }
    [[nodiscard]] const Matrix<point_coord_type> &getCentroids() const { return centroids; }

    void printTimeUsage()
    {
        std::cout << "Assign time: " << assign_time.count() << std::endl;
        std::cout << "Update time: " << update_time.count() << std::endl;
    }

    point_coord_type getMemoryUsage() const
    {
        size_t bytes = sizeof(*this);

        bytes += getMatrixMemoryBytes(centroids);
        bytes += getMatrixMemoryBytes(old_centroids);
        bytes += getMatrixMemoryBytes(sums);
        bytes += sizeof(Point) * points.capacity();
        bytes += sizeof(Center) * centers.capacity();
        bytes += getMatrixMemoryBytes(group_lowers);
        bytes += getVectorMemoryBytes(groupparts);
        bytes += getVectorMemoryBytes(point_normSquares);
        bytes += getVectorMemoryBytes(centroid_normSquares);
        bytes += getVectorMemoryBytes(group);
        bytes += getVectorMemoryBytes(div_center);
        bytes += getVectorMemoryBytes(div_group);

        return static_cast<point_coord_type>(bytes) / (1024.0 * 1024.0);
    }

private:
    void init(const Matrix<point_coord_type> &data);
    bool assignPoints(const Matrix<point_coord_type> &data);
    void recalculateCentroids();

    // 基本参数
    size_t k;            // 聚类数
    size_t ngroups;      // 特征维度
    size_t iterations;   // 当前迭代次数
    size_t numDistances; // 距离计算次数
    size_t n;            // 数据点数
    size_t d;            // 特征维度
    size_t pca_dim;
    size_t feature_cnt;

    // 核心数据
    Matrix<point_coord_type> centroids;     // 当前中心点
    Matrix<point_coord_type> old_centroids; // 上一轮中心点
    Matrix<point_coord_type> sums;
    Matrix<point_coord_type> group_lowers; // 下界矩阵[n][ngroups]
    std::vector<size_t> groupparts;

    std::vector<Point> points;   // 点[n]
    std::vector<size_t> group;   // 组标记[n]
    std::vector<Center> centers; // 中心点向量[k]
    std::vector<point_coord_type> point_normSquares;
    std::vector<point_coord_type> centroid_normSquares;

    std::vector<point_coord_type> div_center; // 中心点移动距离[k]
    std::vector<point_coord_type> div_group;

    std::chrono::milliseconds assign_time;
    std::chrono::milliseconds update_time;
};