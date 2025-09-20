#pragma once
#include "DADC.h"

class AdaptiveKmeansV2PCA
{
public:
    AdaptiveKmeansV2PCA(size_t k, size_t ub);
    ~AdaptiveKmeansV2PCA();

    void setInitialCentroids(const Matrix<point_coord_type> &initial_centroids, const size_t dim);
    void fit(const Matrix<point_coord_type> &data);

    size_t getFeatureCnt() const { return feature_cnt; }
    [[nodiscard]] const std::vector<size_t> getLabels() const
    {
        std::vector<size_t> labels(n, 0);
        for (size_t i = 0; i < n; ++i)
        {
            labels[i] = points[i].label;
        }
        return labels;
    }
    [[nodiscard]] size_t getIterations() const { return iterations; }
    [[nodiscard]] size_t getNumDistances() const { return numDistances; }
    [[nodiscard]] size_t getNumGroups() const { return numGroups; }
    [[nodiscard]] size_t getGroupSize(size_t i) const { return group_size[i]; }
    [[nodiscard]] const Matrix<point_coord_type> &getCentroids() const { return centroids; }
    point_coord_type getMemoryUsage() const
    {
        size_t bytes = sizeof(*this);
        bytes += sizeof(AdaptPointV2) * n;
        bytes += getMatrixMemoryBytes(centroids);
        bytes += getMatrixMemoryBytes(old_centroids);
        bytes += getMatrixMemoryBytes(sums);
        bytes += getMatrixMemoryBytes(group_lowers);
        bytes += getMatrixMemoryBytes(div_group);
        bytes += sizeof(CentroidNormSquareV2) * k;
        bytes += getVectorMemoryBytes(div) + getVectorMemoryBytes(group_size) +
                 getVectorMemoryBytes(cluster_count);
        for (const auto &mat : group_index)
            bytes += getMatrixMemoryBytes(mat);

        return static_cast<point_coord_type>(bytes) / (1024.0 * 1024.0);
    }

    void printTimeUsage()
    {
        std::cout << "Time decomposition: " << assign_time.count() / 1e6 << " " << update_time.count() / 1e6 << std::endl;
    }

private:
    void rearrange_centroids();
    void init(const Matrix<point_coord_type> &data);
    void init_group_generation(const Matrix<point_coord_type> &data);
    void assignPoints(const Matrix<point_coord_type> &data);
    bool recalculateCentroids();
    void verifyCentroids(const Matrix<point_coord_type> &data);

    // 基本参数
    size_t k;            // 聚类数
    size_t numGroups;    // 组数
    size_t iterations;   // 当前迭代次数
    size_t numDistances; // 距离计算次数
    size_t feature_cnt;
    size_t n;       // 数据点数
    size_t d;       // 特征维度
    size_t pca_dim; // PCA降维后的目标维度

    // 核心数据
    std::vector<AdaptPointV2> points;
    Matrix<point_coord_type> centroids;     // 当前中心点
    Matrix<point_coord_type> old_centroids; // 上一轮中心点
    Matrix<point_coord_type> sums;

    std::vector<CentroidNormSquareV2> centroid_normSquares;
    std::vector<size_t> group_size;    // 组的大小
    std::vector<size_t> cluster_count; // 每个簇的点数
    std::vector<point_coord_type> div; // 中心点移动距离[k]

    std::vector<Matrix<size_t>> group_index;
    Matrix<point_coord_type> group_lowers; // 下界矩阵
    Matrix<point_coord_type> div_group;    // 组内中心点移动距离

    std::chrono::nanoseconds assign_time;
    std::chrono::nanoseconds update_time;
};