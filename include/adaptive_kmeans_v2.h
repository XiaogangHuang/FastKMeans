#pragma once
#include "DADC.h"

// struct LowerBound
// {
//     point_coord_type first_lower;
//     point_coord_type second_lower;
//     size_t first_index;
// };

class AdaptiveKmeansV2
{
public:
    AdaptiveKmeansV2(size_t k, size_t ub);
    ~AdaptiveKmeansV2();

    void setInitialCentroids(const Matrix<point_coord_type> &initial_centroids);
    void fit(const Matrix<point_coord_type> &data);
    // void fit_ns(const Matrix<point_coord_type> &data);
    // void fit_2lowers(const Matrix<point_coord_type> &data);

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
    [[nodiscard]] const Matrix<point_coord_type> &getCentroids() const { return centroids; }

    point_coord_type getMemoryUsage() const
    {
        size_t bytes = sizeof(*this);
        bytes += sizeof(AdaptPointV1) * points.size();
        bytes += getMatrixMemoryBytes(centroids);
        bytes += getMatrixMemoryBytes(old_centroids);
        bytes += getMatrixMemoryBytes(sums);
        bytes += getMatrixMemoryBytes(group_lowers);
        bytes += getMatrixMemoryBytes(div_group);
        bytes += getVectorMemoryBytes(group_size) + getVectorMemoryBytes(cluster_count) +
                 getVectorMemoryBytes(centroid_normSquares) + getVectorMemoryBytes(div);
        for (const auto &mat : group_index)
            bytes += getMatrixMemoryBytes(mat);

        return static_cast<point_coord_type>(bytes) / (1024.0 * 1024.0);
    }

private:
    void init(const Matrix<point_coord_type> &data);
    void rearrange_centroids();
    void init_group_generation(const Matrix<point_coord_type> &data);
    void assignPoints(const Matrix<point_coord_type> &data);
    bool recalculateCentroids();

    // void init_ns(const Matrix<point_coord_type> &data);
    // void assignPoints_ns(const Matrix<point_coord_type> &data);
    // bool recalculateCentroids_ns();

    // void init_2lowers(const Matrix<point_coord_type> &data);
    // void init_group_generation_2lowers(const Matrix<point_coord_type> &data);
    // void assignPoints_2lowers(const Matrix<point_coord_type> &data);

    // 基本参数
    size_t k;            // 聚类数
    size_t numGroups;    // 组数
    size_t iterations;   // 当前迭代次数
    size_t numDistances; // 距离计算次数
    size_t feature_cnt;
    size_t n; // 数据点数
    size_t d; // 特征维度

    // 核心数据
    std::vector<AdaptPointV1> points;

    Matrix<point_coord_type> centroids;     // 当前中心点
    Matrix<point_coord_type> old_centroids; // 上一轮中心点
    Matrix<point_coord_type> sums;

    std::vector<size_t> group_size;    // 组的大小
    std::vector<size_t> cluster_count; // 每个簇的点数
    std::vector<point_coord_type> centroid_normSquares;
    std::vector<point_coord_type> div; // 中心点移动距离[k]

    std::vector<Matrix<size_t>> group_index;
    Matrix<point_coord_type> group_lowers; // 下界矩阵
    Matrix<point_coord_type> div_group;    // 组内中心点移动距离

    // Matrix<LowerBound> group_lowers_new;   // 下界矩阵
    // // Norm of sum版本
    // std::vector<point_coord_type> div_global;
    // std::vector<point_coord_type> globallowers_at_last;
    // std::vector<size_t> tau_globallowers;
    // std::vector<Matrix<point_coord_type>> centroids_history;
    // Matrix<point_coord_type> div_ns;
    // std::vector<Matrix<point_coord_type>> div_ns_g;
    // Matrix<size_t> timestamp;
};