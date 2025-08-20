#pragma once
#include "DADC.h"

class UltraSimplifiedElkan
{
public:
    UltraSimplifiedElkan(size_t k);
    ~UltraSimplifiedElkan();

    std::vector<size_t> getLabels();
    void setInitialCentroids(const Matrix<point_coord_type> &initial_centroids, size_t dim = 0);
    void fit(const Matrix<point_coord_type> &data);
    void fit_stepwise(const Matrix<point_coord_type> &data);
    void fit_ns(const Matrix<point_coord_type> &data);
    void fit_ns_stepwise(const Matrix<point_coord_type> &data);

    size_t getFeatureCnt() const { return feature_cnt; }
    [[nodiscard]] size_t getIterations() const { return iterations; }
    [[nodiscard]] size_t getNumDistances() const { return numDistances; }
    [[nodiscard]] const Matrix<point_coord_type> &getCentroids() const { return centroids; }
    point_coord_type getMemoryUsage() const
    {
        size_t bytes = sizeof(*this);

        bytes += getMatrixMemoryBytes(centroids);
        bytes += getMatrixMemoryBytes(old_centroids);
        bytes += getMatrixMemoryBytes(sums);
        bytes += sizeof(CentroidNormSquareV2) * point_normSquares.capacity();
        bytes += sizeof(CentroidNormSquareV2) * centroid_normSquares.capacity();
        for (const auto &hist : centroids_history)
        {
            bytes += getMatrixMemoryBytes(hist);
        }
        bytes += getMatrixMemoryBytes(timestamp);
        bytes += getMatrixMemoryBytes(div_ns);
        bytes += getMatrixMemoryBytes(lower_bounds);
        bytes += getVectorMemoryBytes(near);
        bytes += getVectorMemoryBytes(div);
        bytes += getVectorMemoryBytes(div_ns_);
        bytes += getMatrixMemoryBytes(c_to_c);

        return static_cast<point_coord_type>(bytes) / (1024.0 * 1024.0);
    }

private:
    void init(const Matrix<point_coord_type> &data);
    bool assignPoints(const Matrix<point_coord_type> &data);
    void recalculateCentroids();
    void updateBounds();

    void init_stepwise(const Matrix<point_coord_type> &data);
    bool assignPoints_stepwise(const Matrix<point_coord_type> &data);
    void recalculateCentroids_stepwise();

    void init_ns(const Matrix<point_coord_type> &data);
    bool assignPoints_ns(const Matrix<point_coord_type> &data);
    void recalculateCentroids_ns();

    void init_ns_stepwise(const Matrix<point_coord_type> &data);
    bool assignPoints_ns_stepwise(const Matrix<point_coord_type> &data);
    void recalculateCentroids_ns_stepwise();

    // 基本参数
    size_t k;            // 聚类数
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

    Matrix<point_coord_type> lower_bounds; // 下界矩阵[n][k]
    Matrix<point_coord_type> c_to_c;       // 中心点距离矩阵[k][k]

    std::vector<Point> points;   // 点[n]
    std::vector<Center> centers; // 中心点向量[k]
    std::vector<CentroidNormSquareV2> point_normSquares;
    std::vector<CentroidNormSquareV2> centroid_normSquares;
    std::vector<point_coord_type> near; // 最近中心点距离的一半[k]
    std::vector<point_coord_type> div;  // 中心点移动距离[k]

    // Norm of sum版本
    std::vector<Matrix<point_coord_type>> centroids_history;
    Matrix<point_coord_type> div_ns;
    std::vector<point_coord_type> div_ns_; // 中心点移动距离[k]
    Matrix<uint32_t> timestamp;
};