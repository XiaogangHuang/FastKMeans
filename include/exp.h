#pragma once
#include "utils.h"

class ExponionKmeans
{
public:
    ExponionKmeans(size_t k);
    ~ExponionKmeans();

    void init(const Matrix<point_coord_type> &data);
    void setInitialCentroids(const Matrix<point_coord_type> &initial_centroids);
    void fit(const Matrix<point_coord_type> &data);
    void fit_ns(const Matrix<point_coord_type> &data);

    [[nodiscard]] const std::vector<size_t> &getLabels() const { return labels; }
    [[nodiscard]] size_t getIterations() const { return iterations; }
    [[nodiscard]] size_t getNumDistances() const { return numDistances; }
    [[nodiscard]] const Matrix<point_coord_type> &getCentroids() const { return centroids; }
    point_coord_type getMemoryUsage() const
    {
        size_t bytes = sizeof(*this);

        bytes += getMatrixMemoryBytes(centroids);
        bytes += getMatrixMemoryBytes(old_centroids);
        bytes += getMatrixMemoryBytes(sums);
        bytes += getMatrixMemoryBytes(partitionvalues_halvies);
        bytes += getMatrixMemoryBytes(geometricindices);
        bytes += getMatrixMemoryBytes(c_to_c);

        bytes += getVectorMemoryBytes(point_normSquares);
        bytes += getVectorMemoryBytes(centroid_normSquares);
        bytes += getVectorMemoryBytes(labels);
        bytes += getVectorMemoryBytes(cluster_count);
        bytes += getMatrixMemoryBytes(cumabs);
        bytes += getVectorMemoryBytes(l_hamerly);
        bytes += getVectorMemoryBytes(u_elkan);
        bytes += getVectorMemoryBytes(near);
        bytes += getVectorMemoryBytes(div);
        bytes += getVectorMemoryBytes(tau_lower);
        bytes += getVectorMemoryBytes(max_deltaC_since);

        return static_cast<point_coord_type>(bytes) / (1024.0 * 1024.0);
    }

private:
    void updateBounds();
    void assignPoints(const Matrix<point_coord_type> &data);
    void assignPoints_ns(const Matrix<point_coord_type> &data);
    bool recalculateCentroids();

    size_t get_insertion_index(point_coord_type vin, size_t n_sortedv, std::vector<point_coord_type> &sortedv);
    void update_pairs_parts_indices_halvies();

    // 基本参数
    size_t k;            // 聚类数
    size_t iterations;   // 当前迭代次数
    size_t numDistances; // 距离计算次数
    size_t n;            // 数据点数
    size_t d;            // 特征维度
    size_t npartitions;

    // 核心数据
    Matrix<point_coord_type> sums;
    Matrix<point_coord_type> centroids;     // 当前中心点
    Matrix<point_coord_type> old_centroids; // 上一轮中心点
    std::vector<point_coord_type> point_normSquares;
    std::vector<point_coord_type> centroid_normSquares;
    std::vector<size_t> labels;        // 点的分配标签
    std::vector<size_t> cluster_count; // 每个簇的点数

    // Hamerly算法特有的数据结构
    std::vector<point_coord_type> l_hamerly; // 第二近中心点的下界[n]
    std::vector<point_coord_type> u_elkan;   // 上界向量[n]
    std::vector<point_coord_type> near;      // 最近中心点距离的一半[k]
    std::vector<point_coord_type> div;       // 中心点移动距离[k]
    Matrix<point_coord_type> c_to_c;
    Matrix<point_coord_type> partitionvalues_halvies;
    Matrix<size_t> geometricindices;
    Matrix<std::pair<point_coord_type, int>> geometricpairs_halvies;

    // Norm of sum
    size_t period = 50;
    Matrix<point_coord_type> cumabs;
    std::vector<point_coord_type> max_deltaC_since;
    std::vector<size_t> tau_lower;
};
