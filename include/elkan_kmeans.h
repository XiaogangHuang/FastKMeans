#ifndef ELKAN_KMEANS_H
#define ELKAN_KMEANS_H

#include "DADC.h"

class ElkanKmeans
{
public:
    ElkanKmeans(size_t k);
    ~ElkanKmeans();

    void setInitialCentroids(const Matrix<point_coord_type> &initial_centroids, size_t dim = 0);
    void fit(const Matrix<point_coord_type> &data);
    void fit_stepwise(const Matrix<point_coord_type> &data);

    size_t getFeatureCnt() const { return feature_cnt; }
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
        bytes += getVectorMemoryBytes(point_normSquares);
        bytes += getVectorMemoryBytes(centroid_normSquares);
        bytes += getVectorMemoryBytes(labels);
        bytes += getVectorMemoryBytes(cluster_count);
        bytes += getMatrixMemoryBytes(div_ns);
        bytes += getMatrixMemoryBytes(l_elkan);
        bytes += getVectorMemoryBytes(u_elkan);
        bytes += getVectorMemoryBytes(near);
        bytes += getVectorMemoryBytes(div);
        bytes += getMatrixMemoryBytes(c_to_c);

        return static_cast<point_coord_type>(bytes) / (1024.0 * 1024.0);
    }

private:
    void init(const Matrix<point_coord_type> &data);
    void assignPoints(const Matrix<point_coord_type> &data);
    bool recalculateCentroids();
    void updateBounds();
    void clear();

    void assignPoints_stepwise(const Matrix<point_coord_type> &data);


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
    std::vector<point_coord_type> point_normSquares;
    std::vector<point_coord_type> centroid_normSquares;
    std::vector<size_t> labels;        // 点的分配标签
    std::vector<size_t> cluster_count; // 每个簇的点数

    // Elkan算法特有的数据结构
    Matrix<point_coord_type> div_ns;
    Matrix<point_coord_type> l_elkan;      // 下界矩阵[n][k]
    std::vector<point_coord_type> u_elkan; // 上界向量[n]
    std::vector<point_coord_type> near;    // 最近中心点距离的一半[k]
    std::vector<point_coord_type> div;     // 中心点移动距离[k]
    Matrix<point_coord_type> c_to_c;       // 中心点距离矩阵[k][k]
};

#endif // ELKAN_KMEANS_H