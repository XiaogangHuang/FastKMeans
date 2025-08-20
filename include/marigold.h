#pragma once
#include "utils.h"

class MarigoldKmeans
{
public:
    MarigoldKmeans(size_t k, const Matrix<point_coord_type> &data, const Matrix<point_coord_type> &initial_centroids);
    ~MarigoldKmeans();

    size_t runKmeans();

    size_t getFeatureCnt() const { return feature_cnt; }
    [[nodiscard]] const std::vector<size_t> getLabels() const
    {
        return labels;
    }
    [[nodiscard]] size_t getIterations() const { return iterations; }
    [[nodiscard]] size_t getNumDistances() const { return numDistances; }
    point_coord_type getMemoryUsage() const
    {
        size_t bytes = sizeof(*this);

        bytes += getVectorMemoryBytes(labels) + getVectorMemoryBytes(cluster_count) +
                 getVectorMemoryBytes(l_hamerly) + getVectorMemoryBytes(u_elkan) +
                 getVectorMemoryBytes(near) + getVectorMemoryBytes(div) + getVectorMemoryBytes(dist) +
                 getVectorMemoryBytes(l_pow) + getVectorMemoryBytes(mask);

        bytes += getMatrixMemoryBytes(old_centroids);
        bytes += getMatrixMemoryBytes(data_ss);
        bytes += getMatrixMemoryBytes(centroid_ss);
        bytes += getMatrixMemoryBytes(sums);
        bytes += getMatrixMemoryBytes(centroids);
        bytes += getMatrixMemoryBytes(l_elkan);
        bytes += getMatrixMemoryBytes(c_to_c);

        return static_cast<point_coord_type>(bytes) / (1024.0 * 1024.0);
    }

private:
    void Calculate_squared_botup(const Matrix<point_coord_type> &raw, Matrix<point_coord_type> &squared);
    void MG_SetLabel(const size_t x);

    std::tuple<double, double> DistToLevel_bot(const size_t x, const size_t c, const size_t l, point_coord_type &UB, point_coord_type &LB);
    void Update_bounds();
    bool Recalculate();

    // 基本参数
    size_t k;            // 聚类数
    size_t iterations;   // 当前迭代次数
    size_t numDistances; // 距离计算次数
    size_t feature_cnt;
    size_t n; // 数据点数
    size_t d; // 特征维度
    size_t L;

    Matrix<point_coord_type> data_;
    Matrix<point_coord_type> data_ss;
    Matrix<point_coord_type> centroid_ss;
    Matrix<point_coord_type> centroids;
    Matrix<point_coord_type> old_centroids;

    Matrix<point_coord_type> sums;

    std::vector<size_t> labels;        // 点的分配标签
    std::vector<size_t> cluster_count; // 每个簇的点数

    Matrix<point_coord_type> l_elkan;        // 下界矩阵[n][k]
    std::vector<point_coord_type> l_hamerly; // 上界向量[n]
    std::vector<point_coord_type> u_elkan;   // 上界向量[n]
    std::vector<point_coord_type> near;      // 最近中心点距离的一半[k]
    std::vector<point_coord_type> div;       // 中心点移动距离[k]
    Matrix<point_coord_type> c_to_c;         // 中心点距离矩阵[k][k]

    std::vector<point_coord_type> dist;
    std::vector<size_t> l_pow;
    std::vector<size_t> mask;
};