#ifndef YINYANG_KMEANS_H
#define YINYANG_KMEANS_H

#include "DADC.h"
#include "kmeans.h"

class YinYangKmeans
{
public:
    YinYangKmeans(size_t k);
    ~YinYangKmeans();

    void setInitialCentroids(const Matrix<point_coord_type> &initial_centroids);
    void fit(const Matrix<point_coord_type> &data);
    void fit_stepwise(const Matrix<point_coord_type> &data, size_t &dim);
    void fit_ns(const Matrix<point_coord_type> &data);
    void fit_ns_stepwise(const Matrix<point_coord_type> &data, size_t &dim);

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
        for (const auto &hist : centroids_history)
        {
            bytes += getMatrixMemoryBytes(hist);
        }
        bytes += getMatrixMemoryBytes(timestamp);
        bytes += getMatrixMemoryBytes(div_ns);
        bytes += getMatrixMemoryBytes(div_ns_g);
        bytes += getMatrixMemoryBytes(group_lowers);
        bytes += getVectorMemoryBytes(groupparts);
        bytes += getVectorMemoryBytes(point_normSquares);
        bytes += getVectorMemoryBytes(centroid_normSquares);
        bytes += getVectorMemoryBytes(group);
        bytes += getVectorMemoryBytes(div_center);
        bytes += getVectorMemoryBytes(div_group);
        bytes += getVectorMemoryBytes(div_global);
        bytes += getVectorMemoryBytes(globallowers_at_last);
        bytes += getVectorMemoryBytes(tau_globallowers);

        return static_cast<point_coord_type>(bytes) / (1024.0 * 1024.0);
    }

private:
    void init(const Matrix<point_coord_type> &data);
    bool assignPoints(const Matrix<point_coord_type> &data);
    void recalculateCentroids();

    bool assignPoints_stepwise(const Matrix<point_coord_type> &data);

    // Norm of sum version
    void init_ns(const Matrix<point_coord_type> &data);
    bool assignPoints_ns_first_iter(const Matrix<point_coord_type> &data);
    void recalculateCentroids_ns_first_iter();
    bool assignPoints_ns(const Matrix<point_coord_type> &data);
    void recalculateCentroids_ns();

    bool assignPoints_ns_first_iter_stepwise(const Matrix<point_coord_type> &data);
    bool assignPoints_ns_stepwise(const Matrix<point_coord_type> &data);

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

    // Norm of sum版本
    std::vector<point_coord_type> div_global;
    std::vector<point_coord_type> globallowers_at_last;

    std::vector<Matrix<point_coord_type>> centroids_history;
    Matrix<point_coord_type> div_ns;
    Matrix<point_coord_type> div_ns_g;
    Matrix<size_t> timestamp;
    std::vector<size_t> tau_globallowers;

    std::chrono::milliseconds assign_time;
    std::chrono::milliseconds update_time;
};

#endif // YINYANG_KMEANS_H