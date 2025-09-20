#pragma once
#include "utils.h"

typedef std::vector<std::pair<double, int>> Heap;

class HeapKmeans
{
public:
    HeapKmeans() {}
    ~HeapKmeans() {}
    void initialize(const Matrix<point_coord_type> &aX, const Matrix<point_coord_type> &initial_centroids,
                    unsigned short aK);
    void fit();

    [[nodiscard]] const std::vector<size_t> &getLabels() const { return assignment; }
    [[nodiscard]] size_t getIterations() const { return iterations; }
    [[nodiscard]] size_t getNumDistances() const { return numDistances; }
    point_coord_type getMemoryUsage() const
    {
        size_t bytes = sizeof(*this);

        bytes += getMatrixMemoryBytes(centroids);
        bytes += getMatrixMemoryBytes(old_centroids);
        bytes += getMatrixMemoryBytes(sums);
        bytes += sizeof(Matrix<std::pair<double, int>>);
        bytes += heaps.capacity() * sizeof(std::vector<std::pair<double, int>>);
        for (const auto &row : heaps)
            bytes += row.capacity() * sizeof(std::pair<double, int>);

        bytes += getVectorMemoryBytes(point_normSquares);
        bytes += getVectorMemoryBytes(centroid_normSquares);
        bytes += getVectorMemoryBytes(cluster_count);
        bytes += getVectorMemoryBytes(div);
        bytes += getVectorMemoryBytes(assignment);
        bytes += getVectorMemoryBytes(heapBounds);

        return static_cast<point_coord_type>(bytes) / (1024.0 * 1024.0);
    }

protected:
    void update_bounds();
    bool recalculateCentroids();

    size_t k;
    size_t n;            // 数据点数
    size_t d;            // 特征维度
    size_t numDistances; // 距离计算次数
    size_t iterations;   // 当前迭代次数

    Matrix<point_coord_type> data;
    Matrix<point_coord_type> centroids;
    Matrix<point_coord_type> old_centroids;
    Matrix<point_coord_type> sums;
    std::vector<point_coord_type> point_normSquares;
    std::vector<point_coord_type> centroid_normSquares;
    std::vector<size_t> cluster_count; // 每个簇的点数
    std::vector<point_coord_type> div; // 中心点移动距离[k]

    std::vector<size_t> assignment; // 点的分配标签
    std::vector<Heap> heaps;

    // The heapBounds essentially accumulate the total distance traveled by
    // each center over the iterations of k-means. This value is used to
    // compare with the heap priority to determine if a point's bounds
    // (lower-upper) are violated (i.e. < 0).
    std::vector<point_coord_type> heapBounds;

    // size_t numLowerBounds;

    // Half the distance between each center and its closest other center.
    // std::vector<point_coord_type> s;

    // One upper bound for each point on the distance between that point and
    // its assigned (closest) center.
    // std::vector<point_coord_type> upper;

    // Lower bound(s) for each point on the distance between that point and
    // the centers being tracked for lower bounds, which may be 1 to k.
    // Actual size is n * numLowerBounds.
    // Matrix<point_coord_type> lower;
};
