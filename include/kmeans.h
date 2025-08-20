#ifndef KMEANS_H
#define KMEANS_H

#include "utils.h"

// K-means聚类算法的实现类
class KMeans
{
private:
    size_t n;
    size_t d;
    size_t k;                               // 聚类数
    Matrix<point_coord_type> centroids;     // 聚类中心
    Matrix<point_coord_type> old_centroids; // 聚类中心
    Matrix<point_coord_type> sums;
    std::vector<point_coord_type> point_normSquares;
    std::vector<point_coord_type> centroid_normSquares;
    std::vector<size_t> labels;      // 数据点的标签
    std::vector<size_t> cluster_cnt; //
    size_t numDistances;             // 距离计算次数
    size_t iterations;               // 迭代次数

    // 为每个数据点分配最近的聚类中心
    bool assignClusters(const Matrix<point_coord_type> &data)
    {
        bool changed = false;

        for (size_t i = 0; i < data.size(); ++i)
        {
            point_coord_type minDist = std::numeric_limits<point_coord_type>::max();
            size_t minIndex = 0;

            // 使用局部变量缓存当前数据点
            const auto &point = data[i];

            for (size_t j = 0; j < k; ++j)
            {
                numDistances++;
                point_coord_type dist = euclidean_dist_square(point, centroids[j], point_normSquares[i], centroid_normSquares[j]);
                if (dist < minDist)
                {
                    minDist = dist;
                    minIndex = j;
                }
            }
            if (minIndex != labels[i])
            {
                changed = true;
                for (size_t j = 0; j < d; j++)
                {
                    sums[labels[i]][j] -= data[i][j];
                }
                for (size_t j = 0; j < d; j++)
                {
                    sums[minIndex][j] += data[i][j];
                }
                cluster_cnt[minIndex]++;
                cluster_cnt[labels[i]]--;
                labels[i] = minIndex;
            }
        }

        return changed;
    }

    // 更新聚类中心
    point_coord_type updateCentroids()
    {
        // 使用临时数组存储新的中心点
        std::swap(centroids, old_centroids);

        // 计算新的中心点
        point_coord_type maxShift = 0.0;
        for (size_t i = 0; i < k; ++i)
        {
            if (cluster_cnt[i] > 0)
            {
                const point_coord_type inv_count = 1.0 / cluster_cnt[i];
                point_coord_type sum = 0.0;
                for (size_t j = 0; j < d; ++j)
                {
                    centroids[i][j] = sums[i][j] * inv_count;
                    sum += centroids[i][j] * centroids[i][j];
                }
                numDistances++;
                point_coord_type shift = euclidean_dist_square(centroids[i], old_centroids[i], sum, centroid_normSquares[i]);
                centroid_normSquares[i] = sum;
                maxShift = std::max(maxShift, shift);
            }
            else
            {
                centroids[i] = old_centroids[i];
            }
        }
        return maxShift;
    }

public:
    KMeans(size_t numClusters)
        : k(numClusters), numDistances(0), iterations(0) {}

    // 设置初始聚类中心
    void setInitialCentroids(const Matrix<point_coord_type> &initialCentroids)
    {
        centroids = initialCentroids;
        old_centroids = initialCentroids;
    }

    // 训练模型
    void fit(const Matrix<point_coord_type> &data)
    {
        n = data.size();
        d = data[0].size();
        labels.resize(n, 0);
        point_normSquares.resize(n);
        sums.resize(k, std::vector<point_coord_type>(d, 0.0));
        cluster_cnt.resize(k, 0);
        for (size_t i = 0; i < n; i++)
        {
            size_t ind = labels[i];
            cluster_cnt[ind]++;
            for (size_t j = 0; j < d; j++)
            {
                sums[ind][j] += data[i][j];
            }
            point_normSquares[i] = innerProduct(data[i]);
        }
        centroid_normSquares.resize(k);
        for (size_t i = 0; i < k; i++)
        {
            centroid_normSquares[i] = innerProduct(centroids[i]);
        }
        // 迭代优化
        iterations = 0;
        while (true)
        {
            iterations++;
            bool changed = assignClusters(data);
            if (!changed || updateCentroids() < std::numeric_limits<point_coord_type>::min())
            {
                break;
            }
        }
    }

    point_coord_type getMemoryUsage() const
    {
        size_t bytes = sizeof(*this);
        bytes += getMatrixMemoryBytes(centroids);
        bytes += getMatrixMemoryBytes(old_centroids);
        bytes += getMatrixMemoryBytes(sums);
        bytes += getVectorMemoryBytes(point_normSquares);
        bytes += getVectorMemoryBytes(centroid_normSquares);
        return static_cast<point_coord_type>(bytes) / (1024.0 * 1024.0);
    }

    // 获取聚类中心
    [[nodiscard]] const Matrix<point_coord_type> &getCentroids() const
    {
        return centroids;
    }

    // 获取训练数据的标签
    [[nodiscard]] const std::vector<size_t> &getLabels() const
    {
        return labels;
    }

    // 获取距离计算次数
    [[nodiscard]] size_t getNumDistances() const { return numDistances; }

    // 获取迭代次数
    [[nodiscard]] size_t getIterations() const
    {
        return iterations;
    }
};

#endif // KMEANS_H