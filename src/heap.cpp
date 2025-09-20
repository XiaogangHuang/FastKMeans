#include "heap.h"

void HeapKmeans::fit()
{
    std::greater<std::pair<point_coord_type, int>> heapComp;
    bool converged = false;
    numDistances = 0;
    iterations = 0;
    while (!converged)
    {
        ++iterations;
        for (size_t h = 0; h < k; ++h)
        {
            Heap &heap = heaps[h];
            while (heap.size() > 0)
            {

                if (heapBounds[h] <= heap[0].first)
                    break;

                size_t i = heap[0].second;

                std::pop_heap(heap.begin(), heap.end(), heapComp);
                heap.pop_back();

                size_t closest = assignment[i];
                size_t nextClosest = 0;
                numDistances++;
                point_coord_type u2 = euclidean_dist_square(data[i], centroids[closest], point_normSquares[i], centroid_normSquares[closest]);
                point_coord_type l2 = std::numeric_limits<point_coord_type>::max();
                for (size_t j = 0; j < k; ++j)
                {
                    if (j == closest)
                        continue;
                    numDistances++;
                    point_coord_type dist2 = euclidean_dist_square(data[i], centroids[j], point_normSquares[i], centroid_normSquares[j]);
                    if (dist2 < u2)
                    {
                        l2 = u2;
                        u2 = dist2;
                        nextClosest = closest;
                        closest = j;
                    }
                    else if (dist2 < l2)
                    {
                        l2 = dist2;
                        nextClosest = j;
                    }
                }
                point_coord_type bound = std::sqrt(l2) - std::sqrt(u2);

                if ((bound == 0.0) && (nextClosest < closest))
                {
                    closest = nextClosest;
                }
                if (closest != assignment[i])
                {
                    for (size_t j = 0; j < d; j++)
                    {
                        sums[assignment[i]][j] -= data[i][j];
                    }
                    cluster_count[assignment[i]]--;

                    for (size_t j = 0; j < d; j++)
                    {
                        sums[closest][j] += data[i][j];
                    }
                    cluster_count[closest]++;
                    assignment[i] = closest;
                }

                Heap &newHeap = heaps[closest];
                newHeap.push_back(std::make_pair(heapBounds[closest] + bound, i));
                std::push_heap(newHeap.begin(), newHeap.end(), heapComp);
            }
        }
        converged = recalculateCentroids();
        update_bounds();
    }
}

void HeapKmeans::initialize(const Matrix<point_coord_type> &aX, const Matrix<point_coord_type> &initial_centroids,
                            unsigned short aK)
{
    data = aX;
    centroids = initial_centroids;
    old_centroids = initial_centroids;
    k = aK;
    n = data.size();
    d = data[0].size();
    heapBounds.resize(k, 0.0);
    heaps.resize(k);
    heaps[0].resize(n, std::make_pair(-1.0, 0));
    for (size_t j = 0; j < n; ++j)
    {
        heaps[0][j].second = j;
    }

    assignment.resize(n, 0);
    point_normSquares.resize(n, 0.0);
    cluster_count.resize(k, 0);
    centroid_normSquares.resize(k, 0.0);
    for (size_t i = 0; i < k; i++)
    {
        centroid_normSquares[i] = innerProduct(centroids[i]);
    }

    div.resize(k, 0.0);
    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    for (size_t i = 0; i < n; i++)
    {
        size_t ind = assignment[i];
        cluster_count[ind]++;
        for (size_t j = 0; j < d; j++)
        {
            sums[ind][j] += data[i][j];
        }
        point_normSquares[i] = innerProduct(data[i]);
    }
}

bool HeapKmeans::recalculateCentroids()
{
    bool converged = true;
    std::swap(centroids, old_centroids);

    for (size_t i = 0; i < k; ++i)
    {
        if (cluster_count[i] > 0)
        {
            point_coord_type scale = 1.0 / cluster_count[i];
            point_coord_type sum = 0;
            for (size_t j = 0; j < d; ++j)
            {
                centroids[i][j] = sums[i][j] * scale;
                sum += centroids[i][j] * centroids[i][j];
            }
            numDistances++;
            div[i] = euclidean_dist(centroids[i], old_centroids[i], sum, centroid_normSquares[i]);
            centroid_normSquares[i] = sum;
            if (div[i] > 0)
            {
                converged = false;
            }
        }
        else
        {
            centroids[i] = old_centroids[i];
            div[i] = 0;
        }
    }

    return converged;
}

void HeapKmeans::update_bounds()
{
    size_t furthestMovingCenter = 0;
    point_coord_type longest = div[furthestMovingCenter];
    point_coord_type secondLongest = 0.0;
    for (size_t j = 0; j < k; ++j)
    {
        if (longest < div[j])
        {
            secondLongest = longest;
            longest = div[j];
            furthestMovingCenter = j;
        }
        else if (secondLongest < div[j])
        {
            secondLongest = div[j];
        }
    }

    for (size_t j = 0; j < k; ++j)
    {
        heapBounds[j] += div[j];
        if (j == furthestMovingCenter)
        {
            heapBounds[j] += secondLongest;
        }
        else
        {
            heapBounds[j] += longest;
        }
    }
}
