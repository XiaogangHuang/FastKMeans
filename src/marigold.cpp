#include "marigold.h"
#include <cfloat>

MarigoldKmeans::MarigoldKmeans(size_t _k, const Matrix<point_coord_type> &data, const Matrix<point_coord_type> &initial_centroids)
{
    n = data.size();
    d = data[0].size();
    k = _k;
    data_ = data;
    centroids = initial_centroids;
    old_centroids = initial_centroids;

    feature_cnt = 0;
    numDistances = 0; // Initialize the number of distances calculated
    iterations = 0;

    L = ceil(log10(d) / log10(4));

    l_elkan.resize(n, std::vector<point_coord_type>(k, 0.0));
    l_hamerly.resize(n, 0.0);
    u_elkan.resize(n, std::numeric_limits<point_coord_type>::max());
    near.resize(k, 0.0);
    div.resize(k, 0.0);
    c_to_c.resize(k, std::vector<point_coord_type>(k, 0.0));

    l_pow.resize(L + 1);
    l_pow[0] = 0;
    for (size_t i = 1; i < L; i++)
    {
        l_pow[i] = 1 << (2 * i);
    }
    l_pow[L] = d;

    data_ss.resize(n, std::vector<point_coord_type>(L + 1, 0.0));
    centroid_ss.resize(k, std::vector<point_coord_type>(L + 1, 0.0));

    labels.resize(n, 0);
    cluster_count.resize(k, 0);
    sums.resize(k, std::vector<point_coord_type>(d, 0.0));
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < d; j++)
        {
            sums[0][j] += data[i][j];
        }
        cluster_count[0]++;
    }

    mask.resize(k, 0);
    dist.resize(k, 0.0);
}

MarigoldKmeans::~MarigoldKmeans()
{
}

size_t MarigoldKmeans::runKmeans()
{
    bool converged = false;
    Calculate_squared_botup(data_, data_ss);
    while (!converged)
    {
        Calculate_squared_botup(centroids, centroid_ss);

        for (size_t i = 0; i < n; i++)
        {
            point_coord_type val = near[labels[i]] < l_hamerly[i] ? l_hamerly[i] : near[labels[i]];
            if (u_elkan[i] > val)
            {
                MG_SetLabel(i);
            }
        }
        converged = Recalculate();
        if (!converged)
        {
            // TODO: refactor location of .. you know the drill
            Update_bounds();
        }
        iterations++;
    }

    return 0;
}

void MarigoldKmeans::Calculate_squared_botup(const Matrix<point_coord_type> &raw, Matrix<point_coord_type> &squared)
{
    size_t elements = raw.size();

    for (size_t e = 0; e < elements; e++)
    {
        squared[e][L] = 0;
        for (int l_ = L; l_ > 0; l_--)
        {
            squared[e][l_ - 1] = squared[e][l_];
            for (size_t i = l_pow[l_ - 1]; i < l_pow[l_]; i++)
            {
                squared[e][l_ - 1] += raw[e][i] * raw[e][i];
            }
        }
    }
}

void MarigoldKmeans::MG_SetLabel(const size_t x)
{
    for (size_t j = 0; j < k; j++)
    {
        dist[j] = data_ss[x][0] + centroid_ss[j][0];
    }
    size_t old_label = labels[x];

    std::fill(mask.begin(), mask.end(), 1);
    point_coord_type val;
    point_coord_type UB, LB;
    size_t mask_sum = k;
    size_t l = 0;
    while (l <= L && mask_sum > 0)
    {
        for (size_t j = 0; j < k; j++)
        {
            if (mask[j] != 1)
            {
                continue;
            }

            val = std::max(l_elkan[x][j], 0.5 * c_to_c[labels[x]][j]);
            if (u_elkan[x] < val)
            {                // Elkan check
                mask[j] = 0; // Mark as pruned centroid
            }
            else
            {
                if (l == 0)
                {
                    numDistances++;
                }

                DistToLevel_bot(x, j, l, UB, LB);
                LB = sqrt(std::max(0.0, LB));
                if (LB > l_elkan[x][j])
                {
                    l_elkan[x][j] = LB; // Keep maximum LB per c
                }

                UB = sqrt(std::max(0.0, UB));
                if (UB < u_elkan[x])
                {
                    labels[x] = j;
                    u_elkan[x] = UB; // Keep minimum UB across c
                }
            }
        }
        mask_sum = 0;
        for (size_t j = 0; j < k; j++)
        {
            mask_sum += mask[j];
        }
        l++;
    }
    if (old_label != labels[x])
    {
        for (size_t j = 0; j < d; j++)
        {
            sums[old_label][j] -= data_[x][j];
        }
        for (size_t j = 0; j < d; j++)
        {
            sums[labels[x]][j] += data_[x][j];
        }
        cluster_count[old_label]--;
        cluster_count[labels[x]]++;
    }
}

std::tuple<double, double> MarigoldKmeans::DistToLevel_bot(const size_t x, const size_t c, const size_t l, point_coord_type &UB, point_coord_type &LB)
{
    if (l > 0)
    {
        for (size_t i = l_pow[l - 1]; i < l_pow[l]; i++)
        {
            dist[c] -= 2 * data_[x][i] * centroids[c][i];
        }
        feature_cnt += l_pow[l] - l_pow[l - 1];
    }

    double margin = 2 * sqrt((data_ss[x][l]) * (centroid_ss[c][l]));

    LB = dist[c] - margin; // sqrt(std::max(0.0,dist - margin));
    UB = dist[c] + margin; // sqrt(std::max(0.0,dist + margin));

    return {LB, UB};
}

void MarigoldKmeans::Update_bounds()
{
    for (size_t i = 0; i < n; i++)
    {
        size_t smallest_id = labels[i] == 0 ? 1 : 0;
        for (size_t j = 0; j < k; j++)
        {
            double val = l_elkan[i][j] - div[j];
            l_elkan[i][j] = 0 < val ? val : 0;
            if ((labels[i] != j) && (l_elkan[i][j] <= l_elkan[i][smallest_id]))
            {
                smallest_id = j;
            }
        }
        u_elkan[i] += div[labels[i]];
        l_hamerly[i] = l_elkan[i][smallest_id];
    }
    for (size_t i = 0; i < k; i++)
    {
        c_to_c[i][i] = 0;
        for (size_t j = i + 1; j < k; j++)
        {
            double tmp = 0;
            for (size_t f = 0; f < d; f++)
            {
                tmp += ((centroids[i][f] - centroids[j][f]) *
                        (centroids[i][f] - centroids[j][f]));
            }
            numDistances++;
            feature_cnt += d;
            tmp = tmp > 0 ? std::sqrt(tmp) : 0.0;

            c_to_c[i][j] = (tmp);
            c_to_c[j][i] = c_to_c[i][j];
        }
        double smallest = DBL_MAX;
        for (size_t j = 0; j < k; j++)
        {
            if (i == j)
                continue;
            if (c_to_c[i][j] < smallest)
            {
                smallest = c_to_c[i][j];
                near[i] = 0.5 * smallest;
            }
        }
    }
}

bool MarigoldKmeans::Recalculate()
{
    std::swap(centroids, old_centroids);
    point_coord_type sum_div = 0.0;
    for (size_t i = 0; i < k; i++)
    {
        if (cluster_count[i] > 0)
        {
            point_coord_type scale = 1.0 / cluster_count[i];
            point_coord_type sum = 0.0;
            for (size_t j = 0; j < d; j++)
            {
                point_coord_type temp = sums[i][j] * scale;
                centroids[i][j] = temp;
                sum += (temp - old_centroids[i][j]) * (temp - old_centroids[i][j]);
            }
            numDistances++;
            feature_cnt += d;
            div[i] = sum > 0 ? std::sqrt(sum) : 0.0;
            sum_div += div[i];
        }
        else
        {
            centroids[i] = old_centroids[i];
            div[i] = 0;
        }
    }
    return sum_div == 0.0;
}