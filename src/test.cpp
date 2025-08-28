#include <filesystem>
#include "file_io.h"
#include "pca.h"

#include "ball_kmeans.h"
#include "elkan_kmeans.h"
#include "hamerly_kmeans.h"
#include "yykmeans.h"
#include "adaptive_kmeans_v2.h"
#include "adaptive_kmeans_v2_pca.h"
#include "adaptive_kmeans_v2_lloyd.h"
#include "marigold.h"
#include "bv_kmeans_v1.h"
#include "exp.h"
#include "heap.h"

// 提取和验证命令行参数的函数
void parseAndValidateArguments(size_t argc, char *argv[], std::string &alg, std::string &filename, int &rs_seed,
                               int &read_pca_from_file, int &read_centroids_from_file, size_t &k, size_t &ub,
                               point_coord_type &percent, size_t &pca_dim)
{
    int opt;
    while ((opt = getopt(argc, argv, "a:r:f:s:k:l:p:u:d:")) != -1)
    {
        switch (opt)
        {
        case 'a':
            alg = optarg;
            break;
        case 'f':
            filename += optarg;
            break;
        case 's':
            rs_seed = std::stoi(optarg);
            break;
        case 'k':
            k = std::stoi(optarg);
            break;
        case 'r':
            read_pca_from_file = std::stoi(optarg);
            break;
        case 'l':
            read_centroids_from_file = std::stoi(optarg);
            break;
        case 'p':
            percent = std::stod(optarg);
            break;
        case 'u':
            ub = std::stoi(optarg);
            break;
        case 'd':
            pca_dim = std::stoi(optarg);
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -f <filename> -s <seed> -l <centroids_from_file> -k <cluster_num> -p <percent> -u <group_size_limit>" << std::endl;
            exit(1);
        }
    }

    if (rs_seed == -1)
        rs_seed = static_cast<unsigned>(time(nullptr));

    // rs_seed = 1749559948;
    if (filename.empty() || k <= 0)
    {
        std::cerr << "Error: You must specify a data file, and the number of clusters must be greater than 0." << std::endl;
        exit(1);
    }
    else if (percent > 1.0 || percent < 0)
    {
        std::cerr << "Error: The ratio must be between 0 and 1.0." << std::endl;
        exit(1);
    }
}

template <typename T, typename = void>
struct has_getFeatureCnt : std::false_type
{
};

template <typename T>
struct has_getFeatureCnt<T, std::void_t<decltype(std::declval<const T>().getFeatureCnt())>> : std::true_type
{
};

template <typename T>
void print_feature_cnt(const T &algo, [[maybe_unused]] size_t width_feature)
{
    if constexpr (has_getFeatureCnt<T>::value)
    {
        size_t cnt = algo.getFeatureCnt();
        std::cout << std::setw(width_feature) << cnt;
    }
    else
    {
        // 什么也不做
    }
}

template <typename T>
void print_report(const std::string &algo_name, const T &algo, point_coord_type mem_usage, long time_ms, size_t width_dist, size_t width_feature = 0)
{
    std::cout << std::left << std::setw(29) << algo_name;
    std::cout << std::setw(7) << algo.getIterations()
              << std::setw(width_dist) << algo.getNumDistances()
              << std::setw(10) << time_ms
              << std::setw(12) << algo.getMemoryUsage() + mem_usage
              << std::setw(16) << (algo.getIterations() ? time_ms / algo.getIterations() : 0);
    if (algo_name != "Elkan K-means" && algo_name != "Yinyang K-means")
    {
        print_feature_cnt(algo, width_feature);
    }

    std::cout << std::endl;
}

template <typename AlgoType, typename InitFunc, typename FitFunc>
void run_and_report(const std::string &name, AlgoType &algo, InitFunc init, FitFunc fit, point_coord_type mem_usage,
                    size_t width_dist, size_t width_feature)
{
    init(algo);
    auto start = std::chrono::high_resolution_clock::now();
    fit(algo);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    print_report(name, algo, mem_usage, duration.count(), width_dist, width_feature);
}

int main(int argc, char *argv[])
{
    try
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        std::string filename = "../Data/", alg = "";
        int rs_seed = -1, read_pca_from_file = 1, read_centroids_from_file = 0;
        std::string label_file = "../Data/init_centroids.txt";
        size_t k = 10, pca_dim = 10000, ub = 13;
        point_coord_type percent = 0.9;
        parseAndValidateArguments(argc, argv, alg, filename, rs_seed, read_pca_from_file, read_centroids_from_file, k, ub, percent, pca_dim);

        // 读取数据
        std::cout << "Reading (generating) data and initial centroids..." << std::endl;
        Matrix<point_coord_type> data;
        if (filename.size() >= 6 && filename.substr(filename.size() - 6) == ".fvecs")
        {
            data = read_fvecs<point_coord_type>(filename.c_str());
            std::string txt_filename = filename.substr(0, filename.size() - 6) + ".txt";
            if (!std::filesystem::exists(txt_filename)) // 如果文件不存在，则写入
            {
                writeMatrixToFile(txt_filename, data);
                std::cout << "The .fvecs data has been written to " << txt_filename << std::endl;
            }
        }
        else
        {
            data = readMatrixFromFile<point_coord_type>(filename.c_str());
        }
        size_t rows = data.size(), cols = data[0].size();
        std::cout << "Successfully read data from the file" << std::endl;

        std::vector<size_t> init_centroids;
        Matrix<point_coord_type> initial_centroids;
        if (read_centroids_from_file != 0)
        {
            initial_centroids = readInitialCentroidsFromFile<point_coord_type>(data, label_file.c_str(), init_centroids);
            std::cout << "Successfully read the initial centroids from the file" << std::endl;
        }
        else
        {
            std::cout << "Using k-means++ to generate initial centroids(" << rs_seed << ")" << std::endl;
            initial_centroids = initializeCentroidsKMeansPlusPlus(data, init_centroids, k, rs_seed);
        }

        Matrix<point_coord_type> pca_data;
        std::filesystem::path data_path(filename);
        std::string base_name = data_path.stem().string(); // 提取数据文件名（无扩展名）
        std::filesystem::path pca_dir = "../Data/PCA";
        std::filesystem::path pca_file_data = pca_dir / (base_name + "_pca_data.dvecs");
        std::filesystem::path pca_file_matrix = pca_dir / (base_name + "_pca_matrix.dvecs");
        if (std::filesystem::exists(pca_file_data) && std::filesystem::exists(pca_file_matrix) && read_pca_from_file == 1)
        {
            std::cout << "Detected saved PCA file, reading..." << std::endl;
            pca_data = read_dvecs<point_coord_type>(pca_file_data.string());
            Matrix<point_coord_type> pca_matrix = read_dvecs<point_coord_type>(pca_file_matrix.string());
            if (pca_dim == 10000)
            {
                pca_dim = 0;
                determine_pca_dim(pca_matrix[0], pca_dim, percent);
            }
            std::cout << "Successfully read the transformed data" << std::endl;
        }
        else
        {
            std::cout << "Perform PCA transformation" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            pca_dim = 0;
            performPCA(pca_file_matrix, data, pca_data, pca_dim, percent);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "PCA dimension reduction completed, time taken: " << duration.count() << " ms" << std::endl;
            write_dvecs(pca_file_data.string(), pca_data);
        }
        Matrix<point_coord_type> pca_centroids(k);
        for (size_t i = 0; i < k; ++i)
        {
            pca_centroids[i] = pca_data[init_centroids[i]];
        }
        std::cout << std::fixed << std::setprecision(4) << "# Dimensions: " << rows << " x " << cols << ". # clusters: " << k << std::endl;
        auto end1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
        std::cout << "Preliminary work takes time: " << duration1.count() << " ms" << std::endl;
        std::cout << std::endl;
        // 打印表头
        size_t width_dist = std::max<size_t>(11, static_cast<size_t>(std::to_string(rows * k * 600).length())) + 1;
        size_t width_feature = std::max<size_t>(13, static_cast<size_t>(std::to_string(cols * rows * k * 1200).length())) + 1;
        std::cout << std::left << std::setw(29) << "Algorithm" << std::setw(7) << "#Iter" << std::setw(width_dist)
                  << "#Distance" << std::setw(10) << "Time(ms)" << std::setw(12)
                  << "Mem(Mb)" << std::setw(16)
                  << "Time(ms)/iter" << std::setw(width_feature) << "#Feature" << std::endl;
        std::cout << std::string(71 + width_dist + width_feature, '-') << std::endl;

        if (false)
        {
            Matrix<point_coord_type> rot_data;
            random_orthogonal_transform(data, rot_data, rs_seed);
            auto var_vec = compute_dimension_variance(rot_data);
            // for (size_t it = 0; it < cols; it++)
            // {
            //     std::cout << var_vec[it] << " ";
            // }
            // std::cout << std::endl;

            size_t num_pairs = 10000;
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_int_distribution<size_t> dist_1(0, rows - 1);
            std::vector<double> error1(cols, 0.0), error2(cols, 0.0);
            for (size_t it = 0; it < num_pairs; ++it)
            {
                size_t i, j;
                do
                {
                    i = dist_1(gen);
                    j = dist_1(gen);
                } while (i == j); // 避免同一个样本配成一对
                double true_dist = euclidean_dist_square(data[i], data[j]);
                for (size_t jt = 0; jt < cols; jt++)
                {
                    error1[jt] += true_dist - lower_bound_transform(pca_data[i], pca_data[j], jt);
                    error2[jt] += true_dist - lower_bound_transform(rot_data[i], rot_data[j], jt);
                }
            }
            for (size_t jt = 0; jt < cols; jt++)
            {
                std::cout << jt << " " << error1[jt] / num_pairs << " " << error2[jt] / num_pairs << std::endl;
            }
        }

        point_coord_type perc_bv = 0.3;
        data = Matrix<point_coord_type>(data.begin(), data.end());
        pca_data = Matrix<point_coord_type>(pca_data.begin(), pca_data.end());
        point_coord_type data_memory_usage = 1.0 * getMatrixMemoryBytes(data) / (1024.0 * 1024.0);
        point_coord_type pca_data_memory_usage = 1.0 * getMatrixMemoryBytes(pca_data) / (1024.0 * 1024.0);
        if (alg == "")
        {
            {
                KMeans kmeans(k);
                run_and_report("K-means", kmeans, [&](auto &a)
                            { a.setInitialCentroids(initial_centroids); }, [&](auto &a)
                            { a.fit(data); }, data_memory_usage, width_dist, width_feature);
                // saveLabelsToFile(kmeans.getLabels(), "../result/kmeans_labels.txt");
            }

            {
                HamerlyKmeans hamerlyKmeans(k);
                run_and_report("Hamerly K-means", hamerlyKmeans, [&](auto &a)
                            { a.setInitialCentroids(initial_centroids); }, [&](auto &a)
                            { a.fit(data); }, data_memory_usage, width_dist, width_feature);
                // saveLabelsToFile(hamerlyKmeans.getLabels(), "../result/hamerly_kmeans_labels.txt");
            }
            
            {
                HeapKmeans heapKmeans;
                run_and_report("Heap K-means", heapKmeans, [&](auto &a)
                            { a.initialize(data, initial_centroids, k); }, [&](auto &a)
                            { a.fit(); }, data_memory_usage, width_dist, width_feature);
                // saveLabelsToFile(heapKmeans.getLabels(), "../result/heap_kmeans_labels.txt");
            }

            {
                ExponionKmeans expKmeansNS(k);
                run_and_report("Exponion K-means(ns)", expKmeansNS, [&](auto &a)
                            { a.setInitialCentroids(initial_centroids); }, [&](auto &a)
                            { a.fit_ns(data); }, data_memory_usage, width_dist, width_feature);
                // saveLabelsToFile(expKmeansNS.getLabels(), "../result/exp_kmeans_labels.txt");
            }

            {
                BallKmeans ballKmeans(data, k);
                run_and_report("Ball K-means", ballKmeans, [&](auto &a)
                            { a.setInitialCentroids(initial_centroids); }, [&](auto &a)
                            { a.fit(); }, data_memory_usage, width_dist, width_feature);
                // saveLabelsToFile(ballKmeans.getLabels(), "../result/ball_kmeans_labels.txt");
            }

            {
                YinYangKmeans yykmeans(k);
                run_and_report("Yinyang K-means", yykmeans, [&](auto &a)
                            { a.setInitialCentroids(initial_centroids); }, [&](auto &a)
                            { a.fit(data); }, data_memory_usage, width_dist, width_feature);
                // saveLabelsToFile(yykmeans.getLabels(), "../result/yinyang_kmeans_labels.txt");
            }

            {
                BVKmeans bvkmeans(k);
                run_and_report("BV K-means", bvkmeans, [&](auto &a)
                            { a.setInitialCentroids(perc_bv, initial_centroids); }, [&](auto &a)
                            { a.fit(data); }, data_memory_usage, width_dist, width_feature);
                // saveLabelsToFile(bvkmeans.getLabels(), "../result/bv_kmeans_labels.txt");
            }

            {
                ElkanKmeans elkanKmeans(k);
                run_and_report("Elkan K-means", elkanKmeans, [&](auto &a)
                            { a.setInitialCentroids(initial_centroids); }, [&](auto &a)
                            { a.fit(data); }, data_memory_usage, width_dist, width_feature);
                // saveLabelsToFile(elkanKmeans.getLabels(), "../result/elkan_kmeans_labels.txt");
            }

            {
                MarigoldKmeans marigold(k, data, initial_centroids);
                run_and_report("Marigold K-means", marigold, [](auto & /*unused*/) {}, [](auto &a)
                            { a.runKmeans(); }, data_memory_usage, width_dist, width_feature);
                // saveLabelsToFile(marigold.getLabels(), "../result/marigold_kmeans_labels.txt");
            }

            {
                AdaptiveKmeansV2 adaptivekmeansv2(k, ub);
                run_and_report("Adaptive K-means v2", adaptivekmeansv2, [&](auto &a)
                            { a.setInitialCentroids(initial_centroids); }, [&](auto &a)
                            { a.fit(data); }, data_memory_usage, width_dist, width_feature);
                // saveLabelsToFile(adaptivekmeansv2.getLabels(), "../result/adaptive_kmeans_labels.txt");
            }

            {
                AdaptiveKmeansV2PCA adaptivekmeansv2PCA(k, ub);
                run_and_report("Adaptive K-means v2(PCA)", adaptivekmeansv2PCA, [&](auto &a)
                            { a.setInitialCentroids(pca_centroids, pca_dim); }, [&](auto &a)
                            { a.fit(pca_data); }, pca_data_memory_usage, width_dist, width_feature);
                // saveLabelsToFile(adaptivekmeansv2PCA.getLabels(), "../result/adaptive_kmeans_v2_pca_labels.txt");
            }
        }
        else if (alg == "adaptive")
        {
            // {
            //     AdaptiveKmeansV2 adaptivekmeansv2(k, ub);
            //     run_and_report("Adaptive K-means v2", adaptivekmeansv2, [&](auto &a)
            //                 { a.setInitialCentroids(initial_centroids); }, [&](auto &a)
            //                 { a.fit(data); }, data_memory_usage, width_dist, width_feature);
            // }

            {
                AdaptiveKmeansV2PCA adaptivekmeansv2PCA(k, ub);
                run_and_report("Adaptive K-means v2(PCA)", adaptivekmeansv2PCA, [&](auto &a)
                            { a.setInitialCentroids(pca_centroids, pca_dim); }, [&](auto &a)
                            { a.fit(pca_data); }, pca_data_memory_usage, width_dist, width_feature);
            }
        }
        else if (alg == "exp")
        {
            ExponionKmeans expKmeans(k);
            run_and_report("Exponion K-means", expKmeans, [&](auto &a)
                           { a.setInitialCentroids(initial_centroids); }, [&](auto &a)
                           { a.fit(data); }, data_memory_usage, width_dist, width_feature);

            ExponionKmeans expKmeansNS(k);
            run_and_report("Exponion K-means(ns)", expKmeansNS, [&](auto &a)
                           { a.setInitialCentroids(initial_centroids); }, [&](auto &a)
                           { a.fit_ns(data); }, data_memory_usage, width_dist, width_feature);
        }
        else if (alg == "bv")
        {
            BVKmeans bvkmeans(k);
            run_and_report("BV K-means", bvkmeans, [&](auto &a)
                           { a.setInitialCentroids(perc_bv, initial_centroids); }, [&](auto &a)
                           { a.fit(data); }, data_memory_usage, width_dist, width_feature);
        }
        else if (alg == "heap")
        {
            HeapKmeans heapKmeans;
            run_and_report("Heap K-means", heapKmeans, [&](auto &a)
                           { a.initialize(data, initial_centroids, k); }, [&](auto &a)
                           { a.fit(); }, data_memory_usage, width_dist, width_feature);
        }
        else if (alg == "marigold")
        {
            MarigoldKmeans marigold(k, data, initial_centroids);
            run_and_report("Marigold K-means", marigold, [](auto & /*unused*/) {}, [](auto &a)
                           { a.runKmeans(); }, data_memory_usage, width_dist, width_feature);
        }
        else if (alg == "yinyang")
        {
            YinYangKmeans yykmeans(k);
            run_and_report("Yinyang K-means", yykmeans, [&](auto &a)
                           { a.setInitialCentroids(initial_centroids); }, [&](auto &a)
                           { a.fit(data); }, data_memory_usage, width_dist, width_feature);
        }
        else if (alg == "yinyang_pca")
        {
            YinYangKmeans yykmeans_pca(k);
            run_and_report("Yinyang K-means(PCA)", yykmeans_pca, [&](auto &a)
                           { a.setInitialCentroids(pca_centroids); }, [&](auto &a)
                           { a.fit_stepwise(pca_data, pca_dim); }, pca_data_memory_usage, width_dist, width_feature);
        }
        else if (alg == "elkan")
        {
            ElkanKmeans elkanKmeans(k);
            run_and_report("Elkan K-means", elkanKmeans, [&](auto &a)
                           { a.setInitialCentroids(initial_centroids); }, [&](auto &a)
                           { a.fit(data); }, data_memory_usage, width_dist, width_feature);
        }
        else if (alg == "elkan_pca")
        {
            ElkanKmeans elkanKmeansPCA(k);
            run_and_report("Elkan K-means(PCA)", elkanKmeansPCA, [&](auto &a)
                           { a.setInitialCentroids(pca_centroids, pca_dim); }, [&](auto &a)
                           { a.fit_stepwise(pca_data); }, pca_data_memory_usage, width_dist, width_feature);
        }
        else if (alg == "supp")
        {
            YinYangKmeans yykmeans_pca(k);
            run_and_report("Yinyang K-means(PCA)", yykmeans_pca, [&](auto &a)
                           { a.setInitialCentroids(pca_centroids); }, [&](auto &a)
                           { a.fit_stepwise(pca_data, pca_dim); }, pca_data_memory_usage, width_dist, width_feature);

            ElkanKmeans elkanKmeansPCA(k);
            run_and_report("Elkan K-means(PCA)", elkanKmeansPCA, [&](auto &a)
                           { a.setInitialCentroids(pca_centroids, pca_dim); }, [&](auto &a)
                           { a.fit_stepwise(pca_data); }, pca_data_memory_usage, width_dist, width_feature);
        }
        else if (alg == "lloyd")
        {
            KMeans kmeans(k);
            run_and_report("K-means", kmeans, [&](auto &a)
                           { a.setInitialCentroids(initial_centroids); }, [&](auto &a)
                           { a.fit(data); }, data_memory_usage, width_dist, width_feature);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
