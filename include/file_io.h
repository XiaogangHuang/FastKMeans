#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <random>
#include <type_traits>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include "utils.h"

// 使用k-means++方法初始化聚类中心
template <typename T>
Matrix<T> initializeCentroidsKMeansPlusPlus(const Matrix<T> &data, std::vector<size_t> &init_centroids, size_t k, unsigned seed)
{
    const size_t dataset_rows = data.size();
    const size_t dataset_cols = data[0].size();

    // 初始化中心点矩阵
    Matrix<T> centroids(k, std::vector<T>(dataset_cols));
    std::vector<size_t> labels(dataset_rows, std::numeric_limits<size_t>::max());

    // 标记已选择的点
    std::vector<bool> flag(dataset_rows, false);

    // 初始化随机数生成器
    std::mt19937 gen(seed);
    std::uniform_int_distribution<size_t> dis(0, dataset_rows - 1);

    // 随机选择第一个中心点
    size_t initial_row = dis(gen);
    centroids[0] = data[initial_row];
    flag[initial_row] = true;
    init_centroids.push_back(initial_row);
    // 存储每个点到最近中心点的距离
    std::vector<T> nearest(dataset_rows, std::numeric_limits<T>::max());

    // 选择剩余的中心点
    for (size_t i = 0; i < k - 1; ++i)
    {
        std::vector<T> p(dataset_rows, 0);

        for (size_t j = 0; j < dataset_rows; j++)
        {
            T t_dist = euclidean_dist_square(data[j], centroids[i]);
            if (i == 0 && !flag[j])
            {
                nearest[j] = t_dist;
                labels[j] = 0;
            }
            else if (t_dist < nearest[j])
            {
                nearest[j] = t_dist;
                labels[j] = i;
            }
            p[j] = (j == 0) ? nearest[j] : p[j - 1] + nearest[j];
        }

        // Use uniform distribution to generate a random number [0, 1)
        std::uniform_real_distribution<T> real_distrib(0.0, 1.0);
        const T rand_num = real_distrib(gen);

        for (size_t j = 0; j < dataset_rows; j++)
        {
            p[j] = p[j] / p[dataset_rows - 1]; // 轮盘法选择下一个类中心
            if (rand_num < p[j])
            {
                centroids[i + 1] = data[j];
                flag[j] = true;
                nearest[j] = 0;
                labels[j] = i + 1;
                init_centroids.push_back(j);
                break;
            }
        }
    }

    for (size_t j = 0; j < dataset_rows; j++)
    {
        if (flag[j])
            continue;
        T t_dist = euclidean_dist_square(data[j], centroids[k - 1]);
        if (t_dist < nearest[j])
        {
            labels[j] = k - 1;
        }
    }

    // 将init_centroids保存到文件
    std::ofstream init_centroids_file("../Data/init_centroids.txt");
    for (size_t i = 0; i < init_centroids.size(); ++i)
    {
        init_centroids_file << init_centroids[i] << std::endl;
    }
    init_centroids_file.close();

    // // 将init_centroids保存到文件
    // std::ofstream init_centroids_file1("/home/hxg/Code/ball-k-means-master/data+centers/centroids/init_centroids.txt");
    // for (size_t i = 0; i < init_centroids.size(); ++i)
    // {
    //     for (size_t j = 0; j < dataset_cols - 1; j++)
    //     {
    //         init_centroids_file1 << data[init_centroids[i]][j] << ",";
    //     }
    //     init_centroids_file1 << data[init_centroids[i]][dataset_cols - 1] << std::endl;
    // }
    // init_centroids_file1.close();
    return centroids;
}

// 从文件读取矩阵数据
template <typename T>
Matrix<T> readMatrixFromFile(const char *filename)
{
    // 打开文件
    int fd = open(filename, O_RDONLY);
    if (fd == -1)
    {
        throw std::runtime_error("Unable to open file: " + std::string(filename));
    }

    // 获取文件大小
    off_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET); // 重置文件指针

    // 使用 mmap 将文件内容映射到内存
    char *data = (char *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED)
    {
        close(fd);
        throw std::runtime_error("Memory mapping failed");
    }
    close(fd);

    Matrix<T> matrix;
    char *ptr = data;
    bool first_line = true;
    size_t expected_cols = 0;

    while (ptr < data + file_size)
    {
        // 查找行结束符 '\n'
        char *line_end = (char *)memchr(ptr, '\n', data + file_size - ptr);
        if (!line_end)
        {
            // 处理最后一行（如果没有换行符）
            if (ptr < data + file_size)
            {
                line_end = data + file_size;
            }
            else
            {
                break;
            }
        }

        // 将一行的内容读取到 stringstream
        std::string line(ptr, line_end - ptr);
        std::stringstream ss(line);
        std::vector<T> row;
        std::string token;

        // 检查是否包含逗号
        bool has_comma = line.find(',') != std::string::npos;

        if (has_comma)
        {
            // 使用逗号分隔
            while (std::getline(ss, token, ','))
            {
                // 去除token前后的空格
                token.erase(0, token.find_first_not_of(" \t"));
                token.erase(token.find_last_not_of(" \t") + 1);
                if (!token.empty())
                {
                    try
                    {
                        row.push_back(static_cast<T>(std::stod(token)));
                    }
                    catch (const std::invalid_argument &e)
                    {
                        continue;
                    }
                }
            }
        }
        else
        {
            // 使用空格分隔
            while (ss >> token)
            {
                try
                {
                    row.push_back(static_cast<T>(std::stod(token)));
                }
                catch (const std::invalid_argument &e)
                {
                    continue;
                }
            }
        }

        // 第一次读取，记录列数
        if (first_line)
        {
            expected_cols = row.size();
            first_line = false;
        }

        // 如果行数据符合维度要求，则加入结果
        if (row.size() == expected_cols && !row.empty())
        {
            matrix.push_back(std::move(row));
        }

        ptr = line_end + 1; // 移动指针到下一行
    }

    // 解除内存映射
    munmap(data, file_size);

    if (matrix.empty())
    {
        throw std::runtime_error("The file is empty or has an incorrect format.");
    }

    return matrix;
}

// 从文件中读取初始类中心
template <typename T>
Matrix<T> readInitialCentroidsFromFile(const Matrix<T> &data, const std::string &filename, std::vector<size_t> &init_centroids)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file containing initial centroids: " + filename);
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        size_t id;
        if (ss >> id)
        {
            init_centroids.push_back(id);
        }
    }

    if (init_centroids.empty())
    {
        throw std::runtime_error("The initial centroids file is empty.");
    }
    Matrix<T> centroids(init_centroids.size());
    for (size_t i = 0; i < init_centroids.size(); i++)
    {
        centroids[i] = data[init_centroids[i]];
    }

    return centroids;
}

// 读取稀疏索引:值 格式文件，返回SparseMatrix
template <typename T>
SparseMatrix<T> readSparseMatrixFromFile(const char *filename)
{
    int fd = open(filename, O_RDONLY);
    if (fd == -1)
    {
        throw std::runtime_error("Unable to open file: " + std::string(filename));
    }

    off_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);

    char *data = (char *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED)
    {
        close(fd);
        throw std::runtime_error("Memory mapping failed");
    }
    close(fd);

    SparseMatrix<T> sparse_mat;
    char *ptr = data;
    size_t max_col_index = 0;
    size_t row_idx = 0;

    while (ptr < data + file_size)
    {
        char *line_end = (char *)memchr(ptr, '\n', data + file_size - ptr);
        if (!line_end)
        {
            line_end = data + file_size;
        }

        std::string line(ptr, line_end - ptr);
        std::stringstream ss(line);

        // 行号列号值的解析：第一列一般是label，跳过不读
        std::string first_token;
        ss >> first_token; // 跳过标签或行号等

        std::string token;
        while (ss >> token)
        {
            // token 格式形如 957:0.5162
            size_t pos = token.find(':');
            if (pos == std::string::npos)
                continue;

            try
            {
                size_t col_idx = std::stoul(token.substr(0, pos));
                T value = static_cast<T>(std::stod(token.substr(pos + 1)));

                sparse_mat.rows.push_back(row_idx);
                sparse_mat.cols.push_back(col_idx);
                sparse_mat.values.push_back(value);

                if (col_idx > max_col_index)
                    max_col_index = col_idx;
            }
            catch (...)
            {
                // 解析异常跳过
                continue;
            }
        }

        ++row_idx;
        ptr = line_end + 1;
    }

    munmap(data, file_size);

    sparse_mat.nrows = row_idx;
    sparse_mat.ncols = max_col_index + 1;

    if (sparse_mat.rows.empty())
    {
        throw std::runtime_error("The file is empty or has an incorrect format.");
    }

    return sparse_mat;
}

template <typename T>
Matrix<T> read_fvecs(const char *filename)
{
    static_assert(std::is_same<T, double>::value, "This version only supports reading into double");

    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Unable to open .fvecs file: " + std::string(filename));
    }

    Matrix<T> vectors;

    while (true)
    {
        int dim = 0;
        if (!file.read(reinterpret_cast<char *>(&dim), sizeof(int)))
            break;

        if (dim <= 0)
        {
            throw std::runtime_error("Illegal dimension value, read failed.");
        }

        std::vector<float> temp(dim);
        if (!file.read(reinterpret_cast<char *>(temp.data()), sizeof(float) * dim))
            break;
        std::vector<T> vec(temp.begin(), temp.end());
        vectors.push_back(std::move(vec));
    }

    return vectors;
}

template <typename T>
void write_fvecs(const std::string &filename, const Matrix<T> &data)
{
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                  "write_fvecs only supports float or double data types");

    std::ofstream out(filename, std::ios::binary);
    if (!out)
    {
        throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    for (const auto &vec : data)
    {
        int dim = static_cast<int>(vec.size());
        out.write(reinterpret_cast<const char *>(&dim), sizeof(int));

        // 若是 double 类型，需要先转换为 float（fvecs 是 float 格式）
        if constexpr (std::is_same<T, double>::value)
        {
            std::vector<float> float_vec(vec.begin(), vec.end());
            out.write(reinterpret_cast<const char *>(float_vec.data()), sizeof(float) * dim);
        }
        else
        {
            out.write(reinterpret_cast<const char *>(vec.data()), sizeof(float) * dim);
        }
    }
}

template <typename T>
Matrix<T> read_dvecs(const std::string &filename)
{
    static_assert(std::is_same<T, double>::value, "This function only supports double type.");

    std::ifstream in(filename, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("Unable to open .dvecs file: " + filename);
    }

    Matrix<T> result;

    while (in.peek() != EOF)
    {
        int dim = 0;
        if (!in.read(reinterpret_cast<char *>(&dim), sizeof(int)))
            break;

        std::vector<T> vec(dim);
        if (!in.read(reinterpret_cast<char *>(vec.data()), sizeof(double) * dim))
            break;

        result.push_back(std::move(vec));
    }

    return result;
}

template <typename T>
void write_dvecs(const std::string &filename, const Matrix<T> &data)
{
    static_assert(std::is_same<T, double>::value, "This function only supports double type.");

    std::ofstream out(filename, std::ios::binary);
    if (!out)
    {
        throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    for (const auto &vec : data)
    {
        int dim = static_cast<int>(vec.size());
        out.write(reinterpret_cast<const char *>(&dim), sizeof(int));
        out.write(reinterpret_cast<const char *>(vec.data()), sizeof(double) * dim);
    }
}

template <typename T>
void saveLabelsToFile(const std::vector<T> &labels, const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    for (const auto &label : labels)
    {
        file << label << " ";
    }
    file << std::endl;
}

template <typename T>
void writeMatrixToFile(const std::string &filename, const Matrix<T> &data, char delimiter = ' ')
{
    std::ofstream outFile(filename);
    if (!outFile.is_open())
    {
        throw std::runtime_error("Unable to open output file: " + filename);
    }

    for (const auto &row : data)
    {
        for (size_t i = 0; i < row.size(); ++i)
        {
            outFile << row[i];
            if (i + 1 < row.size())
                outFile << delimiter;
        }
        outFile << '\n';
    }

    outFile.close();
}
