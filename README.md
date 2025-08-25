# FastKMeans

FastKMeans is a high-performance implementation of the Lloyd's algorithm.

## Experimental Environment

All experiments are conducted on a single machine equipped with an **Intel(R) Core(TM) i5-12600KF CPU** and **32 GB RAM**, running **Ubuntu 20.04**.  
The algorithms are implemented in **C++** and compiled with **G++ 9.4.0** using the `-O3` optimization flag.  
To ensure fair comparison, all experiments are executed in a **single-threaded environment** without manual hardware-specific optimizations (e.g., explicit SIMD instructions).

## Prerequisites

- `cmake`
- `Eigen`

## Build Instructions

```bash
mkdir build
cd build
cmake ..
make
```

The executable will be generated under `./bin/`.


## Compared algorithms
We compare the following algorithms:
| __Algorithm__ | __Paper__ |
|-------------|------------|
|Lloyd  |   Least squares quantization in PCM|
|Hamerly|Making k-means even faster|
|Heap|Accelerating Lloyd’s algorithm for k-means clustering|
|Exponion|Fast k-means with accurate bounds|
|Ball|Ball k-means: Fast adaptive clustering with no bounds|
|Yinyang|Yinyang k-means: A drop-in replacement of the classic k-means with consistent speedup|
|Block|Speeding up k-means by approximating Euclidean distances via block vectors|
| Elkan         | Using the triangle inequality to accelerate k-means|
| Marigold         | Marigold: Efficient k-means clustering in high dimensions|


## Dataset Format

The dataset file should be a plain text file, where each line represents a data point.  
Features should be separated by spaces. Example:

```
5.1 3.5 1.4 0.2
4.9 3.0 1.4 0.2
6.2 3.4 5.4 2.3
```


## Usage

Some of the benchmark datasets are available at:  
[https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)

1. Download the datasets.
2. Run the program with the following command:

```bash
./bin/kmeans -f <filename> -s <seed> -k <num_clusters>
```

## Example

```bash
./bin/kmeans -f Epileptic.txt -k 10
```
