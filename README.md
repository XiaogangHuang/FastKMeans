# FastKMeans



## Experimental Environment

All experiments are conducted on a single machine equipped with an Intel(R) Core(TM) i5-12600KF CPU and 32 GB of RAM, running Ubuntu 20.04. The algorithms are implemented in C++ and compiled with G++ 9.4.0 using the -O3 optimization flag. To ensure a fair comparison, all experiments are executed in a single-threaded environment without manual hardware-specific optimizations such as explicit SIMD instructions.

## Prerequisites

* cmake Eigen

## Building Instruction

```
mkdir build
cd build
cmake ..
make
```

## Usage


Some of the tested datasets are available at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/. 

1. Download the datasets.

2. Clustering
```
./bin/kmeans -f <filename> -s <seed> -k <cluster_num>
```
