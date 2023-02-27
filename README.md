# FT-BLAS: A High-Performance BLAS Implementation with Online Fault Tolerance.

This repository archives a fault-tolerant BLAS library. We adopt a novel hybrid fault-tolerant design that enables near-zero performance overhead due to fault tolerance.

## How to properly cite the work

Yujia Zhai, Elisabeth Giem, Quan Fan, Kai Zhao, Jinyang Liu, Zizhong Chen, FT-BLAS: a high performance BLAS implementation with online fault tolerance, Proceedings of THE 35th ACM International Conference on Supercomputing, Virtual Event, June, 2021.

```
@inproceedings{zhai2021ft,
  title={FT-BLAS: a high performance BLAS implementation with online fault tolerance},
  author={Zhai, Yujia and Giem, Elisabeth and Fan, Quan and Zhao, Kai and Liu, Jinyang and Chen, Zizhong},
  booktitle={Proceedings of the ACM International Conference on Supercomputing},
  pages={127--138},
  year={2021}
}
```

## How to build

To build the non-fault-tolerant library:

```
mkdir build && cd build
cmake .. && make -j
```

It is similar to build the fault-tolerant library:

```
mkdir build && cd build
cmake .. -DUSE_FAULT_TOLERANT=ON && make -j
```

One then should be able to explore the example binaries or one's own applications linked with FT-BLAS.

## Project scope

Currently we most emphasize the double-precision and include other data types in future roadmap. The table below summarizes the sub-routines currently supported by FT-BLAS. 

## Developers

Yujia Zhai, Vineeth Suvarna, Boyi Zhang 



