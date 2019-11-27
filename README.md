Depth estimation from stereo images using Semi-Global Matching algorithm
![left](https://github.com/FinleyPan/sgm_cl/blob/master/data/left.png)
![right](https://github.com/FinleyPan/sgm_cl/blob/master/data/right.png)
![disparity](https://github.com/FinleyPan/sgm_cl/blob/master/data/disparity.png)

# Prerequisites
- CMake
- [OpenCV](https://github.com/opencv/opencv)
- OpenCL 1.2

# Build Step
```
$ git clone git@github.com:FinleyPan/sgm_cl.git sgm_cl
$ cd sgm_cl
$ mkdir build
$ cd build
$ cmake ..
$ make
```   

# Literature
*Hirschmuller, H. (2007). Stereo processing by semiglobal matching and mutual information. IEEE Transactions on pattern analysis and machine intelligence, 30(2), 328-341.*
