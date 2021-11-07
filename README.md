# my-little-automl
Skeleton for routines ml actions automatization

## Support GPU for lightgbm

https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#id17

```bash
sudo apt install ocl-icd-opencl-dev
sudo apt install build-essential libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev
sudo apt install libboost-filesystem-dev
sudo apt install cmake
```

and then:

```bash
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build
cd build
cmake -DUSE_GPU=1 ..
# if you have installed NVIDIA CUDA to a customized location, you should specify paths to OpenCL headers and library like the following:
# cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
make -j4
```

and:

```bash
cd LightGBM/python-package
python setup.py install --gpu

```