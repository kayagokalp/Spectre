all: cpu gpu sgpu

cpu-debug:
	g++ -g -o f_cpuspec_layers.out src/spectra.cpp -std=c++11 -fopenmp -I/truba/sw/centos7.9/app/gromacs/2021.2-impi-mkl-oneapi-2021.2-GOLD/include/ -I/truba/home/kgokalp/truba-home-docs/eigen/ -I/truba/home/kgokalp/truba-home-docs/spectra/include/ -DNDEBUG -lmkl_rt 
cpu:
	g++ -o f_cpuspec_layers.out src/*.cpp -std=c++11 -O3 -fopenmp -I${MKLROOT}/include/ -I/truba/home/kgokalp/truba-home-docs/eigen/ -I/truba/home/kgokalp/truba-home-docs/spectra/include/ -DNDEBUG -lmkl_rt 
gpu:
	nvcc -ccbin g++ -o f_gpuspec_layers.out src/*.cpp -std=c++11 -O3  -Xcompiler -fopenmp  -I${MKLROOT}/include/ -lmkl_rt -lcublas -lcusolver -DGPU -I/truba/home/kgokalp/truba-home-docs/eigen/ -I/truba/home/kgokalp/truba-home-docs/spectra/include/ -Xcompiler -DNDEBUG 
gpu-debug:
	nvcc -ccbin g++ -o f_gpuspec_layers.out src/spectra.cpp -std=c++11 -g  -Xcompiler -fopenmp  -I/usr/include/mkl -lmkl_rt -lcublas -lcusolver -DGPU -I/data/aakyildiz/Apps/eigen/ -I/data/aakyildiz/Apps/spectra/include/ -Xcompiler -DNDEBUG

sgpu:
	nvcc -ccbin g++ -o f_sgpuspec_layers.out src/spectra.cpp -std=c++11 -O3  -Xcompiler -fopenmp  -I${MKLROOT}/include/ -lmkl_rt -lcublas -lcusolver -DSMART -I/truba/home/kgokalp/truba-home-docs/eigen/ -I/truba/home/kgokalp/truba-home-docs/spectra/include/ -Xcompiler -DNDEBUG




