all: cpu gpu sgpu

cpu:
	g++ -o cpuspec_layers.out src/spectra.cpp -std=c++11 -O3 -fopenmp -I/usr/include/mkl -I/home/users/aakyildiz/Apps/eigen/ -I/home/users/aakyildiz/Apps/spectra/include/ -DNDEBUG -lmkl_rt 
gpu:
	nvcc -ccbin g++ -o gpuspec_layers.out src/spectra.cpp -std=c++11 -O3  -Xcompiler -fopenmp  -I/usr/include/mkl -lmkl_rt -lcublas -lcusolver -DGPU -I/home/users/aakyildiz/Apps/eigen/ -I/home/users/aakyildiz/Apps/spectra/include/ -Xcompiler -DNDEBUG

sgpu:
	nvcc -ccbin g++ -o sgpuspec_layers.out src/spectra.cpp -std=c++11 -O3  -Xcompiler -fopenmp  -I/usr/include/mkl -lmkl_rt -lcublas -lcusolver -DSMART -I/home/users/aakyildiz/Apps/eigen/ -I/home/users/aakyildiz/Apps/spectra/include/ -Xcompiler -DNDEBUG




