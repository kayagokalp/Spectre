all: cpu gpu sgpu

cpu-debug:
	g++ -g -o f_cpuspec_layers.out src/spectra.cpp -std=c++11 -fopenmp -I/usr/include/mkl -I/data/aakyildiz/Apps/eigen/ -I/data/aakyildiz/Apps/spectra/include/ -DNDEBUG -lmkl_rt 
cpu:
	g++ -o f_cpuspec_layers.out src/spectra.cpp -std=c++11 -O3 -fopenmp -I/usr/include/mkl -I/data/aakyildiz/Apps/eigen/ -I/data/aakyildiz/Apps/spectra/include/ -DNDEBUG -lmkl_rt 
gpu:
	nvcc -ccbin g++ -o f_gpuspec_layers.out src/spectra.cpp -std=c++11 -O3  -Xcompiler -fopenmp  -I/usr/include/mkl -lmkl_rt -lcublas -lcusolver -DGPU -I/data/aakyildiz/Apps/eigen/ -I/data/aakyildiz/Apps/spectra/include/ -Xcompiler -DNDEBUG

sgpu:
	nvcc -ccbin g++ -o f_sgpuspec_layers.out src/spectra.cpp -std=c++11 -O3  -Xcompiler -fopenmp  -I/usr/include/mkl -lmkl_rt -lcublas -lcusolver -DSMART -I/data/aakyildiz/Apps/eigen/ -I/data/aakyildiz/Apps/spectra/include/ -Xcompiler -DNDEBUG




