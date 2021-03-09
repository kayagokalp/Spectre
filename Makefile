all: cpu gpu sgpu

cpu:
	sudo /gandalf/apps/gcc/8.2.0/bin/g++ -o cpuspec_layers.out src/spectra.cpp -std=c++11 -O3 -fopenmp -I/opt/intel/parallel_studio_xe_2020/compilers_and_libraries_2020/linux/mkl/include/ -I/home/aakyildiz/Apps/eigen/ -I/home/aakyildiz/Apps/spectra/include/ -DNDEBUG -lmkl_rt 
gpu:
	sudo /usr/local/cuda-10.0/bin/nvcc -ccbin /gandalf/apps/gcc/8.2.0/bin/g++ -o gpuspec_layers.out src/spectra.cpp -std=c++11 -O3  -Xcompiler -fopenmp  -I/opt/intel/parallel_studio_xe_2020/compilers_and_libraries_2020/linux/mkl/include/ -lmkl_rt -lcublas -lcusolver -DGPU -I/home/aakyildiz/Apps/eigen/ -I/home/aakyildiz/Apps/spectra/include/ -Xcompiler -DNDEBUG

sgpu:
	sudo /usr/local/cuda-10.0/bin/nvcc -ccbin /gandalf/apps/gcc/8.2.0/bin/g++ -o sgpuspec_layers.out src/spectra.cpp -std=c++11 -O3  -Xcompiler -fopenmp  -I/opt/intel/parallel_studio_xe_2020/compilers_and_libraries_2020/linux/mkl/include/ -lmkl_rt -lcublas -lcusolver -DSMART -I/home/aakyildiz/Apps/eigen/ -I/home/aakyildiz/Apps/spectra/include/ -Xcompiler -DNDEBUG




