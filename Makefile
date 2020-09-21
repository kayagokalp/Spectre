all: cpu gpu sgpu

cpu:
	g++ -o cpuspec.out src/spectra.cpp -std=c++11 -O3 -fopenmp -I/opt/intel/parallel_studio_xe_2020/compilers_and_libraries_2020/linux/mkl/include/ -I/home/kaya/Apps/eigen/ -I/home/kaya/Apps/spectra-0.9.0/include/ -DNDEBUG -lmkl_rt 
gpu:
	nvcc -o gpuspec.out src/spectra.cpp -std=c++11 -O3  -Xcompiler -fopenmp  -I/opt/intel/parallel_studio_xe_2020/compilers_and_libraries_2020/linux/mkl/include/ -lmkl_rt -lcublas -lcusolver -DGPU -I/home/kaya/Apps/eigen/ -I/home/kaya/Apps/spectra-0.9.0/include/ -Xcompiler -DNDEBUG

sgpu:
	nvcc -o gpuspecs.out src/spectra.cpp -std=c++11 -O3  -Xcompiler -fopenmp  -I/opt/intel/parallel_studio_xe_2020/compilers_and_libraries_2020/linux/mkl/include/ -lmkl_rt -lcublas -lcusolver -DSMART -I/home/kaya/Apps/eigen/ -I/home/kaya/Apps/spectra-0.9.0/include/ -Xcompiler -DNDEBUG



