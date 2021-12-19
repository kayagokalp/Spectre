#ifndef __HELPERS_H__
#define __HELPERS_H__
#include <iostream>
#define EIGEN_USE_MKL_ALL
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Spectra/GenEigsRealShiftSolver.h>
#include <Eigen/SparseCore>
#include <Spectra/MatOp/DenseGenRealShiftSolve.h>
#include <Eigen/SVD>
#include <math.h>
#include "consts.h"
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <algorithm>  
using namespace Eigen;
using namespace Spectra;
using namespace std;

#ifdef GPU
#define MAX_THREADS_NO 128
extern cublasHandle_t handle[MAX_THREADS_NO];
extern cudaStream_t stream[MAX_THREADS_NO];
extern cusolverDnHandle_t dnhandle[MAX_THREADS_NO];
struct rinfo
{
	double *gpu_mem = nullptr;
	int gpu_id;
	unsigned long no_bytes;
	unsigned int no_elements;

	rinfo(int gpu_id, int x1, int y1, int z1, int x2, int y2, int z2, int x3, int y3, int z3)  : gpu_id(gpu_id)
	{
		
		int nnxyz = (3 * x1 * y1 * z1) + (3 * x2 * y2 * z2) + (3 * x3 * y3 * z3);
		no_elements = 10 * nnxyz * nnxyz;
		
		no_bytes =  no_elements * sizeof(double);
		cout<<"no_bytes "<<no_bytes<<endl;
		gpuErrchk(cudaMalloc((void**)&gpu_mem, no_bytes));
	}

};
#ifdef GPU
struct TaskMemoryReq{
	size_t T1;
	size_t T1_h;
	size_t T2;
	size_t T3;
};

inline void read_config(TaskMemoryReq &task_memory_reqs, string& filename){
	ifstream configFile(filename);
	string line = "";
	while(getline(configFile, line))
	{
		cout<<"line : "<<line<<endl;
    		istringstream linestream(line);
		string data;
		string type;
   		size_t val1;
		linestream>>type;
		linestream>>val1;
		cout<<"type " <<type <<endl;
		if(type == "T1:"){
			task_memory_reqs.T1 = val1; 	
		}else if(type == "T1_h:"){
			task_memory_reqs.T1_h = val1;
		}else if(type == "T2:"){
			task_memory_reqs.T2 = val1;
		}else if(type == "T3:"){
			task_memory_reqs.T3 = val1;
		}
		cout<<"val1 : "<<val1<<endl;
	}
}

class GPUManager {

	public:	
		size_t remaining_memory_;


	GPUManager(size_t remaining_memory, int device_id): remaining_memory_(remaining_memory), device_id_(device_id) {};
	GPUManager(): remaining_memory_(0), device_id_(0) {};

	bool check_for_task(size_t required_memory){
		bool isGPU = false;
		#pragma omp critical
		{
		if (remaining_memory_ >= required_memory){
			cout << "GPU " << device_id_ << " has "<<remaining_memory_ << " task asked for "<<required_memory<<endl;
			isGPU = true;
		}else{
			cout << "GPU " << device_id_ << " has "<<remaining_memory_ << " task asked for "<<required_memory<<endl;
			isGPU = false;
		}
		}
		return isGPU;
	}

	void* allocate_memory(size_t required_bytes, bool &success){
		void *memory = nullptr;
		#pragma omp critical
		{
		remaining_memory_ = max(((size_t)0), (remaining_memory_ - required_bytes));
		}
		return memory;
	}
	void free_memory(void* ptr_to_delete, long number_of_bytes){
		#pragma omp critical
		{
		remaining_memory_ += number_of_bytes;
		}
	}

	cublasHandle_t create_cublas_handle(int thread_id){
		#pragma omp critical 
		{
		if (cublasCreate(&handle_[thread_id]) != CUBLAS_STATUS_SUCCESS)
		{
			std::cout << "blas handler initialization failed" << std::endl;
		}
		cublasSetStream(handle_[thread_id], stream_[thread_id]);
		}
		return handle_[thread_id];
	}
	
	cudaStream_t create_cuda_stream(int thread_id){
		#pragma omp critical
		{
		cudaStreamCreate(&stream_[thread_id]);
		}
		return stream_[thread_id];
	}

	cusolverDnHandle_t create_cusolver_handle(int thread_id){
		#pragma omp critical
		{
		if (cusolverDnCreate(&dnhandle_[thread_id]) != CUSOLVER_STATUS_SUCCESS)
		{
			std::cout << "solver handler initialization failed" << std::endl;
		}
		cusolverDnSetStream(dnhandle_[thread_id], stream_[thread_id]);
		}
		return dnhandle_[thread_id];
	}

	void destroy_cublas_handle(int thead_id){
		#pragma omp critical
		cublasDestroy(handle_[thead_id]);
	}

	void destroy_cudastream(int thread_id){
		#pragma omp critical
		cudaStreamDestroy(stream_[thread_id]);
	}

	private:
		int device_id_;
		cublasHandle_t handle_[MAX_THREADS_NO];
		cudaStream_t stream_[MAX_THREADS_NO];
		cusolverDnHandle_t dnhandle_[MAX_THREADS_NO];
};
#endif
extern int ttt;
#pragma omp threadprivate(ttt)

inline static const char *_cudaGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_SUCCESS";

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";

        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";

        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

    }

    return "<unknown>";
}


inline void __cusolveSafeCall(cusolverStatus_t err, const char *file, const int line)
{
    if(CUSOLVER_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUSOLVE error in file '%s', line %d\n\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, \
                                _cudaGetErrorEnum(err)); \
        cudaDeviceReset(); assert(0); \
    }
}
inline void cusolveSafeCall(cusolverStatus_t err) { __cusolveSafeCall(err, __FILE__, __LINE__); }

extern rinfo **rinfos;
#endif

inline void outputMatrix(string& fileName,MatrixXd &matrixToOutput){
	ofstream matrixFile;
	matrixFile.open(fileName);
	matrixFile<<matrixToOutput.rows()<<"\n"<<matrixToOutput.cols()<<"\n";
	matrixFile<<matrixToOutput;
	matrixFile.close();	
}

inline void alloc4D(dtype**** &ptr, int n1, int n2, int n3, int n4) {
  ptr = new dtype***[n1];
  for(int i = 0; i < n1; i++) {
    ptr[i] = new dtype**[n2];
    for(int j = 0; j < n2; j++) {
      ptr[i][j] = new dtype*[n3];
      for(int k = 0; k < n3; k++) {
	ptr[i][j][k] = new dtype[n4];
      }
    }
  }
}

inline void free4D(dtype**** &ptr, int n1, int n2, int n3, int n4) {
  for(int i = 0; i < n1; i++) {
    for(int j = 0; j < n2; j++) {
      for(int k = 0; k < n3; k++) {
        delete [] ptr[i][j][k];
      }
      delete [] ptr[i][j];
    }
    delete [] ptr[i];
  }
  delete [] ptr;
}

inline void alloc5D(dtype***** &ptr, int n1, int n2, int n3, int n4, int n5) {
  ptr = new dtype****[n1];
  for(int i = 0; i < n1; i++) {
    alloc4D(ptr[i], n2, n3, n4, n5);
  }
}

inline void free5D(dtype***** &ptr, int n1, int n2, int n3, int n4, int n5) {
  for(int i = 0; i < n1; i++) {
    free4D(ptr[i], n2, n3, n4, n5);
  }
  delete [] ptr;
}

inline void alloc6D(dtype****** &ptr, int n1, int n2, int n3, int n4, int n5, int n6) {
  ptr = new dtype*****[n1];
  for(int i = 0; i < n1; i++) {
    alloc5D(ptr[i], n2, n3, n4, n5, n6);
  }
}

inline void free6D(dtype****** &ptr, int n1, int n2, int n3, int n4, int n5, int n6) {
  for(int i = 0; i < n1; i++) {
    free5D(ptr[i], n2, n3, n4, n5, n6);
  }
  delete [] ptr;
}


inline VectorXd cheb_di(dtype start, dtype end, int N) {
  VectorXd v_di(N);
  v_di.setZero();
  for(int i = 1; i <= N; i += 2) {
    v_di(i-1) = (end - start) / (1.0 - ((i-1) * (i-1)));
  }
  return v_di;
}

inline VectorXd cheb_int(dtype start, dtype end, int N) {
  dtype scale = 2 / (end - start);  
  VectorXd vecint(N);
  vecint.setZero();
  for(int k = 0; k < N; k += 2) {
    vecint(k) = 2 / ((1 - (k * k)) * scale);
  }
  return vecint;
}

inline void cheb(int N, 
	  MatrixXd &IT, MatrixXd &FT) {
  for(int i = 1; i <= N; i++) {
    for(int j = 1; j <= N; j++) {
      IT(i-1,j-1) = cos((j-1) * pi * (N - i)/(N - 1));
    } 
  }  
  FT = IT.inverse();

}
inline void derivative(dtype start, dtype end, int N,
		MatrixXd& D) {
  dtype scale = (end - start) / 2;
  int odd = (N % 2 == 1);
  
  int DN = N / 2;

  D.setZero();
  for(int i = 0; i < DN; i++) {
    D(0, 2*i + 1) = 2*i + 1;
  }
  

  for(int i = 2; i <= N; i++) {
    if(i % 2 == 0) {
      for(int j = 1; j < DN + odd; j++) { 
	D(i-1, 2*j) = 4 * j;
      } 
    } else {
      for(int j = 1; j < DN; j++) {
	D(i-1, 2*j + 1) = 2 * (2*j + 1);
      } 
    }
  }

  for(int i = 1; i < D.rows(); i++) {
    for(int j = 0; j < i; j++) {
      D(i, j) = 0;
    }
  }


  for(int i = 0; i < D.rows(); i++) {
    for(int j = i; j < D.cols(); j++) {
      D(i,j) /= scale;
    }
  }
}

inline VectorXd lobat(int N) {
  VectorXd x(N);
  int nm1 = N - 1;
  for(int i = 0; i < N; i++) {
    x(i) = sin((pi * ((-1 * nm1) + (2.0 * i))) / (2.0 * nm1));
  }
  return x;
}

inline void slobat(dtype start, dtype end, int N, 
	    VectorXd& s) {
  VectorXd lbt = lobat(N);
  dtype alpha = (end - start) / 2;
  dtype beta = (end + start) / 2;
  VectorXd z = VectorXd::Constant(N, 1);; 
  s  = alpha * lbt + beta * z;
}

inline void inner_product_helper(dtype start, dtype end, int N,
			  MatrixXd& V) {
  int np2 = 2 * N;
  MatrixXd IT2(np2, np2); IT2.setZero();
  MatrixXd FT2(np2, np2); FT2.setZero();
  cheb(np2, IT2, FT2);

  MatrixXd IT(N, N); IT.setZero();
  MatrixXd FT(N, N); FT.setZero();
  cheb(N, IT, FT);

  VectorXd v_di = cheb_di(start, end, np2);
  MatrixXd v_d2N = (v_di.transpose() * FT2).asDiagonal(); 

  MatrixXd merged(2 * N, N);
  merged << MatrixXd::Identity(N,N), MatrixXd::Zero(N,N);
  MatrixXd S = IT2 * merged * FT;
  V = S.transpose() * v_d2N * S;
}
#endif
