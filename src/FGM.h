#pragma once
#include "helpers.h"
#include "Shape.h"

//Functionally graded material
class FGM
{
	public:
		// int np[3]; //not to do this everytime
		// int nxyz;  //not to do this everytime
		unsigned int ** np;
		int * nxyz;
		
		int no_gpus;
		#ifdef GPU
		GPUManager *gpu_mans;
		TaskMemoryReq task_memory_reqs_;
		#endif
		// Shape shape;
		// Material mats[2];
		Shape * shapes;

		// const double ctrl_y;
		// const double ctrl_z;

		// Tensor<double, 3> mu;
		// Tensor<double, 3> lame;
		// Tensor<double, 3> rho;
		Tensor<double, 3> * rho;


		Tensor<double,3> * Q11T;
		Tensor<double,3> * Q12T;
		Tensor<double,3> * Q13T;
		Tensor<double,3> * Q14T;
		Tensor<double,3> * Q22T;
		Tensor<double,3> * Q23T;
		Tensor<double,3> * Q24T;
		Tensor<double,3> * Q33T;
		Tensor<double,3> * Q34T;
		Tensor<double,3> * Q44T;
		Tensor<double,3> * Q55T;
		Tensor<double,3> * Q56T;
		Tensor<double,3> * Q66T;

		Tensor<double,3> * JAC;
		// MatrixXd VD_mu;
		// MatrixXd VD_lame;
		// MatrixXd VD_rho;

		MatrixXd * VD_lame11; 
		MatrixXd * VD_lame22;
		MatrixXd * VD_lame33;
		MatrixXd * VD_lame12;
		MatrixXd * VD_lame13;
		MatrixXd * VD_lame23;
		MatrixXd * VD_lame44;
		MatrixXd * VD_lame55;
		MatrixXd * VD_lame66;
		MatrixXd * VD_lame14;
		MatrixXd * VD_lame24;
		MatrixXd * VD_lame34;
		MatrixXd * VD_lame56;
		MatrixXd * VD_ro;
		

		// MatrixXd M;
		// MatrixXd K;
		MatrixXd * M;
		MatrixXd * K;

		MatrixXd MM;
		MatrixXd KK;
		int nnxyz;

		int num_shapes;

		//FGM(): ctrl_y(0), ctrl_z(0){}
		#ifdef GPU
		FGM(unsigned int n_shapes, Shape* shps, GPUManager gpu_mans[] = nullptr, int no_gpus = 0, TaskMemoryReq task_memory_reqs = TaskMemoryReq{0,0,0,0});
		#endif
		#ifndef GPU
		FGM(unsigned int n_shapes, Shape* shps);
		#endif
		~FGM();
		// FGM(Shape &_shape,
		//     Material &first, Material &second,
		//     double _ctrl_y, double _ctrl_z) : shape(_shape),
		//                                       ctrl_y(_ctrl_y), ctrl_z(_ctrl_z),
		//                                       mats{first, second},
		//                                       np{_shape.spaces[0].no_points, _shape.spaces[1].no_points, _shape.spaces[2].no_points},
		//                                       nxyz(np[0] * np[1] * np[2]),
		//                                       mu(np[0], np[1], np[2]),
		//                                       lame(np[0], np[1], np[2]),
		//                                       rho(np[0], np[1], np[2]),
		//                                       VD_mu(nxyz, nxyz),
		//                                       VD_lame(nxyz, nxyz),
		//                                       VD_rho(nxyz, nxyz),
		//                                       M(3 * nxyz, 3 * nxyz),
		//                                       K(3 * nxyz, 3 * nxyz)
		// {
		//   mu.setZero();
		//   lame.setZero();
		//   rho.setZero();
		//   VD_mu.setZero();
		//   VD_lame.setZero();
		//   VD_rho.setZero();
		//   M.setZero();
		//   K.setZero();
		//   FG_var_MT()n
		// }
#ifdef GPU
		bool check_and_allocate_gpu(size_t required_memory, cudaStream_t &stream, cublasHandle_t &cublasHandle, int &device_id, void* &gpu_mem_beg);
		bool check_and_allocate_gpu_t2(size_t required_memory, cusolverDnHandle_t &cusolverHandle, cudaStream_t &stream, cublasHandle_t &cublasHandle, int &device_id, void* &gpu_mem_beg);
		bool check_and_allocate_gpu_in_task(size_t required_memory, int device_id, double* &gpu_mem_beg);
#endif
#ifdef GPU
		void T1_system_matrices_honeycomb_GPU(unsigned int l, double *gpu_mem_beg, cublasHandle_t &hndl, cudaStream_t &strm);
		
		void T1_system_matrices_GPU(unsigned int l, double * gpu_mem_beg, cublasHandle_t &cuHandle, cudaStream_t &cuStream);
#endif


		void T1_system_matrices_honeycomb_CPU(unsigned int l);
		
		void T1_system_matrices_CPU(unsigned int l);

		bool T1_system_matrices(unsigned int l);


		void T2_svd_CPU(MatrixXd &BC, MatrixXd &V);

#ifdef GPU
		//void T2_svd_GPU(MatrixXd &BC, MatrixXd &V);
		bool T2_svd_GPU(double *gpu_mem_beg, cusolverDnHandle_t &cuslv,cublasHandle_t &hndl, cudaStream_t &strm,MatrixXd &BC, MatrixXd &V,int device_id);
#endif

		bool T2_svd(MatrixXd &BC, MatrixXd &V);

#ifdef GPU
		bool T3_mul_inv_GPU(double *gpu_mem_beg,cusolverDnHandle_t &cuslv,cublasHandle_t &hndl, cudaStream_t &strm, MatrixXd &a0, MatrixXd &P,int device_id);
#endif

		void T3_mul_inv_CPU(MatrixXd &a0, MatrixXd &P);

		bool T3_mul_inv(MatrixXd &a0, MatrixXd &P);

		void T4_eigen(MatrixXd &a0, int &nconv, double &small_eig);
		

		void compute_gpu_costs(const int noeigs, const int ncv, int &nconv, double &small_eig, const double shift = 0.01, const int max_iter = -1, const double tol = -1, int g_id = 0, int sample_size = 1);

		void compute_cpu_costs(const int noeigs, const int ncv, int &nconv, double &small_eig,
				const double shift = 0.01, const int max_iter = -1, const double tol = -1, const int sample_size = 1);

		void removeDuplicateRows(MatrixXd &mat);

		MatrixXd removeZeroRows(MatrixXd &mat);

		void prepareBoundaryCondition(MatrixXd &mat, unsigned int l);

		void compute(const int noeigs, const int ncv, int &nconv, double &small_eig,
				const double shift = 0.01, const int max_iter = -1, const double tol = -1);

		MatrixXd beta_matrix_3d(MatrixXd &BC_3D, int xyz, unsigned int l);
		

		void FG_var_MT_CNT(unsigned int l);

		void FG_var_MT_honeycomb(unsigned int l);

		void tensor3(VectorXd &v_d3Nt, MatrixXd &Sst, int n,
				Tensor<double, 3> &X);

		void inner_helper(Tensor<double, 3> &Axyz, Tensor<double, 3> &Xadl, Tensor<double, 3> &Ybem, Tensor<double, 3> &Zcfn,
				MatrixXd &VD, unsigned int l);
		


		void inner_product_honeycomb(unsigned int l);
		
		void inner_product(unsigned int l);

		MatrixXd boundary_condition_3d(int xyz, int ol, unsigned int l);

};