#pragma once
#include "helpers.h"
#include "Shape.h"


/**
 * @file fgm.h
 *
 * <h1> FGM </h1>
 *
 * Functionally Graded Material, represents a {@link FGM#num_shapes}
 * layered composite structure. Calculations are done in 4 steps: 
 * <ol>
 *	<li> T1 - System Matrices </li>
 *	<li> T2 - SVD decomposition </li>
 *	<li> T3 - Multiply and Inverse </li>
 *	<li> T4 - Eigen Decomposition </li>
 * </ol>
 *  
 *  @see Shape
 *
 * */
class FGM
{
	public:
		// int np[3]; //not to do this everytime
		// int nxyz;  //not to do this everytime
		unsigned int ** np;
		int * nxyz;

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

		FGM(unsigned int n_shapes, Shape* shps);
		~FGM();

#ifdef GPU
		/**
		 * <h1> T1 - System Matrices Honeycomb GPU </h1>
		 *  
		 * T1 - System Matrices Honeycomb GPU calculates K and M matrices for honeycomb layers in the FGM using GPU.
		 * @param l - id of the {@link Shape}
		 *
		 */
		void T1_system_matrices_honeycomb_GPU(unsigned int l);
		
		/**
		 * <h1> T1 - System Matrices GPU </h1>
		 *  
		 * T1 - System Matrices GPU calculates K and M matrices for non-honeycomb layers in the FGM using GPU.
		 * @param l - id of the {@link Shape}
		 *
		 */
		void T1_system_matrices_GPU(unsigned int l);
#endif

		/**
		 * <h1> T1 - System Matrices Honeycomb CPU </h1>
		 *  
		 * T1 - System Matrices Honeycomb CPU calculates K and M matrices for honeycomb layers in the FGM using CPU.
		 * @param l - id of the {@link Shape}
		 *
		 */
		void T1_system_matrices_honeycomb_CPU(unsigned int l);

		/**
		 * <h1> T1 - System Matrices CPU </h1>
		 *  
		 * T1 - System Matrices CPU calculates K and M matrices for non-honeycomb layers in the FGM using CPU.
		 * @param l - id of the {@link Shape}
		 *
		 */
		void T1_system_matrices_CPU(unsigned int l);

		/**
		 * <h1> T1 - System Matrices </h1>
		 *  
		 * T1 - System Matrices selects the correct T1 function depending on the following:
		 * 	<ul>
		 * 		<li>If SMART version is used, using the pre-calculated costs faster option is selected. </li>
		 * 		<li>If the layer that is being processed is a honeycombed layer (currently we are working on 3 layered FGMs in which the middle layer is always honeycomb) honeycomb version is called</li>
		 * 	</ul> 
		 * @param l - id of the {@link Shape}
		 *
		 */
		bool T1_system_matrices(unsigned int l);


		void T2_svd_CPU(MatrixXd &BC, MatrixXd &V);

#ifdef GPU
		void T2_svd_GPU(MatrixXd &BC, MatrixXd &V);
#endif

		bool T2_svd(MatrixXd &BC, MatrixXd &V);

#ifdef GPU
		void T3_mul_inv_GPU(MatrixXd &a0, MatrixXd &P);
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
