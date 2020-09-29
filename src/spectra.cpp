#include <math.h>
#include <chrono>
#include <limits>
#include <algorithm>

#include "omp.h"
#include "consts.h"
#include "helpers.h"
#include "mkl.h"

#define CPUID 0
#define GPUID 1

class FGM;

#ifdef SMART
#define MAX_TASKS 4
#define MAX_GPUS 4
#define MAX_CPUS 32  
double gloads[MAX_GPUS];
double cloads[MAX_CPUS];

omp_lock_t glocks[MAX_GPUS];
omp_lock_t clocks[MAX_CPUS];

double gcosts[MAX_TASKS] = {1,1,2,1};
double ccosts[MAX_TASKS] = {8,2,10,1};

#define GPU_MULT 1
#endif

#ifdef GPU
#define MAX_THREADS_NO 128
cublasHandle_t handle[MAX_THREADS_NO];
cudaStream_t stream[MAX_THREADS_NO];
cusolverDnHandle_t dnhandle[MAX_THREADS_NO];

int ttt;
#pragma omp threadprivate(ttt)

struct rinfo {
  double* gpu_mem = nullptr;
  int gpu_id;
  int no_elements, no_bytes;

  rinfo(int gpu_id, int x, int y, int z) : gpu_id(gpu_id) {
    int nxyz = x * y * z;
    no_elements = 75 * nxyz * nxyz;
    no_bytes = no_elements * sizeof(double);
    gpuErrchk(cudaMalloc((void**)&gpu_mem, no_bytes));
   }
};

rinfo** rinfos;
#endif

struct Space {
public:
  dtype start;
  dtype end;
  int no_points;
  
  MatrixXd IT; 
  MatrixXd FT;
  MatrixXd D; 
  VectorXd s;
  MatrixXd V;
  MatrixXd Q1;

  Space(dtype start, dtype end, int no_points) 
    : start(start), end(end), no_points(no_points), 
      IT(no_points, no_points), FT(no_points, no_points), D(no_points, no_points), s(no_points) {
    discretize();
  }  

  void discretize() {
    cheb(no_points, IT, FT); 
    DBG(cout << "IT\n" << IT << endl;);
    DBG(cout << "FT\n" << FT << endl;);
    derivative(start, end, no_points, D); 
    DBG(cout << "D\n" << D << endl;);
    slobat(start, end, no_points, s); 
    DBG(cout << "s\n" << s << endl;);
    inner_product_helper(start, end, no_points, V);
    DBG(cout << "V\n" << V << endl;);
    Q1 = IT * D * FT;
    DBG(cout << "Q1\n" << Q1 << endl;);
  }
};

class Shape {
public:
  Shape(dtype x_dim, dtype y_dim, dtype z_dim,
	int x_sample, int y_sample, int z_sample,
	dtype xcurve = 0, dtype ycurve = 0) :  
    dim{x_dim, y_dim, z_dim}, curve{xcurve, ycurve},
    is_curved(~(xcurve == 0 && ycurve == 0)),
    spaces{Space(-x_dim/2, x_dim/2, x_sample), Space(-y_dim/2, y_dim/2, y_sample), Space(0, z_dim, z_sample)},
    xyz(x_sample * y_sample * z_sample), 
    VD(xyz, xyz), 
    QDx(xyz, xyz), QDy(xyz, xyz), QDz(xyz, xyz) {
      QDx.setZero();
      QDy.setZero();
      QDz.setZero();
      vector_map_nojac();
  }

  void vector_map_nojac() {
    int npx = spaces[0].no_points;
    int npy = spaces[1].no_points;
    int npz = spaces[2].no_points;

    int xyz = npx * npy * npz;
    MatrixXd VDx = MatrixXd::Zero(xyz, xyz); 
    MatrixXd VDy = MatrixXd::Zero(xyz, xyz);
    MatrixXd VDz = MatrixXd::Zero(xyz, xyz);

    for(int i = 1; i <= npx; i++) {
      for(int j = 1; j <= npy; j++) {
	for(int k = 1; k <= npz; k++) {
	  int I = ((i-1) * npy * npz) + ((j-1) * npz) + k;

	  for(int l = 1; l <= npx; l++) {
	    int J = ((l-1) * npy * npz) + ((j-1) * npz) + k;
	    VDx(J-1, I-1) += spaces[0].V(l-1, i-1);
            QDx(J-1, I-1) += spaces[0].Q1(l-1, i-1);
	  }

	  for(int l = 1; l <= npy; l++) {
	    int J = ((i-1) * npy * npz) + ((l-1) * npz) + k;
	    VDy(J-1, I-1) += spaces[1].V(l-1, j-1);
            QDy(J-1, I-1) += spaces[1].Q1(l-1, j-1);
	  }

	  for(int l = 1; l <= npz; l++) {
	    int J = ((i-1) * npy * npz) + ((j-1) * npz) + l;
	    VDz(J-1, I-1) += spaces[2].V(l-1, k-1);
	    QDz(J-1, I-1) += spaces[2].Q1(l-1, k-1);
	  }
	}
      }
    }

    VD = VDx * VDy * VDz;
  }
 
  const dtype dim[3];  
  const bool is_curved;
  const dtype curve[2];
  
  Space spaces[3];

  const int xyz;
  MatrixXd VD;
  MatrixXd QDx;
  MatrixXd QDy;
  MatrixXd QDz;
};

class Material {
public:
  Material(dtype _mod_elasticity, 
	   dtype _poisson_ratio, 
	   dtype _density)
    : mod_elasticity(_mod_elasticity), 
      poisson_ratio(_poisson_ratio), 
      density(_density) {}

  //member variables
  const dtype mod_elasticity;
  const dtype poisson_ratio;
  const dtype density;
};

//Functionally graded material
class FGM {
public:
  FGM(Shape& _shape, 
      Material& first, Material& second, 
      double _ctrl_y, double _ctrl_z) : 
    shape(_shape),
    ctrl_y(_ctrl_y), ctrl_z(_ctrl_z), 
    mats{first, second},
    np{_shape.spaces[0].no_points, _shape.spaces[1].no_points, _shape.spaces[2].no_points},
    nxyz(np[0] * np[1] * np[2]),
    mu(np[0], np[1], np[2]), 
    lame(np[0], np[1], np[2]),
    rho(np[0], np[1], np[2]),
    VD_mu(nxyz, nxyz),
    VD_lame(nxyz, nxyz),
    VD_rho(nxyz, nxyz),
    M(3 * nxyz, 3 * nxyz), 
    K(3 * nxyz, 3 * nxyz)
  {
    mu.setZero();
    lame.setZero();
    rho.setZero();
    VD_mu.setZero();
    VD_lame.setZero();
    VD_rho.setZero();
    M.setZero();
    K.setZero();
    FG_var_MT();
    inner_product();
  }

#ifdef GPU
  void T1_system_matrices_GPU() {
    int ub = 0;
    int ue = nxyz-1; 
    int vb = nxyz;
    int ve = 2 * nxyz - 1;
    int wb = 2 * nxyz;
    int we = 3 * nxyz - 1;

    M(seq(ub, ue), seq(ub, ue)) = VD_rho;
    M(seq(vb, ve), seq(vb, ve)) = VD_rho;
    M(seq(wb, we), seq(wb, we)) = VD_rho;

    MatrixXd epx = MatrixXd::Zero(nxyz, 3 * nxyz);
    epx(seq(0, nxyz - 1), seq(ub, ue)) = shape.QDx;
    MatrixXd epy = MatrixXd::Zero(nxyz, 3 * nxyz);
    epy(seq(0, nxyz - 1), seq(vb, ve)) = shape.QDy;
    MatrixXd epz = MatrixXd::Zero(nxyz, 3 * nxyz);
    epz(seq(0, nxyz - 1), seq(wb, we)) = shape.QDz;

    MatrixXd gammaxy = MatrixXd::Zero(nxyz, 3 * nxyz);
    gammaxy(seq(0, nxyz - 1), seq(ub, ue)) = shape.QDy;
    gammaxy(seq(0, nxyz - 1) ,seq(vb, ve)) = shape.QDx;
    MatrixXd gammayz = MatrixXd::Zero(nxyz, 3 * nxyz);
    gammayz(seq(0, nxyz - 1), seq(vb, ve)) = shape.QDz;
    gammayz(seq(0, nxyz - 1), seq(wb, we)) = shape.QDy;
    MatrixXd gammaxz = MatrixXd::Zero(nxyz, 3 * nxyz);
    gammaxz(seq(0, nxyz - 1), seq(ub, ue)) = shape.QDz;
    gammaxz(seq(0, nxyz - 1), seq(wb, we)) = shape.QDx;

    const double tu = 2.0;
    const double van = 1.0;
    const double ziro = 0.0;
    
    double *d_VD_lame, *d_VD_mu, *d_epx, *d_epy, *d_epz, *d_gammaxy, *d_gammayz, *d_gammaxz, *d_epxyz, *d_temp_K, *d_K, *d_temp, *gpu_mem;
    gpu_mem = rinfos[ttt]->gpu_mem;
    
    int nc = epx.cols();    
    d_VD_lame = gpu_mem; //1
    d_VD_mu = d_VD_lame + VD_lame.size();  //1
    d_epx = d_VD_mu + VD_mu.size(); //3
    d_epy = d_epx + epx.size(); //3
    d_epz = d_epy + epy.size(); //3
    d_gammaxy = d_epz + epz.size(); //3
    d_gammaxz = d_gammaxy + gammaxy.size(); //3
    d_gammayz = d_gammaxz + gammaxz.size(); //3
    d_epxyz = d_gammayz + gammayz.size(); //3
    d_temp_K = d_epxyz + epx.size(); //9
    d_K = d_temp_K + (nc * nc); //9
    d_temp = d_K + (nc * nc); //3

    cudaMemcpy(d_VD_lame, VD_lame.data(), VD_lame.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_VD_mu, VD_mu.data(), VD_mu.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_epx, epx.data(), epx.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_epy, epy.data(), epy.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_epz, epz.data(), epz.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gammaxy, gammaxy.data(), gammaxy.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gammaxz, gammaxz.data(), gammaxz.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gammayz, gammayz.data(), gammayz.size() * sizeof(double), cudaMemcpyHostToDevice);
								    
    cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, epx.rows(), epx.cols(), &van, d_epx, epx.rows(), &van, d_epy, epy.rows(), d_epxyz, epx.rows());
    cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, epx.rows(), epx.cols(), &van, d_epxyz, epx.rows(), &van, d_epz, epz.rows(), d_epxyz, epx.rows()); //(x + y + z)
    
    cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, VD_lame.rows(), nc, VD_lame.cols(), &van, d_VD_lame, VD_lame.rows(), d_epxyz, epx.rows(), &ziro, d_temp, VD_lame.rows()); //VD_lame * epxyz
    cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epx.rows(), &van, d_epxyz, epx.rows(), d_temp, VD_lame.rows(), &ziro, d_temp_K, nc); //epxyzT * VD_lame * epxyz
    cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &ziro, d_K, nc, &van, d_temp_K, nc, d_K, nc);

    cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, VD_mu.rows(), nc, VD_mu.cols(), &van, d_VD_mu, VD_mu.rows(), d_epx, epx.rows(), &ziro, d_temp, VD_mu.rows()); //VD_mu * epx
    cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epx.rows(), &van, d_epx, epx.rows(), d_temp, VD_mu.rows(), &ziro, d_temp_K, nc); //epxT * VD_mu * epx
    cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &tu, d_temp_K, nc, d_K, nc);
    
    cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, VD_mu.rows(), nc, VD_mu.cols(), &van, d_VD_mu, VD_mu.rows(), d_epy, epy.rows(), &ziro, d_temp, VD_mu.rows()); //VD_mu * epy
    cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epy.rows(), &van, d_epy, epy.rows(), d_temp, VD_mu.rows(), &ziro, d_temp_K, nc); //epyT * VD_mu * epy
    cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &tu, d_temp_K, nc, d_K, nc);
    
    cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, VD_mu.rows(), nc, VD_mu.cols(), &van, d_VD_mu, VD_mu.rows(), d_epz, epz.rows(), &ziro, d_temp, VD_mu.rows()); //VD_mu * epz
    cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epz.rows(), &van, d_epz, epz.rows(), d_temp, VD_mu.rows(), &ziro, d_temp_K, nc); //epzT * VD_mu * epz
    cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &tu, d_temp_K, nc, d_K, nc);
    
    cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, VD_mu.rows(), nc, VD_mu.cols(), &van, d_VD_mu, VD_mu.rows(), d_gammaxy, gammaxy.rows(), &ziro, d_temp, VD_mu.rows()); //VD_mu * gammaxy
    cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammaxy.rows(), &van, d_gammaxy, gammaxy.rows(), d_temp, VD_mu.rows(), &ziro, d_temp_K, nc); //gammaxy * VD_mu * gammaxy
    cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc);
    
    cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, VD_mu.rows(), nc, VD_mu.cols(), &van, d_VD_mu, VD_mu.rows(), d_gammaxz, gammaxz.rows(), &ziro, d_temp, VD_mu.rows()); //VD_mu * gammaxz
    cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammaxz.rows(), &van, d_gammaxz, gammaxz.rows(), d_temp, VD_mu.rows(), &ziro, d_temp_K, nc); //gammaxz * VD_mu * gammaxz
    cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc);
    
    cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, VD_mu.rows(), nc, VD_mu.cols(), &van, d_VD_mu, VD_mu.rows(), d_gammayz, gammayz.rows(), &ziro, d_temp, VD_mu.rows()); //VD_mu * gammayz
    cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammayz.rows(), &van, d_gammayz, gammayz.rows(), d_temp, VD_mu.rows(), &ziro, d_temp_K, nc); //gammayz * VD_mu * gammayz
    cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc);
    cudaStreamSynchronize(stream[ttt]);

    K = MatrixXd::Zero(nc, nc);
    cudaMemcpy(K.data(), d_K, nc * nc * sizeof(double), cudaMemcpyDeviceToHost);
  }
#endif
  
  void T1_system_matrices_CPU() {
    int ub = 0;
    int ue = nxyz-1; 
    int vb = nxyz;
    int ve = 2 * nxyz - 1;
    int wb = 2 * nxyz;
    int we = 3 * nxyz - 1;

    M(seq(ub, ue), seq(ub, ue)) = VD_rho;
    M(seq(vb, ve), seq(vb, ve)) = VD_rho;
    M(seq(wb, we), seq(wb, we)) = VD_rho;

    MatrixXd epx = MatrixXd::Zero(nxyz, 3 * nxyz);
    epx(seq(0, nxyz - 1), seq(ub, ue)) = shape.QDx;
    MatrixXd epy = MatrixXd::Zero(nxyz, 3 * nxyz);
    epy(seq(0, nxyz - 1), seq(vb, ve)) = shape.QDy;
    MatrixXd epz = MatrixXd::Zero(nxyz, 3 * nxyz);
    epz(seq(0, nxyz - 1), seq(wb, we)) = shape.QDz;

    MatrixXd gammaxy = MatrixXd::Zero(nxyz, 3 * nxyz);
    gammaxy(seq(0, nxyz - 1), seq(ub, ue)) = shape.QDy;
    gammaxy(seq(0, nxyz - 1) ,seq(vb, ve)) = shape.QDx;
    MatrixXd gammayz = MatrixXd::Zero(nxyz, 3 * nxyz);
    gammayz(seq(0, nxyz - 1), seq(vb, ve)) = shape.QDz;
    gammayz(seq(0, nxyz - 1), seq(wb, we)) = shape.QDy;
    MatrixXd gammaxz = MatrixXd::Zero(nxyz, 3 * nxyz);
    gammaxz(seq(0, nxyz - 1), seq(ub, ue)) = shape.QDz;
    gammaxz(seq(0, nxyz - 1), seq(wb, we)) = shape.QDx;


    MatrixXd epxyz = epx + epy + epz;
    K = (epxyz.transpose() * (VD_lame * epxyz)) +  
    2 * ((epx.transpose() * (VD_mu * epx)) +  
         (epy.transpose() * (VD_mu * epy)) +  
	 (epz.transpose() * (VD_mu * epz))) +
        (gammaxy.transpose() * (VD_mu * gammaxy)) + 
        (gammaxz.transpose() * (VD_mu * gammaxz)) +
	(gammayz.transpose() * (VD_mu * gammayz));
  }
  
  void T1_system_matrices() {
#ifdef SMART
    int decision = GPUID;
    int tid = omp_get_thread_num();
    int cid = sched_getcpu();
    int gid = rinfos[tid]->gpu_id;

    if(gloads[gid] > (cloads[cid] + ccosts[0]) * GPU_MULT) {
      decision = CPUID;
    }
    
    if(decision == GPUID) {            
      //cout << tid << " decision 1 GPU - " << gloads[gid] << " " << cloads[cid] << endl;

      omp_set_lock(&glocks[gid]);
      gloads[gid] += gcosts[0];
      omp_unset_lock(&glocks[gid]);

      T1_system_matrices_GPU();

      omp_set_lock(&glocks[gid]);
      gloads[gid] -= gcosts[0];
      omp_unset_lock(&glocks[gid]);
    } else {
      //cout << tid << " decision 1 CPU - " << gloads[gid] << " " << cloads[cid] << endl;

      omp_set_lock(&clocks[cid]);
      cloads[cid] += ccosts[0];
      omp_unset_lock(&clocks[cid]);

      T1_system_matrices_CPU();

      omp_set_lock(&clocks[cid]);
      cloads[cid] -= ccosts[0];
      omp_unset_lock(&clocks[cid]);
    }
#elif defined GPU
    T1_system_matrices_GPU();
#else
    T1_system_matrices_CPU();
#endif    
  }
  
  void T2_svd(MatrixXd &BC, MatrixXd &V) {
    JacobiSVD<MatrixXd> svd(BC, ComputeFullV);    
    V = svd.matrixV();    
  }

#ifdef GPU
  void T3_mul_inv_GPU(MatrixXd& a0, MatrixXd& P) {
    const double van = 1.0;
    const double ziro = 0.0;	 
    
    double* gpu_mem = rinfos[ttt]->gpu_mem;
    double *d_K, *d_M, *d_P, *d_K_phy, *d_M_phy, *d_a0, *d_temp, *d_M_phy_i, *d_work;
    int *d_pivot, *d_info, Lwork;

    int nc = P.cols();
    d_K = gpu_mem; //9
    d_M = d_K + (9 * nxyz * nxyz); //9
    d_P = d_M + (9 * nxyz * nxyz); //max 9
    d_K_phy = d_P + (3 * nxyz * nc); //max 9 
    d_M_phy = d_K_phy + (nc * nc);  //max 9
    d_a0 = d_M_phy + (nc * nc); //max 9
    d_temp = d_a0 + (nc * nc); //max 9
    d_M_phy_i = d_temp + (3 * nxyz * nc); //max 9
    d_pivot = (int *)(d_M_phy_i + (nc * nc)); //max 1
    d_info = d_pivot + nc; //max 1
    d_work = (double *)(d_info + nc); //max 1.
    
    cudaMemcpy(d_K, K.data(), K.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M.data(), M.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, P.data(), P.size() * sizeof(double), cudaMemcpyHostToDevice);
    MatrixXd Id = MatrixXd::Identity(nc, nc);
    cudaMemcpy(d_M_phy_i, Id.data(), Id.size() * sizeof(double), cudaMemcpyHostToDevice);

    cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, K.rows(), nc, K.cols(), &van, d_K, K.rows(), d_P, P.rows(), &ziro, d_temp, K.rows()); //alpha * K * P + beta * K
    cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, P.rows(), &van, d_P, P.rows(), d_temp, K.rows(), &ziro, d_K_phy, P.cols()); //Pt * K * P
    cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, M.rows(), nc, M.cols(), &van, d_M, M.rows(), d_P, P.rows(), &ziro, d_temp, M.rows()); //M * P
    cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, P.rows(), &van, d_P, P.rows(), d_temp, M.rows(), &ziro, d_M_phy, P.cols()); //Pt * M * P
    cusolverDnDgetrf_bufferSize (dnhandle[ttt], nc, nc, d_M_phy, nc, &Lwork); 
    cusolverDnDgetrf(dnhandle[ttt], nc, nc, d_M_phy, nc, d_work, d_pivot, d_info);
    cusolverDnDgetrs(dnhandle[ttt], CUBLAS_OP_N, nc, nc, d_M_phy, nc, d_pivot, d_M_phy_i, nc, d_info);
    cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, nc, &van, d_M_phy_i, nc, d_K_phy, nc, &ziro, d_a0, nc); //M_phy_i * K_phy
    cudaStreamSynchronize(stream[ttt]);

    MatrixXd at(nc, nc);
    a0 = at;
    cudaMemcpy(a0.data(), d_a0, nc * nc * sizeof(double), cudaMemcpyDeviceToHost);    
  }
#endif
  
  void T3_mul_inv_CPU(MatrixXd& a0, MatrixXd& P) {
    MatrixXd K_phy = P.transpose() * (K * P);
    MatrixXd M_phy = P.transpose() * (M * P);
    a0 = M_phy.inverse() * K_phy;
  }
  
  void T3_mul_inv(MatrixXd& a0, MatrixXd& P) {
#ifdef SMART
    int decision = GPUID;
    int tid = omp_get_thread_num();
    int cid = sched_getcpu();
    int gid = rinfos[tid]->gpu_id;

    if(gloads[gid] > (cloads[cid] + ccosts[2]) * GPU_MULT) {
      decision = CPUID;
    }
    
    if(decision == GPUID) {            
      omp_set_lock(&glocks[gid]);
      gloads[gid] += gcosts[1];
      omp_unset_lock(&glocks[gid]);

      T3_mul_inv_GPU(a0, P);

      omp_set_lock(&glocks[gid]);
      gloads[gid] -= gcosts[1];
      omp_unset_lock(&glocks[gid]);
    } else {
      omp_set_lock(&clocks[cid]);
      cloads[cid] += ccosts[1];
      omp_unset_lock(&clocks[cid]);

      T3_mul_inv_CPU(a0, P);

      omp_set_lock(&clocks[cid]);
      cloads[cid] -= ccosts[1];
      omp_unset_lock(&clocks[cid]);
    }    
#elif defined GPU
    T3_mul_inv_GPU(a0, P);
#else
    T3_mul_inv_CPU(a0, P);
#endif
  }
  
  void T4_eigen(MatrixXd& a0, int& nconv, double& small_eig){
    MatrixXd MM = a0;
    DenseGenRealShiftSolve<double> op(MM);
    GenEigsRealShiftSolver<double, LARGEST_MAGN, DenseGenRealShiftSolve<double> >  eigs(&op, 10, 50, 0);
    
    eigs.init();
    nconv = eigs.compute();
    
    Eigen::VectorXcd evalues;
    if(eigs.info() == SUCCESSFUL) {
      evalues = eigs.eigenvalues();
      small_eig = evalues(nconv-1).real();
    }
  }
  
  void compute(const int noeigs, const int ncv, int& nconv, double& small_eig, 
	       const double shift = 0.01, const int max_iter = -1, const double tol = -1) {

#ifdef OMP_TIMER    
    double t1t = omp_get_wtime();
#endif
    T1_system_matrices();
#ifdef OMP_TIMER    
    cout << "system-matrices: " << omp_get_wtime() - t1t << " secs " << endl;
#endif
      
    MatrixXd BC_3D_I = boundary_condition_3d(0, 0);
    MatrixXd BC_3D_II = boundary_condition_3d(0, 1);

    MatrixXd BC_1 = beta_matrix_3d(BC_3D_I, 0);
    MatrixXd BC_2 = beta_matrix_3d(BC_3D_II, 0);
    MatrixXd BC(BC_1.rows() + BC_2.rows(), BC_1.cols());
    BC << BC_1, BC_2;

    MatrixXd V;
#ifdef OMP_TIMER    
    double t2t = omp_get_wtime();
#endif
    T2_svd(BC, V);
#ifdef OMP_TIMER    
    cout << "SVD: " << omp_get_wtime() - t2t << " secs " << endl;
#endif

    MatrixXd P = V(seq(0, V.rows() - 1), seq(BC.rows(), BC.cols() - 1));
    MatrixXd a0;
#ifdef OMP_TIMER
    double t3t = omp_get_wtime();
#endif
    T3_mul_inv(a0, P);
#ifdef OMP_TIMER    
    cout << "Mul-and-Inv: " << omp_get_wtime()-t3t << " secs" << endl;
#endif

#ifdef OMP_TIMER
    double t4t = omp_get_wtime();
#endif
    T4_eigen(a0, nconv, small_eig);  
#ifdef OMP_TIMER    
  cout << "Eigen: " << omp_get_wtime() - t4t << " secs - nconv = " << nconv << endl;
#endif
  }
  
  MatrixXd beta_matrix_3d(MatrixXd& BC_3D, int xyz) {
    MatrixXd BC = MatrixXd::Zero(3 * nxyz / np[xyz], 3 * nxyz);
    int ids[3];
    for(int dim = 0; dim < 3; dim++) {
      for(int i = 0; i < np[0]; i++) {
	ids[0] = i;
	for(int j = 0; j < np[1]; j++) {
	  ids[1] = j;
	  for(int k = 0; k < np[2]; k++) {
	    ids[2] = k;

	    int idx = dim * (nxyz / np[xyz]);
	    if(xyz == 0) idx += j * np[2] + k; 
	    else if(xyz == 1) idx += i * np[2] + k;
	    else if(xyz == 2) idx += i * np[1] + j;
	    int idy = dim * nxyz + 
	      i * np[1] * np[2] + 
	      j * np[2] + 
	      k;

	    BC(idx, idy) = BC_3D(dim, ids[xyz]);

	  }
	}
      }
    }
    return BC;
  } 

  void FG_var_MT() {
    VectorXd& x = shape.spaces[0].s;
    VectorXd& y = shape.spaces[1].s;
    VectorXd& z = shape.spaces[2].s;
    
    double K_m = (mats[0].mod_elasticity / 3) / (1 - 2 * mats[0].poisson_ratio);
    double G_m = (mats[0].mod_elasticity / 2) / (1 + mats[0].poisson_ratio);

    double K_c = (mats[1].mod_elasticity / 3) / (1 - 2 * mats[1].poisson_ratio);
    double G_c = (mats[1].mod_elasticity / 2) / (1 + mats[1].poisson_ratio);

    double V_min = 0;
    double V_max = 1;

    //for matlab conversion - can be removed later
    double c = shape.dim[2];
    double b = shape.dim[1];
    double p = ctrl_z;
    double q = ctrl_y;
    double rho_m = mats[0].density;
    double rho_c = mats[1].density;
    
    for(int j = 0; j < np[1]; j++) {
      for(int k = 0; k < np[2]; k++) {
	//bu satirda funtion pointer olabilir
	double vcijk = V_min + (V_max-V_min) * pow((z(k)/c), p) * pow((0.5+y(j)/b), q);
	double vmijk  = 1 - vcijk;
	double rhotemp = (rho_c * vcijk) + (rho_m * vmijk);
	double K = K_m + (K_c - K_m) * vcijk / (1 + (1 - vcijk) * (3 * (K_c - K_m) / (3*K_m + 4*G_m)));
	double f1 = G_m*(9*K_m+8*G_m)/(6*(K_m+2*G_m));
	double G = G_m + (G_c-G_m) * vcijk/(1 + (1- vcijk)*( (G_c-G_m)/(G_m+f1)));
	double eijk = 9*K*G/(3*K+G);
	double poisijk = (3*K-2*G)/(2*(3*K+G));  
	double mutemp = eijk / (2 * (1 + poisijk));
	double lametemp = (2 * mutemp * poisijk) / (1 - 2 * poisijk);
	
	for(int i = 0; i < np[0]; i++) {
	  rho(i,j,k) = rhotemp; 
	  mu(i,j,k) = mutemp; 
	  lame(i,j,k) = lametemp;
	}
      }
    }
  }
  
  void tensor3(VectorXd& v_d3Nt, MatrixXd& Sst, int n,
	       Tensor<double, 3>& X) {
    for(int i = 0; i < n; i++) {
      for(int j = 0; j < n; j++) {
	for(int k = 0; k < n; k++) {
	  double xijk = 0;
	  for(int l = 0; l < 3 * n; l++) {
	    xijk += v_d3Nt(l) * Sst(l,i) *  Sst(l,j) *  Sst(l,k);
	  }
	  X(i, j, k) = xijk;
	}
      }
    }
  }

  void inner_helper(Tensor<double, 3>& Axyz, Tensor<double, 3>& Xadl, Tensor<double, 3>& Ybem, Tensor<double, 3>& Zcfn,
		    MatrixXd& VD) {
    double**** Xadmn;
    alloc4D(Xadmn, np[0], np[0], np[1], np[2]);
    for(int i = 0; i < np[0]; i++) {
      for(int j = 0; j < np[0]; j++) {
	for(int k = 0; k < np[1]; k++) {
	  for(int l = 0; l < np[2]; l++) {
	    double sum = 0;
	    for(int m = 0; m < np[0]; m++) {
	      sum += Xadl(i, j, m) * Axyz(m, k, l);
	    }
	    Xadmn[i][j][k][l] = sum;
	  }
	}
      }
    }
    
    double***** Gamma_adnbe;
    alloc5D(Gamma_adnbe, np[0], np[0], np[2], np[1], np[1]);
    for(int i = 0; i < np[0]; i++) {
      for(int j = 0; j < np[0]; j++) {
	for(int k = 0; k < np[2]; k++) {
          for(int l = 0; l < np[1]; l++) {
             for(int m = 0; m < np[1]; m++) {
	       double sum = 0;
	       for(int o = 0; o < np[1]; o++) {
		 sum += Xadmn[i][j][o][k] * Ybem(o, l, m);
	       }
	       Gamma_adnbe[i][j][k][l][m] = sum;
	     }
          }
        }
      }
    }
    

    for(int i = 0; i < np[0]; i++) {
      for(int j = 0; j < np[0]; j++) {
        for(int k = 0; k < np[1]; k++) {
          for(int l = 0; l < np[1]; l++) {
	    for(int m = 0; m < np[2]; m++) {
	      for(int o = 0; o < np[2]; o++) {
		double sum = 0;
		for(int v = 0; v < np[2]; v++) {
		  sum += Gamma_adnbe[i][j][v][k][l] * Zcfn(v, m, o);
		}
		int row = (i)*np[1]*np[2]+(k)*np[2] + m;
		int col = (j)*np[1]*np[2]+(l)*np[2] + o;                        

		VD(row, col) += sum;
	      }
	    }
	  }
	}
      }
    }

    free4D(Xadmn, np[0], np[0], np[1], np[2]);
    free5D(Gamma_adnbe, np[0], np[0], np[2], np[1], np[1]);
  }

  void inner_product() {
    MatrixXd IFT[3][3][2]; 
    for(int i = 0; i < 3; i++) { //xyz loop
      for(int j = 0; j < 3; j++) { //123 loop
	int sz = (j + 1) * np[i];
	IFT[i][j][0] = MatrixXd::Zero(sz, sz); 
	IFT[i][j][1] = MatrixXd::Zero(sz, sz);
	cheb(sz, IFT[i][j][0], IFT[i][j][1]);
      }
    }
    
    VectorXd v_d3N[3];
    for(int i = 0; i < 3; i++) {
      VectorXd temp = cheb_int(shape.spaces[i].start, shape.spaces[i].end, 3 * np[i]);
      v_d3N[i] = (temp.transpose() * IFT[i][2][1]).transpose();
    }


    MatrixXd Ss[3];
    for(int i = 0; i < 3; i++) {
      MatrixXd I = MatrixXd::Identity(np[i], np[i]);
      MatrixXd Z = MatrixXd::Zero(np[i], np[i]);
      MatrixXd C(3 * np[i], np[i]);
      C << I, Z, Z;
      Ss[i] = IFT[i][2][0] * C * IFT[i][0][1];
    }

    Tensor<double, 3> Xadl (np[0], np[0], np[0]); Xadl.setZero();
    tensor3(v_d3N[0], Ss[0], np[0], Xadl);
    Tensor<double, 3> Ybem(np[1], np[1], np[1]); Ybem.setZero();
    tensor3(v_d3N[1], Ss[1], np[1], Ybem);
    Tensor<double, 3> Zcfn(np[2], np[2], np[2]); Zcfn.setZero();
    tensor3(v_d3N[2], Ss[2], np[2], Zcfn);

    inner_helper(mu, Xadl, Ybem, Zcfn, VD_mu);
    inner_helper(rho, Xadl, Ybem, Zcfn, VD_rho);
    inner_helper(lame, Xadl, Ybem, Zcfn, VD_lame);
  }

  MatrixXd boundary_condition_3d(int xyz, int ol) {
    double bc[3] = {1,1,1}; 
    RowVectorXd e(np[xyz]); for(int i = 0; i < np[xyz]; i++) e(i) = 0; 
    if(ol == 0) {
      e(0) = 1.0;  
    } else {                                                                               
      e(np[xyz] - 1) = 1.0;      
    }
    MatrixXd BC(3, np[xyz]);
    BC << (bc[0] * e), (bc[1] * e), (bc[2] * e);
    return BC;
  }

  int np[3]; //not to do this everytime  
  int nxyz; //not to do this everytime

  Shape shape;
  Material mats[2];
  const double ctrl_y;
  const double ctrl_z;
  
  Tensor<double, 3> mu;
  Tensor<double, 3> lame; 
  Tensor<double, 3> rho; 

  MatrixXd VD_mu;
  MatrixXd VD_lame;
  MatrixXd VD_rho;

  MatrixXd M;
  MatrixXd K;
};

ostream& operator<<(ostream& os, const Space& spc) {
  os << spc.start << "\t" << spc.end << "\t" << spc.no_points;
  return os;
}

ostream& operator<<(ostream& os, const Material& mat) {
  os << mat.mod_elasticity << "\t" << mat.poisson_ratio << "\t" << mat.density;
  return os;
}

ostream& operator<<(ostream& os, const Shape& shp) {
  os << "\tDims  : " << shp.dim[0] << "\t" << shp.dim[1] << "\t" << shp.dim[2] << "\n" 
     << "\tCurved: " << shp.curve[0] << "\t" << shp.curve[1] << "\n" 
     << "\t\tX-space: " << shp.spaces[0] << "\n" 
     << "\t\tY-space: " << shp.spaces[1] << "\n" 
     << "\t\tZ-space: " << shp.spaces[2] << "\n";
  return os; 
}

ostream& operator<<(ostream& os, const FGM& fgm) {
  os  << "Shape -------------------------------------------\n"
     << fgm.shape 
     << "Materials ---------------------------------------\n" 
     << "\tMat 1: " << fgm.mats[0] << "\n" 
     << "\tMat 2: " << fgm.mats[1] << "\n" 
     << "Parameters --------------------------------------\n" 
     << "\tCtrl : " << fgm.ctrl_y << "\t" << fgm.ctrl_z << "\n";
  cout << "-------------------------------------------------\n"; 
  return os;
}

//Fiber induced composite
class FIC {};
//Laminated composite
class LCO {};

int main(int argc, char** argv) {
#ifndef GPU
  if(argc < 5) {
    cout << "Usage: " << argv[0] << " xdim ydim zdim nthreads " << endl;
    return 0;
  }
#else
 if(argc < 6) {
    cout << "Usage: " << argv[0] << " xdim ydim zdim nthreads ngpus" << endl;
    return 0;
  }
#endif
  
  Eigen::setNbThreads(1);
  
  mkl_set_num_threads_local(1);

  int xdim = atoi(argv[1]);
  int ydim = atoi(argv[2]);
  int zdim = atoi(argv[3]);
  int nthreads = atoi(argv[4]);
  omp_set_num_threads(nthreads);


#ifdef SMART
  for(int i = 0; i < MAX_GPUS; i++) gloads[i] = 0;
  for(int i = 0; i < MAX_CPUS; i++) cloads[i] = 0;
  for(int i = 0; i < MAX_GPUS; i++) omp_init_lock(&glocks[i]);
  for(int i = 0; i < MAX_CPUS; i++) omp_init_lock(&clocks[i]);
#endif

#ifdef GPU
  rinfos = new rinfo*[nthreads];

  int no_gpus = atoi(argv[5]);
  bool failed = false;
#pragma omp parallel num_threads(nthreads) 
  {
    ttt = omp_get_thread_num();
#pragma omp critical
    {
      gpuErrchk(cudaSetDevice((ttt % no_gpus)));
      //cudaSetDeviceFlags(cudaDeviceScheduleYield);
      cudaFree(0);

      int deviceID; cudaGetDevice(&deviceID);
      if(deviceID != (ttt % no_gpus)) {
	cout << "device ID is not equal to the given one " << endl;
	failed = true;
      }
      cudaDeviceProp prop; cudaGetDeviceProperties(&prop, deviceID);
      cout << "GPU " <<  deviceID << ": " << prop.name << " is assigned to " << ttt << endl;

      if(cublasCreate(&handle[ttt]) != CUBLAS_STATUS_SUCCESS) {
	std::cout << "blas handler initialization failed" << std::endl;
	failed = true;
      }

      if(cusolverDnCreate(&dnhandle[ttt]) != CUSOLVER_STATUS_SUCCESS) {
	std::cout << "solver handler initialization failed" << std::endl;
	failed = true;
      }
          
      cudaStreamCreate(&stream[ttt]);
      cublasSetStream(handle[ttt], stream[ttt]);
      cusolverDnSetStream(dnhandle[ttt], stream[ttt]);
      rinfos[ttt] = new rinfo(ttt % no_gpus, xdim, ydim, zdim);
    }
  }
  
  if(failed) {exit(1);}
#endif

  double min_ctrl_y = 0.1, max_ctrl_y = 0.40, min_ctrl_z = 0.1, max_ctrl_z = 0.40;
  double interval = 0.025;
  
  vector<pair<double, double> > problems;
  for(double cy = min_ctrl_y; cy <= max_ctrl_y; cy += interval) {
    for(double cz = min_ctrl_z; cz <= max_ctrl_z; cz += interval) {
      problems.push_back(make_pair(cy, cz));
    }
  }
  
  omp_set_num_threads(nthreads);
  cout << "No problems: " << problems.size() << endl;
  
  double smallest_mineig =  std::numeric_limits<double>::max();
  double best_y = -1, best_z = -1;
  double ostart = omp_get_wtime();

  Material first(1, 0.3, 1);
  Material second(200.0/70.0, 0.3, 5700.0/2702.0);
  Shape shape(2, 1, 0.3, xdim, ydim, zdim);

#pragma omp parallel for  schedule(dynamic, 1)
  //{
  for(int i = 0; i < problems.size(); i++) {
    double start = omp_get_wtime();
    FGM fgm(shape, first, second, problems[i].first, problems[i].second);
    int nconv;
    double mineig;
    fgm.compute(10, 60, nconv, mineig, 0.01, 100, 0.01);
    double end = omp_get_wtime();

#pragma omp critical    
    if(nconv > 0) {
      cout << i << " " << fgm.ctrl_y << " " << fgm.ctrl_z << " " << mineig << " " << end - start << endl;
      if(mineig < smallest_mineig) {
	smallest_mineig = mineig;
	best_y = fgm.ctrl_y;
	best_z = fgm.ctrl_z;
      }
    } else {
      cout << "No eigen: " << fgm.ctrl_y << " " << fgm.ctrl_z << " " << end - start << endl;
    }
  }
  //}
 cout << endl << "Result:" << endl;
 cout << "Smallest eig: " << smallest_mineig << " - (" << best_y << ", " << best_z << ")" << endl;
 double oend = omp_get_wtime();
 cout << "Total time: " << oend - ostart << endl;

#ifdef GPU
  for(int i = 0; i < nthreads; i++) {
    cudaStreamDestroy(stream[i]);
    cublasDestroy(handle[i]);
  }
#endif

  return 0;
}


  

  
