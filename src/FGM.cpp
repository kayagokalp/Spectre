#include "FGM.h"

FGM::~FGM(){
	delete[] np;
	delete[] nxyz;
	delete[] rho;
	delete[] JAC;
	delete[] Q11T;
	delete[] Q12T;
	delete[] Q13T;
	delete[] Q14T;
	delete[] Q22T;
	delete[] Q23T;
	delete[] Q24T;
	delete[] Q33T;
	delete[] Q34T;
	delete[] Q44T;
	delete[] Q55T;
	delete[] Q56T;
	delete[] Q66T;
	delete[] VD_lame11;
	delete[] VD_lame22;
	delete[] VD_lame33;
	delete[] VD_lame12;
	delete[] VD_lame13;
	delete[] VD_lame23;
	delete[] VD_lame44;
	delete[] VD_lame55;
	delete[] VD_lame66;
	delete[] VD_lame14;
	delete[] VD_lame24;
	delete[] VD_lame34;
	delete[] VD_lame56;
	delete[] VD_ro;
	delete[] M;
	delete[] K;
}

#ifndef GPU
FGM::FGM(unsigned int n_shapes, Shape* shps)
{
  num_shapes = n_shapes;
  shapes = shps;
  np = new unsigned int * [num_shapes];
  nxyz = new int[num_shapes];
  nnxyz = 0;
  rho = new Tensor<double, 3>[num_shapes];
  JAC = new Tensor<double,3>[num_shapes];
  Q11T = new Tensor<double,3>[num_shapes];
  Q12T = new Tensor<double,3>[num_shapes];
  Q13T = new Tensor<double,3>[num_shapes];
  Q14T = new Tensor<double,3>[num_shapes];
  Q22T = new Tensor<double,3>[num_shapes];
  Q23T = new Tensor<double,3>[num_shapes];
  Q24T = new Tensor<double,3>[num_shapes];
  Q33T = new Tensor<double,3>[num_shapes];
  Q34T = new Tensor<double,3>[num_shapes];
  Q44T = new Tensor<double,3>[num_shapes];
  Q55T = new Tensor<double,3>[num_shapes];
  Q56T = new Tensor<double,3>[num_shapes];
  Q66T = new Tensor<double,3>[num_shapes];


  VD_lame11 = new MatrixXd[num_shapes];
  VD_lame22 = new MatrixXd[num_shapes];
  VD_lame33 = new MatrixXd[num_shapes];
  
  VD_lame12 = new MatrixXd[num_shapes];
  VD_lame13 = new MatrixXd[num_shapes];
  VD_lame23 = new MatrixXd[num_shapes];

  VD_lame44 = new MatrixXd[num_shapes];
  VD_lame55 = new MatrixXd[num_shapes];
  VD_lame66 = new MatrixXd[num_shapes];

  VD_lame14 = new MatrixXd[num_shapes];
  VD_lame24 = new MatrixXd[num_shapes];
  VD_lame34 = new MatrixXd[num_shapes];
  VD_lame56 = new MatrixXd[num_shapes];
  VD_ro = new MatrixXd[num_shapes];

  M = new MatrixXd[num_shapes];
  K = new MatrixXd[num_shapes];
  for(unsigned int i = 0; i < n_shapes; i++){
    np[i] = new unsigned int[3];
    for(unsigned int j = 0; j < 3; j++){
      np[i][j] = shapes[i].spaces[j].no_points;
    }

    JAC[i] = shapes[i].jac;
    nxyz[i] = np[i][0] * np[i][1] * np[i][2];
    nnxyz += nxyz[i];

    rho[i] = Tensor<double, 3>(np[i][0], np[i][1], np[i][2]);
    rho[i].setZero();

    Q11T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q11T[i].setZero();
    Q12T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q12T[i].setZero();
    Q13T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q13T[i].setZero();
    Q14T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q14T[i].setZero();
    Q22T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q22T[i].setZero();
    Q23T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q23T[i].setZero();
    Q24T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q24T[i].setZero();
    Q33T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q33T[i].setZero();
    Q34T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q34T[i].setZero();
    Q44T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q44T[i].setZero();
    Q55T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q56T[i].setZero();
    Q66T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q66T[i].setZero();
    
    VD_lame11[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame11[i].setZero();
    VD_lame22[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame22[i].setZero();
    VD_lame33[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame33[i].setZero();


    VD_lame12[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame12[i].setZero();
    VD_lame13[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame13[i].setZero();
    VD_lame23[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame23[i].setZero();

    VD_lame44[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame44[i].setZero();
    VD_lame55[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame55[i].setZero();
    VD_lame66[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame66[i].setZero();

    VD_lame14[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame14[i].setZero();
    VD_lame24[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame24[i].setZero();
    VD_lame34[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame34[i].setZero();
    VD_lame56[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame56[i].setZero();

    VD_ro[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_ro[i].setZero();

    M[i] = MatrixXd(3 * nxyz[i], 3 * nxyz[i]);
    M[i].setZero();
    K[i] = MatrixXd(3 * nxyz[i], 3 * nxyz[i]);
    K[i].setZero();
    if(i != 1)
      FG_var_MT_CNT(i);
    else
      FG_var_MT_honeycomb(i);

    if(i != 1)
      inner_product(i);
    else
      inner_product_honeycomb(i);
  }

  //initialize MM and KK
  MM = MatrixXd(M[0].rows() + M[1].rows() + M[2].rows(), M[0].cols() + M[1].cols() + M[2].cols());
  MM.setZero();    
  KK = MatrixXd(K[0].rows() + K[1].rows() + K[2].rows(), K[0].cols() + K[1].cols() + K[2].cols());
  KK.setZero();
  //
}
#endif

#ifdef GPU
FGM::FGM(unsigned int n_shapes, Shape* shps, GPUManager gpu_mans[], int no_gpus, TaskMemoryReq task_memory_reqs)
{
  num_shapes = n_shapes;
  shapes = shps;
  task_memory_reqs_ = task_memory_reqs;
  np = new unsigned int * [num_shapes];
  nxyz = new int[num_shapes];
  nnxyz = 0;
  rho = new Tensor<double, 3>[num_shapes];
  JAC = new Tensor<double,3>[num_shapes];
  this->gpu_mans = gpu_mans; 
  this->no_gpus = no_gpus;
  Q11T = new Tensor<double,3>[num_shapes];
  Q12T = new Tensor<double,3>[num_shapes];
  Q13T = new Tensor<double,3>[num_shapes];
  Q14T = new Tensor<double,3>[num_shapes];
  Q22T = new Tensor<double,3>[num_shapes];
  Q23T = new Tensor<double,3>[num_shapes];
  Q24T = new Tensor<double,3>[num_shapes];
  Q33T = new Tensor<double,3>[num_shapes];
  Q34T = new Tensor<double,3>[num_shapes];
  Q44T = new Tensor<double,3>[num_shapes];
  Q55T = new Tensor<double,3>[num_shapes];
  Q56T = new Tensor<double,3>[num_shapes];
  Q66T = new Tensor<double,3>[num_shapes];


  VD_lame11 = new MatrixXd[num_shapes];
  VD_lame22 = new MatrixXd[num_shapes];
  VD_lame33 = new MatrixXd[num_shapes];
  
  VD_lame12 = new MatrixXd[num_shapes];
  VD_lame13 = new MatrixXd[num_shapes];
  VD_lame23 = new MatrixXd[num_shapes];

  VD_lame44 = new MatrixXd[num_shapes];
  VD_lame55 = new MatrixXd[num_shapes];
  VD_lame66 = new MatrixXd[num_shapes];

  VD_lame14 = new MatrixXd[num_shapes];
  VD_lame24 = new MatrixXd[num_shapes];
  VD_lame34 = new MatrixXd[num_shapes];
  VD_lame56 = new MatrixXd[num_shapes];
  VD_ro = new MatrixXd[num_shapes];

  M = new MatrixXd[num_shapes];
  K = new MatrixXd[num_shapes];
  for(unsigned int i = 0; i < n_shapes; i++){
    np[i] = new unsigned int[3];
    for(unsigned int j = 0; j < 3; j++){
      np[i][j] = shapes[i].spaces[j].no_points;
    }

    JAC[i] = shapes[i].jac;
    nxyz[i] = np[i][0] * np[i][1] * np[i][2];
    nnxyz += nxyz[i];

    rho[i] = Tensor<double, 3>(np[i][0], np[i][1], np[i][2]);
    rho[i].setZero();

    Q11T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q11T[i].setZero();
    Q12T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q12T[i].setZero();
    Q13T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q13T[i].setZero();
    Q14T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q14T[i].setZero();
    Q22T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q22T[i].setZero();
    Q23T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q23T[i].setZero();
    Q24T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q24T[i].setZero();
    Q33T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q33T[i].setZero();
    Q34T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q34T[i].setZero();
    Q44T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q44T[i].setZero();
    Q55T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q56T[i].setZero();
    Q66T[i] = Tensor<double,3>(np[i][0], np[i][1], np[i][2]);
    Q66T[i].setZero();
    
    VD_lame11[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame11[i].setZero();
    VD_lame22[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame22[i].setZero();
    VD_lame33[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame33[i].setZero();


    VD_lame12[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame12[i].setZero();
    VD_lame13[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame13[i].setZero();
    VD_lame23[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame23[i].setZero();

    VD_lame44[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame44[i].setZero();
    VD_lame55[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame55[i].setZero();
    VD_lame66[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame66[i].setZero();

    VD_lame14[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame14[i].setZero();
    VD_lame24[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame24[i].setZero();
    VD_lame34[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame34[i].setZero();
    VD_lame56[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_lame56[i].setZero();

    VD_ro[i] = MatrixXd(nxyz[i], nxyz[i]);
    VD_ro[i].setZero();

    M[i] = MatrixXd(3 * nxyz[i], 3 * nxyz[i]);
    M[i].setZero();
    K[i] = MatrixXd(3 * nxyz[i], 3 * nxyz[i]);
    K[i].setZero();
    if(i != 1)
      FG_var_MT_CNT(i);
    else
      FG_var_MT_honeycomb(i);

    if(i != 1)
      inner_product(i);
    else
      inner_product_honeycomb(i);
  }

  //initialize MM and KK
  MM = MatrixXd(M[0].rows() + M[1].rows() + M[2].rows(), M[0].cols() + M[1].cols() + M[2].cols());
  MM.setZero();    
  KK = MatrixXd(K[0].rows() + K[1].rows() + K[2].rows(), K[0].cols() + K[1].cols() + K[2].cols());
  KK.setZero();
  //
}
#endif


#ifdef GPU
bool FGM::T1_system_matrices_honeycomb_GPU(unsigned int l, std::vector<void*> &allocated_blocks, int device_id)
{
  cudaStream_t strm = gpu_mans[device_id].get_cuda_stream(ttt);
  cublasHandle_t hndl = gpu_mans[device_id].get_cublas_handle(ttt);
  bool success = false;
  gpuErrchkMem(cudaSetDevice(device_id),success);
  if(!success)
	  return false;
  int ub = 0;
  int ue = nxyz[l] - 1;
  int vb = nxyz[l];
  int ve = 2 * nxyz[l] - 1;
  int wb = 2 * nxyz[l];
  int we = 3 * nxyz[l] - 1;

  M[l](seq(ub, ue), seq(ub, ue)) = VD_ro[l];
  M[l](seq(vb, ve), seq(vb, ve)) = VD_ro[l];
  M[l](seq(wb, we), seq(wb, we)) = VD_ro[l];

  MatrixXd epx = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  epx(seq(0, nxyz[l] - 1), seq(ub, ue)) = shapes[l].QDx;
  MatrixXd epy = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  epy(seq(0, nxyz[l] - 1), seq(vb, ve)) = shapes[l].QDy;
  MatrixXd epz = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  epz(seq(0, nxyz[l] - 1), seq(wb, we)) = shapes[l].QDz;

  MatrixXd gammaxy = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  gammaxy(seq(0, nxyz[l] - 1), seq(ub, ue)) = shapes[l].QDy;
  gammaxy(seq(0, nxyz[l] - 1), seq(vb, ve)) = shapes[l].QDx;
  MatrixXd gammayz = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  gammayz(seq(0, nxyz[l] - 1), seq(vb, ve)) = shapes[l].QDz;
  gammayz(seq(0, nxyz[l] - 1), seq(wb, we)) = shapes[l].QDy;
  MatrixXd gammaxz = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  gammaxz(seq(0, nxyz[l] - 1), seq(ub, ue)) = shapes[l].QDz;
  gammaxz(seq(0, nxyz[l] - 1), seq(wb, we)) = shapes[l].QDx;

  const double tu = 2.0;
  const double van = 1.0;
  const double ziro = 0.0;

  double *d_VD_lame11, *d_VD_lame22, *d_VD_lame33, *d_VD_lame12, *d_VD_lame13, *d_VD_lame23, *d_VD_lame44, *d_VD_lame55, *d_VD_lame66, *d_VD_lame14, *d_VD_lame24, *d_VD_lame34, *d_VD_lame56, *d_VD_ro, *d_epx, *d_epy, *d_epz, *d_gammaxy, *d_gammayz, *d_gammaxz, *d_epxyz, *d_temp_K, *d_K, *d_temp, *gpu_mem_beg;
  int nc = epx.cols();

  size_t total_bytes = (VD_lame11[l].size() * sizeof(double)) + (VD_lame22[l].size()*sizeof(double)) + (VD_lame33[l].size() * sizeof(double)) + (VD_lame12[l].size()*sizeof(double)) + (VD_lame13[l].size()*sizeof(double)) + (VD_lame23[l].size() * sizeof(double)) + (VD_lame44[l].size() * sizeof(double)) + (VD_lame55[l].size() * sizeof(double)) + (VD_lame66[l].size() * sizeof(double)) + (VD_lame14[l].size() * sizeof(double)) + (VD_lame24[l].size() * sizeof(double)) + (VD_lame34[l].size() * sizeof(double)) + (VD_lame56[l].size() * sizeof(double)) + (VD_ro[l].size() * sizeof(double)) + (epx.size() * sizeof(double)) + (epy.size() * sizeof(double)) + (epz.size() * sizeof(double)) + (gammaxy.size() * sizeof(double)) + (gammaxz.size() * sizeof(double)) + (gammayz.size() * sizeof(double)) + (nc*nc*sizeof(double)) + (nc*nc*sizeof(double)) + (nc*nc*sizeof(double));

  gpuErrchkMem(cudaMalloc((void**)&gpu_mem_beg,total_bytes),success);                    //1
  if(!success)
	  return false;
  allocated_blocks.push_back(gpu_mem_beg);
  gpuErrchk(cudaMemset(gpu_mem_beg,0,total_bytes));

  d_VD_lame11 = gpu_mem_beg;                    //1
  d_VD_lame22 = d_VD_lame11 + VD_lame11[l].size();
  d_VD_lame33 = d_VD_lame22 + VD_lame22[l].size();
  d_VD_lame12 = d_VD_lame33 + VD_lame33[l].size();
  d_VD_lame13 = d_VD_lame12 + VD_lame12[l].size();
  d_VD_lame23 = d_VD_lame13 + VD_lame13[l].size();
  d_VD_lame44 = d_VD_lame23 + VD_lame23[l].size();
  d_VD_lame55 = d_VD_lame44 + VD_lame44[l].size();
  d_VD_lame66 = d_VD_lame55 + VD_lame55[l].size();
  d_VD_lame14 = d_VD_lame66 + VD_lame66[l].size();
  d_VD_lame24 = d_VD_lame14 + VD_lame14[l].size();
  d_VD_lame34 = d_VD_lame24 + VD_lame24[l].size();
  d_VD_lame56 = d_VD_lame34 + VD_lame34[l].size();
  d_VD_ro = d_VD_lame56 + VD_lame56[l].size();   //1
  d_epx = d_VD_ro + VD_ro[l].size();         //3
  d_epy = d_epx + epx.size();             //3
  d_epz = d_epy + epy.size();             //3
  d_gammaxy = d_epz + epz.size();         //3
  d_gammaxz = d_gammaxy + gammaxy.size(); //3
  d_gammayz = d_gammaxz + gammaxz.size(); //3
  d_temp_K = d_gammayz + gammayz.size();        //9
  d_K = d_temp_K + (nc * nc);             //9
  d_temp = d_K + (nc * nc);               //3

  cudaMemcpy(d_VD_lame11, VD_lame11[l].data(), VD_lame11[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame22, VD_lame22[l].data(), VD_lame22[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame33, VD_lame33[l].data(), VD_lame33[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame12, VD_lame12[l].data(), VD_lame12[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame13, VD_lame13[l].data(), VD_lame13[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame23, VD_lame23[l].data(), VD_lame23[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame44, VD_lame44[l].data(), VD_lame44[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame55, VD_lame55[l].data(), VD_lame55[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame66, VD_lame66[l].data(), VD_lame66[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame14, VD_lame14[l].data(), VD_lame14[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame24, VD_lame24[l].data(), VD_lame24[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame34, VD_lame34[l].data(), VD_lame34[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame56, VD_lame56[l].data(), VD_lame56[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_ro, VD_ro[l].data(), VD_ro[l].size() * sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_epx, epx.data(), epx.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_epy, epy.data(), epy.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_epz, epz.data(), epz.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gammaxy, gammaxy.data(), gammaxy.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gammaxz, gammaxz.data(), gammaxz.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gammayz, gammayz.data(), gammayz.size() * sizeof(double), cudaMemcpyHostToDevice);
  //start of (epx.transpose() * sigx)

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame11[l].rows(), nc, VD_lame11[l].cols(), &van, d_VD_lame11, VD_lame11[l].rows(), d_epx, epx.rows(), &ziro, d_temp, VD_lame11[l].rows()); //VD_lame11 * epx
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epx.rows(), &van, d_epx, epx.rows(), d_temp, VD_lame11[l].rows(), &ziro, d_temp_K, nc);//epxT * VD_lame11 * epx
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame12[l].rows(), nc, VD_lame12[l].cols(), &van, d_VD_lame12, VD_lame12[l].rows(), d_epy, epy.rows(), &ziro, d_temp, VD_lame12[l].rows()); //VD_lame12 * epy
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epx.rows(), &van, d_epx, epx.rows(), d_temp, VD_lame12[l].rows(), &ziro, d_temp_K, nc);//epxT * VD_lame12 * epy
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame13[l].rows(), nc, VD_lame13[l].cols(), &van, d_VD_lame13, VD_lame13[l].rows(), d_epz, epz.rows(), &ziro, d_temp, VD_lame13[l].rows()); //VD_lame13 * epz
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epx.rows(), &van, d_epx, epx.rows(), d_temp, VD_lame13[l].rows(), &ziro, d_temp_K, nc);//epxT * VD_lame13 * epz
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  //END OF (epx.transpose() * sigx)

  //start of (epy.transpose() * sigy)

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame12[l].rows(), nc, VD_lame12[l].cols(), &van, d_VD_lame12, VD_lame12[l].rows(), d_epx, epx.rows(), &ziro, d_temp, VD_lame12[l].rows()); //VD_lame12 * epx
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epy.rows(), &van, d_epy, epy.rows(), d_temp, VD_lame12[l].rows(), &ziro, d_temp_K, nc);//epyT * VD_lame12 * epx
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame22[l].rows(), nc, VD_lame22[l].cols(), &van, d_VD_lame22, VD_lame22[l].rows(), d_epy, epy.rows(), &ziro, d_temp, VD_lame22[l].rows()); //VD_lame22 * epy
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epy.rows(), &van, d_epy, epy.rows(), d_temp, VD_lame22[l].rows(), &ziro, d_temp_K, nc);//epyT * VD_lame12 * epy
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame23[l].rows(), nc, VD_lame23[l].cols(), &van, d_VD_lame23, VD_lame23[l].rows(), d_epz, epz.rows(), &ziro, d_temp, VD_lame23[l].rows()); //VD_lame23 * epz
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epy.rows(), &van, d_epy, epy.rows(), d_temp, VD_lame23[l].rows(), &ziro, d_temp_K, nc);//epyT * VD_lame23 * epz
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  //END OF (epy.transpose() * sigy)

  //start of (epz.transpose() * sigz)

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame13[l].rows(), nc, VD_lame13[l].cols(), &van, d_VD_lame13, VD_lame13[l].rows(), d_epx, epx.rows(), &ziro, d_temp, VD_lame13[l].rows()); //VD_lame13 * epx
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epz.rows(), &van, d_epz, epz.rows(), d_temp, VD_lame13[l].rows(), &ziro, d_temp_K, nc);//epzT * VD_lame13 * epx
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame23[l].rows(), nc, VD_lame23[l].cols(), &van, d_VD_lame23, VD_lame23[l].rows(), d_epy, epy.rows(), &ziro, d_temp, VD_lame23[l].rows()); //VD_lame23 * epy
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epz.rows(), &van, d_epz, epz.rows(), d_temp, VD_lame23[l].rows(), &ziro, d_temp_K, nc);//epzT * VD_lame23 * epy
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame33[l].rows(), nc, VD_lame33[l].cols(), &van, d_VD_lame33, VD_lame33[l].rows(), d_epz, epz.rows(), &ziro, d_temp, VD_lame33[l].rows()); //VD_lame33 * epz
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epz.rows(), &van, d_epz, epz.rows(), d_temp, VD_lame33[l].rows(), &ziro, d_temp_K, nc);//epzT * VD_lame33 * epz
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K


  //END OF (epz.transpose() * sigz)
  
  //start of (gammaxy.transpose() * tauxy)


  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame44[l].rows(), nc, VD_lame44[l].cols(), &van, d_VD_lame44, VD_lame44[l].rows(), d_gammaxy, gammaxy.rows(), &ziro, d_temp, VD_lame44[l].rows()); //VD_lame44 * gammaxy
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammaxy.rows(), &van, d_gammaxy, gammaxy.rows(), d_temp, VD_lame44[l].rows(), &ziro, d_temp_K, nc);//gammaxyT * VD_lame44 * gammaxy
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  //END OF (gammaxy.transpose() * tauxy)


  //start of (gammayz.transpose() * tauyz)
  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame55[l].rows(), nc, VD_lame55[l].cols(), &van, d_VD_lame55, VD_lame55[l].rows(), d_gammayz, gammayz.rows(), &ziro, d_temp, VD_lame55[l].rows()); //VD_lame55 * gammayz
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammayz.rows(), &van, d_gammayz, gammayz.rows(), d_temp, VD_lame55[l].rows(), &ziro, d_temp_K, nc);//gammayzT * VD_lame55 * gammayz
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc);//Add to K

  //END OF (gammayz.transpose() * tauyz)


  //start of (gammaxz.transpose() * tauxz)

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame66[l].rows(), nc, VD_lame66[l].cols(), &van, d_VD_lame66, VD_lame66[l].rows(), d_gammaxz, gammaxz.rows(), &ziro, d_temp, VD_lame66[l].rows()); //VD_lame66 * gammaxz
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammaxz.rows(), &van, d_gammaxz, gammaxz.rows(), d_temp, VD_lame66[l].rows(), &ziro, d_temp_K, nc);//gammaxzT * VD_lame66 * gammaxz
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc);//Add to K
  //END OF (gammaxz.transpose() * tauxz)

  cudaStreamSynchronize(strm);
  K[l] = MatrixXd::Zero(nc, nc);
  cudaMemcpy(K[l].data(), d_K, nc * nc * sizeof(double), cudaMemcpyDeviceToHost);
  clear_mem(allocated_blocks, device_id);
  return true;
}



bool FGM::T1_system_matrices_GPU(unsigned int l, std::vector<void*> &allocated_blocks, int device_id)
{
  cudaStream_t strm = gpu_mans[device_id].get_cuda_stream(ttt);
  cublasHandle_t hndl = gpu_mans[device_id].get_cublas_handle(ttt);
  bool success = false;
  gpuErrchkMem(cudaSetDevice(device_id),success);
  if(!success)
	  return false;
  int ub = 0;
  int ue = nxyz[l] - 1;
  int vb = nxyz[l];
  int ve = 2 * nxyz[l] - 1;
  int wb = 2 * nxyz[l];
  int we = 3 * nxyz[l] - 1;

  M[l](seq(ub, ue), seq(ub, ue)) = VD_ro[l];
  M[l](seq(vb, ve), seq(vb, ve)) = VD_ro[l];
  M[l](seq(wb, we), seq(wb, we)) = VD_ro[l];

  MatrixXd epx = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  epx(seq(0, nxyz[l] - 1), seq(ub, ue)) = shapes[l].QDx;
  MatrixXd epy = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  epy(seq(0, nxyz[l] - 1), seq(vb, ve)) = shapes[l].QDy;
  MatrixXd epz = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  epz(seq(0, nxyz[l] - 1), seq(wb, we)) = shapes[l].QDz;

  MatrixXd gammaxy = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  gammaxy(seq(0, nxyz[l] - 1), seq(ub, ue)) = shapes[l].QDy;
  gammaxy(seq(0, nxyz[l] - 1), seq(vb, ve)) = shapes[l].QDx;
  MatrixXd gammayz = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  gammayz(seq(0, nxyz[l] - 1), seq(vb, ve)) = shapes[l].QDz;
  gammayz(seq(0, nxyz[l] - 1), seq(wb, we)) = shapes[l].QDy;
  MatrixXd gammaxz = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  gammaxz(seq(0, nxyz[l] - 1), seq(ub, ue)) = shapes[l].QDz;
  gammaxz(seq(0, nxyz[l] - 1), seq(wb, we)) = shapes[l].QDx;

  const double tu = 2.0;
  const double van = 1.0;
  const double ziro = 0.0;

  double *d_VD_lame11, *d_VD_lame22, *d_VD_lame33, *d_VD_lame12, *d_VD_lame13, *d_VD_lame23, *d_VD_lame44, *d_VD_lame55, *d_VD_lame66, *d_VD_lame14, *d_VD_lame24, *d_VD_lame34, *d_VD_lame56, *d_VD_ro, *d_epx, *d_epy, *d_epz, *d_gammaxy, *d_gammayz, *d_gammaxz, *d_epxyz, *d_temp_K, *d_K, *d_temp, *gpu_mem_beg;
  int nc = epx.cols();

  size_t total_bytes = (VD_lame11[l].size() * sizeof(double)) + (VD_lame22[l].size()*sizeof(double)) + (VD_lame33[l].size() * sizeof(double)) + (VD_lame12[l].size()*sizeof(double)) + (VD_lame13[l].size()*sizeof(double)) + (VD_lame23[l].size() * sizeof(double)) + (VD_lame44[l].size() * sizeof(double)) + (VD_lame55[l].size() * sizeof(double)) + (VD_lame66[l].size() * sizeof(double)) + (VD_lame14[l].size() * sizeof(double)) + (VD_lame24[l].size() * sizeof(double)) + (VD_lame34[l].size() * sizeof(double)) + (VD_lame56[l].size() * sizeof(double)) + (VD_ro[l].size() * sizeof(double)) + (epx.size() * sizeof(double)) + (epy.size() * sizeof(double)) + (epz.size() * sizeof(double)) + (gammaxy.size() * sizeof(double)) + (gammaxz.size() * sizeof(double)) + (gammayz.size() * sizeof(double)) + (nc*nc*sizeof(double)) + (nc*nc*sizeof(double)) + (nc*nc*sizeof(double));

  gpuErrchkMem(cudaMalloc((void**)&gpu_mem_beg,total_bytes),success);                    //1
  if(!success)
	  return false;
  allocated_blocks.push_back(gpu_mem_beg);
  gpuErrchk(cudaMemset(gpu_mem_beg,0,total_bytes));

  d_VD_lame11 = gpu_mem_beg;
  d_VD_lame22 = d_VD_lame11 + VD_lame11[l].size();
  d_VD_lame33 = d_VD_lame22 + VD_lame22[l].size();
  d_VD_lame12 = d_VD_lame33 + VD_lame33[l].size();
  d_VD_lame13 = d_VD_lame12 + VD_lame12[l].size();
  d_VD_lame23 = d_VD_lame13 + VD_lame13[l].size();
  d_VD_lame44 = d_VD_lame23 + VD_lame23[l].size();
  d_VD_lame55 = d_VD_lame44 + VD_lame44[l].size();
  d_VD_lame66 = d_VD_lame55 + VD_lame55[l].size();
  d_VD_lame14 = d_VD_lame66 + VD_lame66[l].size();
  d_VD_lame24 = d_VD_lame14 + VD_lame14[l].size();
  d_VD_lame34 = d_VD_lame24 + VD_lame24[l].size();
  d_VD_lame56 = d_VD_lame34 + VD_lame34[l].size();
  d_VD_ro = d_VD_lame56 + VD_lame56[l].size();   //1
  d_epx = d_VD_ro + VD_ro[l].size();         //3
  d_epy = d_epx + epx.size();             //3
  d_epz = d_epy + epy.size();             //3
  d_gammaxy = d_epz + epz.size();         //3
  d_gammaxz = d_gammaxy + gammaxy.size(); //3
  d_gammayz = d_gammaxz + gammaxz.size(); //3
  d_temp_K = d_gammayz + gammayz.size();        //9
  d_K = d_temp_K + (nc * nc);             //9
  d_temp = d_K + (nc * nc);               //3

  cudaMemcpy(d_VD_lame11, VD_lame11[l].data(), VD_lame11[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame22, VD_lame22[l].data(), VD_lame22[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame33, VD_lame33[l].data(), VD_lame33[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame12, VD_lame12[l].data(), VD_lame12[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame13, VD_lame13[l].data(), VD_lame13[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame23, VD_lame23[l].data(), VD_lame23[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame44, VD_lame44[l].data(), VD_lame44[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame55, VD_lame55[l].data(), VD_lame55[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame66, VD_lame66[l].data(), VD_lame66[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame14, VD_lame14[l].data(), VD_lame14[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame24, VD_lame24[l].data(), VD_lame24[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame34, VD_lame34[l].data(), VD_lame34[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_lame56, VD_lame56[l].data(), VD_lame56[l].size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VD_ro, VD_ro[l].data(), VD_ro[l].size() * sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(d_epx, epx.data(), epx.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_epy, epy.data(), epy.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_epz, epz.data(), epz.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gammaxy, gammaxy.data(), gammaxy.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gammaxz, gammaxz.data(), gammaxz.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gammayz, gammayz.data(), gammayz.size() * sizeof(double), cudaMemcpyHostToDevice);
  //start of (epx.transpose() * sigx)

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame11[l].rows(), nc, VD_lame11[l].cols(), &van, d_VD_lame11, VD_lame11[l].rows(), d_epx, epx.rows(), &ziro, d_temp, VD_lame11[l].rows()); //VD_lame11 * epx
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epx.rows(), &van, d_epx, epx.rows(), d_temp, VD_lame11[l].rows(), &ziro, d_temp_K, nc);//epxT * VD_lame11 * epx
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame12[l].rows(), nc, VD_lame12[l].cols(), &van, d_VD_lame12, VD_lame12[l].rows(), d_epy, epy.rows(), &ziro, d_temp, VD_lame12[l].rows()); //VD_lame12 * epy
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epx.rows(), &van, d_epx, epx.rows(), d_temp, VD_lame12[l].rows(), &ziro, d_temp_K, nc);//epxT * VD_lame12 * epy
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame13[l].rows(), nc, VD_lame13[l].cols(), &van, d_VD_lame13, VD_lame13[l].rows(), d_epz, epz.rows(), &ziro, d_temp, VD_lame13[l].rows()); //VD_lame13 * epz
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epx.rows(), &van, d_epx, epx.rows(), d_temp, VD_lame13[l].rows(), &ziro, d_temp_K, nc);//epxT * VD_lame13 * epz
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame14[l].rows(), nc, VD_lame14[l].cols(), &van, d_VD_lame14, VD_lame14[l].rows(), d_gammaxy, gammaxy.rows(), &ziro, d_temp, VD_lame14[l].rows()); //VD_lame14 * gammaxy
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epx.rows(), &van, d_epx, epx.rows(), d_temp, VD_lame14[l].rows(), &ziro, d_temp_K, nc);//epxT * VD_lame14 * gammaxy
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  //END OF (epx.transpose() * sigx)

  //start of (epy.transpose() * sigy)

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame12[l].rows(), nc, VD_lame12[l].cols(), &van, d_VD_lame12, VD_lame12[l].rows(), d_epx, epx.rows(), &ziro, d_temp, VD_lame12[l].rows()); //VD_lame12 * epx
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epy.rows(), &van, d_epy, epy.rows(), d_temp, VD_lame12[l].rows(), &ziro, d_temp_K, nc);//epyT * VD_lame12 * epx
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame22[l].rows(), nc, VD_lame22[l].cols(), &van, d_VD_lame22, VD_lame22[l].rows(), d_epy, epy.rows(), &ziro, d_temp, VD_lame22[l].rows()); //VD_lame22 * epy
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epy.rows(), &van, d_epy, epy.rows(), d_temp, VD_lame22[l].rows(), &ziro, d_temp_K, nc);//epyT * VD_lame12 * epy
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame23[l].rows(), nc, VD_lame23[l].cols(), &van, d_VD_lame23, VD_lame23[l].rows(), d_epz, epz.rows(), &ziro, d_temp, VD_lame23[l].rows()); //VD_lame23 * epz
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epy.rows(), &van, d_epy, epy.rows(), d_temp, VD_lame23[l].rows(), &ziro, d_temp_K, nc);//epyT * VD_lame23 * epz
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame24[l].rows(), nc, VD_lame24[l].cols(), &van, d_VD_lame24, VD_lame24[l].rows(), d_gammaxy, gammaxy.rows(), &ziro, d_temp, VD_lame24[l].rows()); //VD_lame24 * gammaxy
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epy.rows(), &van, d_epy, epy.rows(), d_temp, VD_lame24[l].rows(), &ziro, d_temp_K, nc);//epyT * VD_lame24 * gammaxy
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  //END OF (epy.transpose() * sigy)

  //start of (epz.transpose() * sigz)

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame13[l].rows(), nc, VD_lame13[l].cols(), &van, d_VD_lame13, VD_lame13[l].rows(), d_epx, epx.rows(), &ziro, d_temp, VD_lame13[l].rows()); //VD_lame13 * epx
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epz.rows(), &van, d_epz, epz.rows(), d_temp, VD_lame13[l].rows(), &ziro, d_temp_K, nc);//epzT * VD_lame13 * epx
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame23[l].rows(), nc, VD_lame23[l].cols(), &van, d_VD_lame23, VD_lame23[l].rows(), d_epy, epy.rows(), &ziro, d_temp, VD_lame23[l].rows()); //VD_lame23 * epy
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epz.rows(), &van, d_epz, epz.rows(), d_temp, VD_lame23[l].rows(), &ziro, d_temp_K, nc);//epzT * VD_lame23 * epy
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame33[l].rows(), nc, VD_lame33[l].cols(), &van, d_VD_lame33, VD_lame33[l].rows(), d_epz, epz.rows(), &ziro, d_temp, VD_lame33[l].rows()); //VD_lame33 * epz
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epz.rows(), &van, d_epz, epz.rows(), d_temp, VD_lame33[l].rows(), &ziro, d_temp_K, nc);//epzT * VD_lame33 * epz
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame34[l].rows(), nc, VD_lame34[l].cols(), &van, d_VD_lame34, VD_lame34[l].rows(), d_gammaxy, gammaxy.rows(), &ziro, d_temp, VD_lame34[l].rows()); //VD_lame34 * gammaxy
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epz.rows(), &van, d_epz, epz.rows(), d_temp, VD_lame34[l].rows(), &ziro, d_temp_K, nc);//epzT * VD_lame34 * gammaxy
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  //END OF (epz.transpose() * sigz)
  
  //start of (gammaxy.transpose() * tauxy)

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame14[l].rows(), nc, VD_lame14[l].cols(), &van, d_VD_lame14, VD_lame14[l].rows(), d_epx, epx.rows(), &ziro, d_temp, VD_lame14[l].rows()); //VD_lame14 * epx
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammaxy.rows(), &van, d_gammaxy, gammaxy.rows(), d_temp, VD_lame14[l].rows(), &ziro, d_temp_K, nc);//gammaxyT * VD_lame14 * epx
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame24[l].rows(), nc, VD_lame24[l].cols(), &van, d_VD_lame24, VD_lame24[l].rows(), d_epy, epy.rows(), &ziro, d_temp, VD_lame24[l].rows()); //VD_lame24 * epy
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammaxy.rows(), &van, d_gammaxy, gammaxy.rows(), d_temp, VD_lame24[l].rows(), &ziro, d_temp_K, nc);//gammaxyT * VD_lame24 * epy
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame34[l].rows(), nc, VD_lame34[l].cols(), &van, d_VD_lame34, VD_lame34[l].rows(), d_epz, epz.rows(), &ziro, d_temp, VD_lame34[l].rows()); //VD_lame34 * epz
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammaxy.rows(), &van, d_gammaxy, gammaxy.rows(), d_temp, VD_lame34[l].rows(), &ziro, d_temp_K, nc);//gammxyT * VD_lame34 * epz
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame44[l].rows(), nc, VD_lame44[l].cols(), &van, d_VD_lame44, VD_lame44[l].rows(), d_gammaxy, gammaxy.rows(), &ziro, d_temp, VD_lame44[l].rows()); //VD_lame44 * gammaxy
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammaxy.rows(), &van, d_gammaxy, gammaxy.rows(), d_temp, VD_lame44[l].rows(), &ziro, d_temp_K, nc);//gammaxyT * VD_lame44 * gammaxy
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc); //Add to K

  //END OF (gammaxy.transpose() * tauxy)


  //start of (gammayz.transpose() * tauyz)
  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame55[l].rows(), nc, VD_lame55[l].cols(), &van, d_VD_lame55, VD_lame55[l].rows(), d_gammayz, gammayz.rows(), &ziro, d_temp, VD_lame55[l].rows()); //VD_lame55 * gammayz
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammayz.rows(), &van, d_gammayz, gammayz.rows(), d_temp, VD_lame55[l].rows(), &ziro, d_temp_K, nc);//gammayzT * VD_lame55 * gammayz
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc);//Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame56[l].rows(), nc, VD_lame56[l].cols(), &van, d_VD_lame56, VD_lame56[l].rows(), d_gammaxz, gammaxz.rows(), &ziro, d_temp, VD_lame56[l].rows()); //VD_lame56 * gammaxz
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammayz.rows(), &van, d_gammayz, gammayz.rows(), d_temp, VD_lame56[l].rows(), &ziro, d_temp_K, nc);//gammayzT * VD_lame56 * gammaxz
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc);//Add to K
  //END OF (gammayz.transpose() * tauyz)


  //start of (gammaxz.transpose() * tauxz)
  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame56[l].rows(), nc, VD_lame56[l].cols(), &van, d_VD_lame56, VD_lame56[l].rows(), d_gammayz, gammayz.rows(), &ziro, d_temp, VD_lame56[l].rows()); //VD_lame56 * gammayz
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammaxz.rows(), &van, d_gammaxz, gammaxz.rows(), d_temp, VD_lame56[l].rows(), &ziro, d_temp_K, nc);//gammaxzT * VD_lame56 * gammayz
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc);//Add to K

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, VD_lame66[l].rows(), nc, VD_lame66[l].cols(), &van, d_VD_lame66, VD_lame66[l].rows(), d_gammaxz, gammaxz.rows(), &ziro, d_temp, VD_lame66[l].rows()); //VD_lame66 * gammaxz
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammaxz.rows(), &van, d_gammaxz, gammaxz.rows(), d_temp, VD_lame66[l].rows(), &ziro, d_temp_K, nc);//gammaxzT * VD_lame66 * gammaxz
  cublasDgeam(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc);//Add to K
  //END OF (gammaxz.transpose() * tauxz)

  cudaStreamSynchronize(strm);
  K[l] = MatrixXd::Zero(nc, nc);
  cudaMemcpy(K[l].data(), d_K, nc * nc * sizeof(double), cudaMemcpyDeviceToHost);
  clear_mem(allocated_blocks,device_id);
  return true;
}

#endif



void FGM::T1_system_matrices_honeycomb_CPU(unsigned int l)
{
  int ub = 0;
  int ue = nxyz[l] - 1;
  int vb = nxyz[l];
  int ve = 2 * nxyz[l] - 1;
  int wb = 2 * nxyz[l];
  int we = 3 * nxyz[l] - 1;

  M[l](seq(ub, ue), seq(ub, ue)) = VD_ro[l];
  M[l](seq(vb, ve), seq(vb, ve)) = VD_ro[l];
  M[l](seq(wb, we), seq(wb, we)) = VD_ro[l];

  MatrixXd epx = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  epx(seq(0, nxyz[l] - 1), seq(ub, ue)) = shapes[l].QDx;
  MatrixXd epy = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  epy(seq(0, nxyz[l] - 1), seq(vb, ve)) = shapes[l].QDy;
  MatrixXd epz = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  epz(seq(0, nxyz[l] - 1), seq(wb, we)) = shapes[l].QDz;

  MatrixXd gammaxy = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  gammaxy(seq(0, nxyz[l] - 1), seq(ub, ue)) = shapes[l].QDy;
  gammaxy(seq(0, nxyz[l] - 1), seq(vb, ve)) = shapes[l].QDx;
  MatrixXd gammayz = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  gammayz(seq(0, nxyz[l] - 1), seq(vb, ve)) = shapes[l].QDz;
  gammayz(seq(0, nxyz[l] - 1), seq(wb, we)) = shapes[l].QDy;
  MatrixXd gammaxz = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  gammaxz(seq(0, nxyz[l] - 1), seq(ub, ue)) = shapes[l].QDz;
  gammaxz(seq(0, nxyz[l] - 1), seq(wb, we)) = shapes[l].QDx;


  MatrixXd sigx = (VD_lame11[l] * epx) + (VD_lame12[l] * epy) + (VD_lame13[l] * epz);
  MatrixXd sigy = (VD_lame12[l] * epx) + (VD_lame22[l] * epy) + (VD_lame23[l] * epz);
  MatrixXd sigz = (VD_lame13[l] * epx) + (VD_lame23[l] * epy) + (VD_lame33[l] * epz);
  MatrixXd tauxy = (VD_lame44[l] * gammaxy); 
  MatrixXd tauyz = (VD_lame55[l] * gammayz);
  MatrixXd tauxz = (VD_lame66[l] * gammaxz);
  
  K[l]= (epx.transpose() * sigx) + (epy.transpose() * sigy) + (epz.transpose() * sigz) + (gammaxy.transpose() * tauxy) + (gammaxz.transpose() * tauxz) + (gammayz.transpose() * tauyz);

}



void FGM::T1_system_matrices_CPU(unsigned int l)
{
  int ub = 0;
  int ue = nxyz[l] - 1;
  int vb = nxyz[l];
  int ve = 2 * nxyz[l] - 1;
  int wb = 2 * nxyz[l];
  int we = 3 * nxyz[l] - 1;

  M[l](seq(ub, ue), seq(ub, ue)) = VD_ro[l];
  M[l](seq(vb, ve), seq(vb, ve)) = VD_ro[l];
  M[l](seq(wb, we), seq(wb, we)) = VD_ro[l];

  MatrixXd epx = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  epx(seq(0, nxyz[l] - 1), seq(ub, ue)) = shapes[l].QDx;
  MatrixXd epy = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  epy(seq(0, nxyz[l] - 1), seq(vb, ve)) = shapes[l].QDy;
  MatrixXd epz = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  epz(seq(0, nxyz[l] - 1), seq(wb, we)) = shapes[l].QDz;

  MatrixXd gammaxy = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  gammaxy(seq(0, nxyz[l] - 1), seq(ub, ue)) = shapes[l].QDy;
  gammaxy(seq(0, nxyz[l] - 1), seq(vb, ve)) = shapes[l].QDx;
  MatrixXd gammayz = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  gammayz(seq(0, nxyz[l] - 1), seq(vb, ve)) = shapes[l].QDz;
  gammayz(seq(0, nxyz[l] - 1), seq(wb, we)) = shapes[l].QDy;
  MatrixXd gammaxz = MatrixXd::Zero(nxyz[l], 3 * nxyz[l]);
  gammaxz(seq(0, nxyz[l] - 1), seq(ub, ue)) = shapes[l].QDz;
  gammaxz(seq(0, nxyz[l] - 1), seq(wb, we)) = shapes[l].QDx;


  MatrixXd sigx = (VD_lame11[l] * epx) + (VD_lame12[l] * epy) + (VD_lame13[l] * epz) + (VD_lame14[l] * gammaxy);
  MatrixXd sigy = (VD_lame12[l] * epx) + (VD_lame22[l] * epy) + (VD_lame23[l] * epz) + (VD_lame24[l] * gammaxy);
  MatrixXd sigz = (VD_lame13[l] * epx) + (VD_lame23[l] * epy) + (VD_lame33[l] * epz) + (VD_lame34[l] * gammaxy);
  MatrixXd tauxy = (VD_lame44[l] * gammaxy) + (VD_lame14[l] * epx) + (VD_lame24[l] * epy) + (VD_lame34[l] * epz);
  MatrixXd tauyz = (VD_lame55[l] * gammayz) + (VD_lame56[l] * gammaxz);
  MatrixXd tauxz = (VD_lame66[l] * gammaxz) + (VD_lame56[l] * gammayz);
  K[l]= (epx.transpose() * sigx) + (epy.transpose() * sigy) + (epz.transpose() * sigz) + (gammaxy.transpose() * tauxy) + (gammaxz.transpose() * tauxz) + (gammayz.transpose() * tauyz);

}

#ifdef GPU

void FGM::clear_mem(std::vector<void*> &allocated_pool, int device_id){
	gpuErrchk(cudaSetDevice(device_id));
	int get_device_id = -1;
	gpuErrchk(cudaGetDevice(&get_device_id));
	if(get_device_id != device_id){
		exit(1);
	}
	for(int i = 0; i<allocated_pool.size(); i++){
		gpuErrchk(cudaFree(allocated_pool[i]));
	}
	allocated_pool.clear();
}

#endif
bool FGM::T1_system_matrices(unsigned int l)
{

#if defined GPU
  if(l !=1){
	std::vector<void*> allocated_address;
  	int device_id = 0;
	int index = 0;
	bool success = false;
	while(index < no_gpus){
		device_id = (ttt+index) % no_gpus;
		cout<<"T1 is taken by GPU "<< device_id <<  endl;
		success = T1_system_matrices_GPU(l,allocated_address,device_id);
		if(!success){
			std::cout<<"rolling back..."<<std::endl;
			clear_mem(allocated_address,device_id);
			device_id++;
		}else{
			std::cout<<"T1 - GPU done!"<<std::endl;
			break;
		}
		index++;
	}
	if(!success){
		cout<<"T1 is taken by CPU " << endl;
    		T1_system_matrices_CPU(l);
	}
  }
  else{
	bool success = false;
  	int device_id = 0;
	int index = 0;
 	std::vector<void*> allocated_address;
	while(index < no_gpus){
		device_id = (ttt+index) % no_gpus;
		cout<<"T1 is taken by GPU " << device_id << endl;
		success = T1_system_matrices_honeycomb_GPU(l,allocated_address,device_id);
		if(!success){
			std::cout<<"rolling back..."<<std::endl;
			clear_mem(allocated_address,device_id);
			device_id++;
		}else{
			std::cout<<"T1 - GPU done!"<<std::endl;
			break;
		}
		index++;
	}
	if(!success){
		cout<<"T1 is taken by CPU " << endl;
    		T1_system_matrices_honeycomb_CPU(l);
		cout<<"T1 is done by CPU"<<endl;
	}	
  }

  return true;

#else
  if(l != 1)
    T1_system_matrices_CPU(l);
  else
    T1_system_matrices_honeycomb_CPU(l);

  return false;
#endif
}




void FGM::T2_svd_CPU(MatrixXd &BC, MatrixXd &V)
{
  BDCSVD<MatrixXd> svd;
  svd.compute(BC, Eigen::ComputeFullV);
  V = svd.matrixV();
}




#ifdef GPU
bool FGM::T2_svd_GPU(std::vector<void*>& allocated_blocks, int device_id,MatrixXd &BC, MatrixXd &V)
{
  bool success = false;
  cudaStream_t strm = gpu_mans[device_id].get_cuda_stream(ttt);
  cublasHandle_t hndl = gpu_mans[device_id].get_cublas_handle(ttt);
  cusolverDnHandle_t cuslv = gpu_mans[device_id].get_cusolver_handle(ttt);
  gpuErrchkMem(cudaSetDevice(device_id), success);
  if(!success)
	  return false;
  gesvdjInfo_t gesvdj_params = NULL;

  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  cudaError_t cudaStat3 = cudaSuccess;
  cudaError_t cudaStat4 = cudaSuccess;
  cudaError_t cudaStat5 = cudaSuccess;
  const int m = BC.rows();
  const int n = BC.cols();

  double *d_A, *d_S, *d_U, *d_V, *d_work, *gpu_mem_beg;
  int *d_info;

  size_t total_bytes = ((m*n*sizeof(double)) + (n*sizeof(double)) + (m*n*sizeof(double)) + (n*n*sizeof(double)) + (n*sizeof(int)) + (n*sizeof(int)));

  gpuErrchkMem(cudaMalloc((void**)&gpu_mem_beg, total_bytes),success);                    //1
  if(!success)
	  return false;
  allocated_blocks.push_back(gpu_mem_beg);
  gpuErrchk(cudaMemset(gpu_mem_beg, 0, total_bytes));
  
  d_A = gpu_mem_beg;
  d_S = d_A + m * n;
  d_U = d_S + n;
  d_V = d_U + m * m;
  d_info = (int *)(d_V + n * n);
  int lwork = 0;
  d_work = (double *)(d_info + n);

  int info = 0;
  
  // gesvdj configuration
  const double tol = 1.e-7;
  const int max_sweeps = 15;
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  const int econ = 0;
  //numerical results of gesvdj
  double residual = 0;
  int executed_sweeps = 0;

  // Step 1: create cusolver handler and bind stream
  
  //status = cusolverDnCreate(&dnhandle[ttt]);
  //assert(CUSOLVER_STATUS_SUCCESS == status);

  //cudaStat1 = cudaStreamCreateWithFlags(&stream[ttt], cudaStreamNonBlocking);	
  //assert(cudaSuccess == status);
  
  //status = cusolverDnSetStream(dnhandle[ttt], stream);
  //assert(CUSOLVER_STATUS_SUCCESS == status);
  
  //Step 1: THIS STEP IS HANDLED WHEN THE THREADS ARE CREATED
  
  // Step 2: configuratşon of gesvdj
  status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  status = cusolverDnXgesvdjSetTolerance(gesvdj_params, tol); //defualt value of tolerance is machine zero
  assert(CUSOLVER_STATUS_SUCCESS == status);

  status = cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps); //defualt value of max sweeps is 100
  assert(CUSOLVER_STATUS_SUCCESS == status);

  //Step 3: copy the matrix to the device
  //cudaStat1 = cudaMalloc((void**)&d_A, sizeof(double)*m*n);
  //cudaStat2 = cudaMalloc((void**)&d_S, sizeof(double)*n);
  //cudaStat3 = cudaMalloc((void**)&d_U, sizeof(double)*m*m);
  //cudaStat4 = cudaMalloc((void**)&d_V, sizeof(double)*n*n);
  //cudaStat5 = cudaMalloc((void**)&d_info, sizeof(int));
  //assert(cudaSuccess == cudaStat1);
  //assert(cudaSuccess == cudaStat2);
  //assert(cudaSuccess == cudaStat3);
  //assert(cudaSuccess == cudaStat4);
  //assert(cudaSuccess == cudaStat5);

  cudaStat1 = cudaMemcpy(d_A, BC.data(), BC.size() * sizeof(double), cudaMemcpyHostToDevice);
      assert(cudaSuccess == cudaStat1); 	

  //Step 4: query workspace of SVD
  status = cusolverDnDgesvdj_bufferSize(cuslv, jobz, econ, m, n, d_A, m, d_S, d_U, m, d_V, n, &lwork, gesvdj_params);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  
  //Divide here deallocate all the memory if this gpu does not have enough memory
  gpuErrchkMem(cudaMalloc((void**)&d_work,(lwork)*sizeof(double)),success);                    //1
  if(!success)
	  return false;
  allocated_blocks.push_back((void*)d_work);
  gpuErrchk(cudaMemset(d_work, 0, (lwork) * sizeof(double)));
  
  //assert(cudaSuccess == cudaStat1);
  
  //Step 5: compute SVD
  status = cusolverDnDgesvdj(cuslv, jobz, econ, m, n, d_A, m, d_S, d_U, m, d_V, n, d_work, lwork, d_info, gesvdj_params);
 // cudaStat1 = cudaDeviceSynchronize(); // ???
  cudaStat1 = cudaStreamSynchronize(strm);
  
  assert(CUSOLVER_STATUS_SUCCESS == status);
  assert(cudaSuccess == cudaStat1);
  
  cudaStat2 = cudaMemcpy(V.data(), d_V, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat2);
  cudaStat4 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat4);
  if (0 == info){
    printf("gesvdj converges \n");
  }
  else if( 0 > info){
    printf("%d-th parameter is wrong \n", -info);
  }
  else{
    printf("WARNING: info = %d : gesvdj does not converge \n", info);
  }

  //cudaMemset(gpu_mem_beg, 0, (d_work+(lwork*sizeof(double)) - gpu_mem_beg));
  clear_mem(allocated_blocks,device_id);
  return true;
}

#endif



bool FGM::T2_svd(MatrixXd &BC, MatrixXd &V)
{
#if defined GPU
  bool success = false;
  int device_id = 0;
  int index = 0;
  std::vector<void*> allocated_address;
  while(index < no_gpus){
	device_id = (ttt+index) % no_gpus;
  	cout<<"T2 is taken by GPU "<<device_id <<  endl;
 	success = T2_svd_GPU(allocated_address,device_id,BC,V);
  	if(!success){
      		std::cout<<"rolling back..."<<std::endl;
      		clear_mem(allocated_address,device_id);
		device_id++;
	}else{
		std::cout<<"T2 on GPU done!"<<std::endl;
		break;
	}
	index++;
  }
  if(!success){
    	cout<<"T2 is taken by CPU " << endl;
     	T2_svd_CPU(BC,V);
	cout<<"T2 on GPU done!" << endl;
  }
  return success;
#else
  cout<<"T2 is taken by CPU " << endl;
  T2_svd_CPU(BC, V);
  return false;
#endif
}


#ifdef GPU
bool FGM::T3_mul_inv_GPU(std::vector<void*> &allocated_blocks, int device_id, MatrixXd &a0, MatrixXd &P)
{
  bool success = false;
  cudaStream_t strm = gpu_mans[device_id].get_cuda_stream(ttt);
  cublasHandle_t hndl = gpu_mans[device_id].get_cublas_handle(ttt);
  cusolverDnHandle_t cuslv = gpu_mans[device_id].get_cusolver_handle(ttt);
  gpuErrchkMem(cudaSetDevice(device_id), success);
  if(!success)
	  return false;
  const double van = 1.0;
  const double ziro = 0.0;
  double *d_K, *d_M, *d_P, *d_K_phy, *d_M_phy, *d_a0, *d_temp, *d_M_phy_i, *d_work, *gpu_mem_beg;
  int *d_pivot, *d_info, Lwork;
  int nc = P.cols();

  long total_bytes = (KK.rows() * KK.cols() * sizeof(double)) + (MM.rows() * MM.cols() * sizeof(double)) + (P.cols() * P.rows() * sizeof(double)) + (nc*nc*sizeof(double)) + (nc*nc*sizeof(double)) + (nc*nc*sizeof(double)) + (3*KK.rows()*nc*sizeof(double)) + (nc*nc*sizeof(double)) + (nc*sizeof(int)) + (nc*sizeof(int)) + (nc*sizeof(int));

  gpuErrchkMem(cudaMalloc((void**)&gpu_mem_beg, total_bytes),success);                    //1
  if(!success)
	  return false;
  allocated_blocks.push_back(gpu_mem_beg);
  gpuErrchk(cudaMemset(gpu_mem_beg,0,total_bytes));

  d_K = gpu_mem_beg;                            //9
  d_M = d_K + (KK.rows()*KK.cols());            //9
  d_P = d_M + (MM.rows()*MM.cols());            //max 9
  d_K_phy = d_P + (P.rows()*P.cols());          //max 9
  d_M_phy = d_K_phy + (nc * nc);            //max 9
  d_a0 = d_M_phy + (nc * nc);               //max 9
  d_temp = d_a0 + (nc * nc);                //max 9
  d_M_phy_i = d_temp + (3 * KK.rows() * nc);     //max 9
  d_pivot = (int *)(d_M_phy_i + (nc * nc)); //max 1
  d_info = d_pivot + nc;                    //max 1
  d_work = (double *)(d_info + nc);         //max 1.


  cudaMemcpy(d_K, KK.data(), KK.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_M, MM.data(), MM.size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_P, P.data(), P.size() * sizeof(double), cudaMemcpyHostToDevice);
  MatrixXd Id = MatrixXd::Identity(nc, nc);
  cudaMemcpy(d_M_phy_i, Id.data(), Id.size() * sizeof(double), cudaMemcpyHostToDevice);


  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, KK.rows(), nc, KK.cols(), &van, d_K, KK.rows(), d_P, P.rows(), &ziro, d_temp, KK.rows()); //alpha * K * P + beta * K
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, P.rows(), &van, d_P, P.rows(), d_temp, KK.rows(), &ziro, d_K_phy, P.cols());   //Pt * K * P
  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, MM.rows(), nc, MM.cols(), &van, d_M, MM.rows(), d_P, P.rows(), &ziro, d_temp, MM.rows()); //M * P
  cublasDgemm(hndl, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, P.rows(), &van, d_P, P.rows(), d_temp, MM.rows(), &ziro, d_M_phy, P.cols());   //Pt * M * P
  
  
  cusolveSafeCall(cusolverDnDgetrf_bufferSize(cuslv, nc, nc, d_M_phy, nc, &Lwork));
  gpuErrchkMem(cudaMalloc((void**)&d_work, (Lwork)*sizeof(double)),success);                    //1
  if(!success)
	  return false;
  allocated_blocks.push_back(d_work);
  gpuErrchk(cudaMemset(d_work,0,(Lwork*sizeof(double))));

  cusolveSafeCall(cusolverDnDgetrf(cuslv, nc, nc, d_M_phy, nc, d_work, d_pivot, d_info));
  cusolveSafeCall(cusolverDnDgetrs(cuslv, CUBLAS_OP_N, nc, nc, d_M_phy, nc, d_pivot, d_M_phy_i, nc, d_info));

  cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, nc, &van, d_M_phy_i, nc, d_K_phy, nc, &ziro, d_a0, nc); //M_phy_i * K_phy
  cudaStreamSynchronize(strm);

  cudaMemcpy(a0.data(), d_a0, nc * nc * sizeof(double), cudaMemcpyDeviceToHost);
  clear_mem(allocated_blocks,device_id);
  return true;
}
#endif



void FGM::T3_mul_inv_CPU(MatrixXd &a0, MatrixXd &P)
{
  //apply sparse multiplication and then ldt
  SparseMatrix<double> KK_sparse = KK.sparseView();
  SparseMatrix<double> MM_sparse = MM.sparseView();
  SparseMatrix<double> P_sparse = P.sparseView();
  KK_sparse.makeCompressed();
  MM_sparse.makeCompressed();
  P_sparse.makeCompressed();
  MatrixXd K_phy = P_sparse.transpose() * (KK_sparse * P_sparse);
  MatrixXd M_phy = P_sparse.transpose() * (MM_sparse * P_sparse);
  
  a0 = M_phy.llt().solve(K_phy);  
}




bool FGM::T3_mul_inv(MatrixXd &a0, MatrixXd &P)
{
#if defined GPU
	bool success = false;
	int device_id = 0;
	int index = 0;
	while(index < no_gpus){
		std::vector<void*> allocated_address;
		device_id = (ttt+index) % no_gpus;
		cout<<"T3 is taken by GPU " << device_id << endl;
		success = T3_mul_inv_GPU(allocated_address,device_id,a0,P);
		if(!success){
			std::cout<<"rolling back..."<<std::endl;
			clear_mem(allocated_address,device_id);
			device_id++;
		}else{
			std::cout<<"T3 on GPU done!"<<std::endl;
			break;
		}
		index++;
	}
	if(!success){
		std::cout<<"T3 is taken by CPU"<<std::endl;	
		T3_mul_inv_CPU(a0,P);
		std::cout<<"T3 on CPU done!"<<std::endl;
	}
	return success;
#else
	T3_mul_inv_CPU(a0, P);
	return false;
#endif
}




void FGM::T4_eigen(MatrixXd &a0, int &nconv, double &small_eig)
{
  MatrixXd MMM = a0;
  DenseGenRealShiftSolve<double> op(MMM);
  GenEigsRealShiftSolver<DenseGenRealShiftSolve<double>> eigs(op, 10, 50, 0.001);

  eigs.init();
  {
  nconv = eigs.compute();
  }

  Eigen::VectorXcd evalues;
  if (eigs.info() == CompInfo::Successful)
  {
    evalues = eigs.eigenvalues();
    small_eig = evalues(nconv - 1).real();
  }
  if(small_eig < 0 ){
	cout<<"NEG EIG!!"<<endl;
  	exit(1);
  }
}




void FGM::compute_gpu_costs(const int noeigs, const int ncv, int &nconv, double &small_eig, const double shift, const int max_iter, const double tol, int g_id, int sample_size)
{
  /*
      double cost;
      gcosts[PADDING * g_id][0] = 0;
      for (int i = 0; i < sample_size; i++)
      {
      double t1t = omp_get_wtime();
      T1_system_matrices_GPU();
      cost = omp_get_wtime() - t1t;
      gcosts[PADDING * g_id][0] += cost;
      }
      gcosts[PADDING * g_id][0] /= sample_size;

      MatrixXd BC_3D_I = boundary_condition_3d(0, 0);
      MatrixXd BC_3D_II = boundary_condition_3d(0, 1);

      MatrixXd BC_1 = beta_matrix_3d(BC_3D_I, 0);
      MatrixXd BC_2 = beta_matrix_3d(BC_3D_II, 0);
      MatrixXd BC(BC_1.rows() + BC_2.rows(), BC_1.cols());
      BC << BC_1, BC_2;

      MatrixXd V;
      T2_svd(BC, V);

      MatrixXd P = V(seq(0, V.rows() - 1), seq(BC.rows(), BC.cols() - 1));
      MatrixXd a0;
      gcosts[PADDING * g_id][2] = 0;
      for (int i = 0; i < sample_size; i++)
      {
      double t3t = omp_get_wtime();
      T3_mul_inv_GPU(a0, P);
      cost = omp_get_wtime() - t3t;
      gcosts[PADDING * g_id][2] += cost;
      }
      gcosts[PADDING * g_id][2] /= sample_size;
      */
}




void FGM::compute_cpu_costs(const int noeigs, const int ncv, int &nconv, double &small_eig,
				const double shift, const int max_iter, const double tol, const int sample_size)
{
  /*
      double cost;
      ccosts[0] = 0;
      for (int i = 0; i < sample_size; i++)
      {
      double t1t = omp_get_wtime();
      T1_system_matrices_CPU();
      cost = omp_get_wtime() - t1t;
      ccosts[0] += cost;
      }
      ccosts[0] /= sample_size;

      MatrixXd BC_3D_I = boundary_condition_3d(0, 0);
      MatrixXd BC_3D_II = boundary_condition_3d(0, 1);

      MatrixXd BC_1 = beta_matrix_3d(BC_3D_I, 0);
      MatrixXd BC_2 = beta_matrix_3d(BC_3D_II, 0);
      MatrixXd BC(BC_1.rows() + BC_2.rows(), BC_1.cols());
      BC << BC_1, BC_2;

      MatrixXd V;
      ccosts[1] = 0;
      for (int i = 0; i < sample_size; i++)
      {
      double t2t = omp_get_wtime();
      T2_svd(BC, V);
      cost = omp_get_wtime() - t2t;
      ccosts[1] += cost;
      }
      ccosts[1] /= sample_size;

      MatrixXd P = V(seq(0, V.rows() - 1), seq(BC.rows(), BC.cols() - 1));
      MatrixXd a0;
      ccosts[2] = 0;
      for (int i = 0; i < sample_size; i++)
      {
      double t3t = omp_get_wtime();
      T3_mul_inv_CPU(a0, P);
      cost = omp_get_wtime() - t3t;
      ccosts[2] += cost;
      }
      ccosts[2] /= sample_size;

      ccosts[3] = 0;
      for (int i = 0; i < sample_size; i++)
      {
      double t4t = omp_get_wtime();
      T4_eigen(a0, nconv, small_eig);
      cost = omp_get_wtime() - t4t;
      ccosts[3] += cost;
      }
      ccosts[3] /= sample_size;
      */
}





void FGM::removeDuplicateRows(MatrixXd &mat)
{
  vector<Eigen::VectorXd> vec;
  for(int i = 0; i < mat.rows(); i++){
    vec.push_back(mat.row(i));
  }
  sort(vec.begin(), vec.end(), [](Eigen::VectorXd const& t1, Eigen::VectorXd const& t2){ for(int i = 0; i < t1.size(); i++){ if(t1(i) != t2(i)){ return t1(i) < t2(i);}}	return false;});
  vec.erase(unique(vec.begin(), vec.end()), vec.end());
  mat.resize(vec.size(), mat.cols());
  for(int i = 0; i < vec.size(); i++){
    mat.row(i) = vec[i];
  }
}


MatrixXd FGM::removeZeroRows(MatrixXd &mat)
{
  MatrixXd temp = mat.rowwise().any();
  vector<int> vec(temp.data(), temp.data() + temp.rows()*temp.cols());
  vector<int> res;
  for(int i = 0; i < vec.size(); i++){
    if(vec[i] == 1){
      res.push_back(i);
    }
  }
  return mat(res, Eigen::all);
}



void FGM::prepareBoundaryCondition(MatrixXd &mat, unsigned int l)
{
  for(int i = 0; i < 3*np[l][0]; i++){
    if(i == 0){
      mat(seq(0, 0), seq(0, mat.cols()-1)).setZero(); 
    }
    if(i < 3*np[l][0]-1){
      mat(seq((i+1)*np[l][1], (i+1)*np[l][1]), seq(0, mat.cols()-1)).setZero(); 
    }
    //mat(seq((i+1)*np[l][1]-1, (i+1)*np[l][1]-1), seq(0, mat.cols()-1)).setZero(); 
  }
  /*
  mat(seq(0, np[l][1]-1), seq(0, mat.cols()-1)).setZero();
  mat(seq(np[l][1]*(np[l][0]-1), np[l][1]*np[l][0]-1), seq(0, mat.cols()-1)).setZero();
  mat(seq(np[l][1]*np[l][0], np[l][1]*(np[l][0]+1)-1), seq(0, mat.cols()-1)).setZero();
  mat(seq(np[l][1]*(np[l][0]+np[l][1]-1),np[l][1]*(np[l][0]+np[l][1])-1), seq(0, mat.cols()-1)).setZero();	
  mat(seq(np[l][1]*(np[l][0]+np[l][1]-1), np[l][1]*(np[l][0]+np[l][1]+1)-1), seq(0, mat.cols()-1)).setZero();
  mat(seq(np[l][1]*(np[l][0]+np[l][1]+np[l][1]-1), np[l][1]*(np[l][0]+np[l][1]+np[l][1])-1), seq(0, mat.cols()-1)).setZero();
  */
}




void FGM::compute(const int noeigs, const int ncv, int &nconv, double &small_eig,
				const double shift, const int max_iter, const double tol)
{
  int tid = omp_get_thread_num();
  int copyStartX = 0;
  int copyStartY = 0;

  for(int i = 0; i < num_shapes; i++){
    double t1t = omp_get_wtime();
    bool gpu_load = T1_system_matrices(i);
    double cost = omp_get_wtime() - t1t;
#ifdef SMART
    if (gpu_load)
    {
      gcosts[PADDING * rinfos[tid]->gpu_id][0] = cost;
    }
    else
    {
      ccosts[0] = cost;
    }
#endif
    MM(seq(copyStartX, copyStartX + M[i].rows() - 1), seq(copyStartY, copyStartY+M[i].rows() - 1)) = M[i];
    KK(seq(copyStartX, copyStartX + K[i].rows() - 1), seq(copyStartY, copyStartY+K[i].rows() - 1)) = K[i];
    copyStartX += M[i].rows();
    copyStartY += M[i].cols(); 
  }

  MatrixXd BC_3D_I = boundary_condition_3d(2, 1, 0);
  MatrixXd BC_I = beta_matrix_3d(BC_3D_I, 2, 0);
  prepareBoundaryCondition(BC_I, 0);	
  MatrixXd BC_I_U = removeZeroRows(BC_I);

  MatrixXd BC_3D_II = boundary_condition_3d(2, 0, 1);
  MatrixXd BC_II = beta_matrix_3d(BC_3D_II, 2, 1);
  prepareBoundaryCondition(BC_II, 1);	
  MatrixXd BC_II_B = removeZeroRows(BC_II);
  
  MatrixXd BC_3D_III = boundary_condition_3d(2, 1, 1);
  MatrixXd BC_III = beta_matrix_3d(BC_3D_III, 2, 1);
  prepareBoundaryCondition(BC_III, 1);	
  MatrixXd BC_III_U = removeZeroRows(BC_III);
  
  MatrixXd BC_3D_IV = boundary_condition_3d(2, 0, 2);
  MatrixXd BC_IV = beta_matrix_3d(BC_3D_IV, 2, 2);
  prepareBoundaryCondition(BC_IV, 2);	
  MatrixXd BC_IV_B = removeZeroRows(BC_IV);

  MatrixXd BC_3D_V13 = boundary_condition_3d(0, 0, 0);
  MatrixXd BC_V13 = beta_matrix_3d(BC_3D_V13, 0, 0);
  
  MatrixXd BC_3D_V2 = boundary_condition_3d(0, 0, 1);
  MatrixXd BC_V2 = beta_matrix_3d(BC_3D_V2, 0, 1);
  
  MatrixXd BC_3D_VI13 = boundary_condition_3d(0, 1, 0);
  MatrixXd BC_VI13 = beta_matrix_3d(BC_3D_VI13, 0, 0);
  
  MatrixXd BC_3D_VI2 = boundary_condition_3d(0, 1, 1);
  MatrixXd BC_VI2 = beta_matrix_3d(BC_3D_VI2, 0, 1);
  
  MatrixXd BC_3D_VII13 = boundary_condition_3d(1, 0, 0);
  MatrixXd BC_VII13 = beta_matrix_3d(BC_3D_VII13, 1, 0);
  
  MatrixXd BC_3D_VII2 = boundary_condition_3d(1, 0, 1);
  MatrixXd BC_VII2 = beta_matrix_3d(BC_3D_VII2, 1, 1);
  
  MatrixXd BC_3D_VIII13 = boundary_condition_3d(1, 1, 0);
  MatrixXd BC_VIII13 = beta_matrix_3d(BC_3D_VIII13, 1, 0);
  
  MatrixXd BC_3D_VIII2 = boundary_condition_3d(1, 1, 1);
  MatrixXd BC_VIII2 = beta_matrix_3d(BC_3D_VIII2, 1, 1);
/*
  MatrixXd BC(BC_I_U.rows() + BC_III_U.rows() + BC_VII13.rows() + BC_VII2.rows() + BC_VII13.rows() + BC_VIII13.rows() + BC_VIII2.rows() + BC_VIII13.rows() + BC_V13.rows() + BC_V2.rows() + BC_V13.rows() + BC_VI13.rows() + BC_VI2.rows() + BC_VI13.rows(), MM.cols());
  BC.setZero();
  int rowStart = 0;
  BC(seq(0, BC_I_U.rows()-1), seq(0, BC_I_U.cols()-1)) = BC_I_U;
  BC(seq(0, BC_II_B.rows()-1), seq(BC_I_U.cols(), BC_I_U.cols() + BC_II_B.cols()-1)) = -1*BC_II_B;
  
  rowStart = BC_I_U.rows();
  BC(seq(rowStart, rowStart+BC_III_U.rows()-1), seq(BC.cols()-BC_IV_B.cols()-BC_III_U.cols(), BC.cols()-BC_IV_B.cols()-1)) = BC_III_U;
  BC(seq(rowStart, rowStart + BC_IV_B.rows()-1), seq(BC.cols()-BC_IV_B.cols(), BC.cols()-1)) = -1*BC_IV_B;
  
  rowStart += BC_III_U.rows();
  BC(seq(rowStart, rowStart + BC_VII13.rows()-1), seq(0, BC_VII13.cols()-1)) = BC_VII13;
    
  rowStart += BC_VII13.rows();
  BC(seq(rowStart, rowStart + BC_VII2.rows()-1), seq(np[0][0] * np[0][1] * np[0][2] * 3, np[0][0] * np[0][1] * np[0][2] * 3 - 1 + BC_VII2.cols())) = BC_VII2;

  rowStart += BC_VII2.rows();
  BC(seq(rowStart, rowStart + BC_VII13.rows()-1), seq(BC.cols() - BC_VII13.cols(), BC.cols()-1)) = BC_VII13;

  rowStart += BC_VII13.rows();
  BC(seq(rowStart, rowStart + BC_VIII13.rows()-1), seq(0, BC_VIII13.cols()-1)) = BC_VIII13;

  rowStart += BC_VIII13.rows();
  BC(seq(rowStart, rowStart + BC_VIII2.rows()-1), seq(np[0][0]*np[0][1]*np[0][2]*3, np[0][0]*np[0][1]*np[0][2]*3 - 1 + BC_VIII2.cols())) = BC_VIII2; 

  rowStart += BC_VIII2.rows();
  BC(seq(rowStart, rowStart + BC_VIII13.rows()-1), seq(BC.cols()-BC_VIII13.cols(), BC.cols()-1)) = BC_VIII13;

  rowStart += BC_VIII13.rows();
  BC(seq(rowStart, rowStart + BC_V13.rows()-1), seq(0, BC_V13.cols()-1)) = BC_V13;
  rowStart += BC_V13.rows();
  BC(seq(rowStart, rowStart + BC_V2.rows()-1), seq(np[2][0]*np[2][1]*np[2][2]*3, np[2][0]*np[2][1]*np[2][2]*3 - 1 + BC_V2.cols())) = BC_V2;

  rowStart += BC_V2.rows();
  BC(seq(rowStart, rowStart + BC_V13.rows()-1), seq(BC.cols() - BC_V13.cols(), BC.cols()-1)) = BC_V13;	

  rowStart += BC_V13.rows();
  BC(seq(rowStart, rowStart + BC_VI13.rows()-1), seq(0, BC_VI13.cols()-1)) = BC_VI13;

  rowStart += BC_VI13.rows();
  BC(seq(rowStart, rowStart + BC_VI2.rows()-1), seq(np[2][0]*np[2][1]*np[2][2]*3, np[2][0]*np[2][1]*np[2][2]*3-1+BC_VI2.cols())) = BC_VI2;

  rowStart += BC_VI2.rows();
  BC(seq(rowStart, rowStart + BC_VI13.rows()-1), seq(BC.cols()-BC_VI13.cols(), BC.cols()-1)) = BC_VI13;
  removeDuplicateRows(BC);
*/

  MatrixXd BC(BC_I_U.rows() + BC_III_U.rows() + BC_VII13.rows() + BC_VII2.rows() + BC_VII13.rows(), MM.cols());
  BC.setZero();
  int rowStart = 0;
  BC(seq(0, BC_I_U.rows()-1), seq(0, BC_I_U.cols()-1)) = BC_I_U;
  BC(seq(0, BC_II_B.rows()-1), seq(BC_I_U.cols(), BC_I_U.cols() + BC_II_B.cols()-1)) = -1*BC_II_B;
  
  rowStart = BC_I_U.rows();
  BC(seq(rowStart, rowStart+BC_III_U.rows()-1), seq(BC.cols()-BC_IV_B.cols()-BC_III_U.cols(), BC.cols()-BC_IV_B.cols()-1)) = BC_III_U;
  BC(seq(rowStart, rowStart + BC_IV_B.rows()-1), seq(BC.cols()-BC_IV_B.cols(), BC.cols()-1)) = -1*BC_IV_B;
  
  rowStart += BC_III_U.rows();
  BC(seq(rowStart, rowStart + BC_VII13.rows()-1), seq(0, BC_VII13.cols()-1)) = BC_VII13;
    
  rowStart += BC_VII13.rows();
  BC(seq(rowStart, rowStart + BC_VII2.rows()-1), seq(np[0][0] * np[0][1] * np[0][2] * 3, np[0][0] * np[0][1] * np[0][2] * 3 - 1 + BC_VII2.cols())) = BC_VII2;

  rowStart += BC_VII2.rows();
  BC(seq(rowStart, rowStart + BC_VII13.rows()-1), seq(BC.cols() - BC_VII13.cols(), BC.cols()-1)) = BC_VII13;

  MatrixXd V(BC.cols(), BC.cols());
  double t2t = omp_get_wtime();
  bool ranOnGPU = T2_svd(BC, V);
  double cost = omp_get_wtime() - t2t;
#ifdef SMART
  if (ranOnGPU)
  {
    gcosts[PADDING * rinfos[tid]->gpu_id][1] = cost;
  }
  else
  {
    ccosts[1] = cost;
  }
#endif

  MatrixXd P = V(seq(0, V.rows() - 1), seq(BC.rows(), BC.cols() - 1));
  MatrixXd a0(P.cols(), P.cols());
  double t3t = omp_get_wtime();
  bool gpu_load = T3_mul_inv(a0, P);
  cost = omp_get_wtime() - t3t;
#ifdef SMART
  if (gpu_load)
  {
    gcosts[PADDING * rinfos[tid]->gpu_id][2] = cost;
  }
  else
  {
    ccosts[2] = cost;
  }
#endif
  double t4t = omp_get_wtime();
  T4_eigen(a0, nconv, small_eig);
  cost = omp_get_wtime() - t4t;
#ifdef SMART
  ccosts[3] = cost;
#endif
  cout << "T4 (Eigen) => Cost: " << cost << " secs - nconv = " << nconv << endl;
}




MatrixXd FGM::beta_matrix_3d(MatrixXd &BC_3D, int xyz, unsigned int l)
{
  MatrixXd BC = MatrixXd::Zero(3 * nxyz[l] / np[l][xyz], 3 * nxyz[l]);
  int ids[3];
  for (int dim = 0; dim < 3; dim++)
  {
    for (int i = 0; i < np[l][0]; i++)
    {
      ids[0] = i;
      for (int j = 0; j < np[l][1]; j++)
      {
        ids[1] = j;
        for (int k = 0; k < np[l][2]; k++)
        {
          ids[2] = k;

          int idx = dim * (nxyz[l] / np[l][xyz]);
          if (xyz == 0)
            idx += j * np[l][2] + k;
          else if (xyz == 1)
            idx += i * np[l][2] + k;
          else if (xyz == 2)
            idx += i * np[l][1] + j;
          int idy = dim * nxyz[l] +
            i * np[l][1] * np[l][2] +
            j * np[l][2] +
            k;

          BC(idx, idy) = BC_3D(dim, ids[xyz]);
        }
      }
    }
  }
  return BC;
}




void FGM::FG_var_MT_CNT(unsigned int l)
{
  VectorXd &x = shapes[l].spaces[0].s;
  VectorXd &y = shapes[l].spaces[1].s;
  VectorXd &z = shapes[l].spaces[2].s;

  Tensor<double,3> V_cnt = Tensor<double,3>(x.size(), y.size(), z.size());
  V_cnt.setZero();
  Tensor<double,3> ro_star = Tensor<double,3>(x.size(), y.size(), z.size());
  ro_star.setZero();

  Tensor<double,3> nu12 = Tensor<double,3>(x.size(), y.size(), z.size());
  nu12.setZero();

  Tensor<double,3> nu13 = Tensor<double,3>(x.size(), y.size(), z.size());
  nu13.setZero();

  Tensor<double,3> nu23 = Tensor<double,3>(x.size(), y.size(), z.size());
  nu23.setZero();

  Tensor<double,3> E_star_11 = Tensor<double,3>(x.size(), y.size(), z.size());
  E_star_11.setZero();
  
  Tensor<double,3> E_star_22 = Tensor<double,3>(x.size(), y.size(), z.size());
  E_star_22.setZero();

  Tensor<double,3> E_star_33 = Tensor<double,3>(x.size(), y.size(), z.size());
  E_star_33.setZero();

  Tensor<double,3> G12_star = Tensor<double,3>(x.size(), y.size(), z.size());
  G12_star.setZero();

  Tensor<double,3> G13_star = Tensor<double,3>(x.size(), y.size(), z.size());
  G13_star.setZero();

  Tensor<double,3> G23_star = Tensor<double,3>(x.size(), y.size(), z.size());
  G23_star.setZero();

  Tensor<double,3> nu21 = Tensor<double,3>(x.size(), y.size(), z.size());
  nu21.setZero();

  Tensor<double,3> nu31 = Tensor<double,3>(x.size(), y.size(), z.size());
  nu31.setZero();

  Tensor<double,3> nu32 = Tensor<double,3>(x.size(), y.size(), z.size());
  nu32.setZero();

  Tensor<double,3> E11 = Tensor<double,3>(x.size(), y.size(), z.size());
  E11.setZero();

  Tensor<double,3> E22 = Tensor<double,3>(x.size(), y.size(), z.size());
  E22.setZero();

  Tensor<double,3> E33 = Tensor<double,3>(x.size(), y.size(), z.size());
  E33.setZero();

  Tensor<double,3> G12 = Tensor<double,3>(x.size(), y.size(), z.size());
  G12.setZero();
  
  Tensor<double,3> G23 = Tensor<double,3>(x.size(), y.size(), z.size());
  G23.setZero();

  Tensor<double,3> G13 = Tensor<double,3>(x.size(), y.size(), z.size());
  G13.setZero();

  Tensor<double,3> Q11 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q11.setZero();
  Tensor<double,3> Q12 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q12.setZero();
  Tensor<double,3> Q13 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q13.setZero();
  Tensor<double,3> Q22 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q22.setZero();
  Tensor<double,3> Q23 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q23.setZero();
  Tensor<double,3> Q33 = Tensor<double,3>(x.size(), y.size(), z.size());
        Q33.setZero();
  Tensor<double,3> Q44 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q44.setZero();
  Tensor<double,3> Q55 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q55.setZero();
  Tensor<double,3> Q66 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q66.setZero();

  Tensor<double,3> delta = Tensor<double,3>(x.size(), y.size(), z.size());
  delta.setZero();



  double V_star_cnt = shapes[l].material.v_str_cnt; 
  double ro_cnt = shapes[l].material.ro_cnt; 
  double ro_m = shapes[l].material.mod_ro; 

  double h = shapes[l].dim[2]; 

  double nu12_cnt = shapes[l].material.poissons[0]; 
  double nu13_cnt = shapes[l].material.poissons[1]; 
  double nu23_cnt = shapes[l].material.poissons[2]; 
  double nu_m = shapes[l].material.poisson_ratio; 

  double eta_star_1 = shapes[l].material.eta_star_1; 
  double eta_star_2 = shapes[l].material.eta_star_2; 
  double eta_star_3 = shapes[l].material.eta_star_3;
  double E11_cnt = shapes[l].material.elasticity[0]; 
  double E22_cnt = shapes[l].material.elasticity[1];
  double E33_cnt = shapes[l].material.elasticity[2]; 
  double E_m = shapes[l].material.mod_elasticity;


  double G12_cnt = shapes[l].material.sheer_elasticity[0]; 
  double G13_cnt = shapes[l].material.sheer_elasticity[1]; 
  double G23_cnt = shapes[l].material.sheer_elasticity[2]; 
  double G_m = shapes[l].material.mod_sheer_elasticity; 


  double e0 = shapes[l].material.e; 
  double em = 1-sqrt(1-e0); 	

  double alpha = shapes[l].material.alpha; 
  double alpha_prime = sqrt(alpha); 
  double theta = shapes[l].theta;  


  for(int i = 0; i < x.size(); i++){
    for(int j = 0; j < y.size(); j++){
      for(int k = 0; k < z.size(); k++){
        switch (shapes[l].material.CNT_type)
        {
          case Material::CNT_TYPE::UD:
            {
              V_cnt(i,j,k) =V_star_cnt; 		
            }break;

          case Material::CNT_TYPE::FGV:
            {
              V_cnt(i,j,k) = (1 + 2 * (z(k) - h/2)/h)*V_star_cnt;
            }break;
          case Material::CNT_TYPE::FGO:
            {
              V_cnt(i,j,k) = 2 * (1 - 2 * abs(z(k) - h/2)/h)*V_star_cnt;
            }break;
          case Material::CNT_TYPE::FGX:
            {
              V_cnt(i,j,k) = (4 * abs(z(k) - h/2)/h) * V_star_cnt;
            }break;
        }
        
        ro_star(i,j,k) = ro_cnt * V_cnt(i,j,k) + ro_m * (1 - V_cnt(i,j,k));
        nu12(i,j,k) = nu12_cnt * V_star_cnt + nu_m * (1 - V_star_cnt);
        nu13(i,j,k) = nu13_cnt * V_star_cnt + nu_m * (1 - V_star_cnt);
        nu23(i,j,k) = nu23_cnt * V_star_cnt + nu_m * (1 - V_star_cnt);	
        E_star_11(i,j,k) = eta_star_1 * V_cnt(i,j,k) * E11_cnt + (1 - V_cnt(i,j,k)) * E_m;
        E_star_22(i,j,k) = eta_star_2 / (V_cnt(i,j,k) / E22_cnt + (1 - V_cnt(i,j,k)) / E_m);
        E_star_33(i,j,k) = eta_star_3 / (V_cnt(i,j,k) / E33_cnt + (1 - V_cnt(i,j,k)) / E_m);
        G12_star(i,j,k)  = eta_star_3 / (V_cnt(i,j,k) / G12_cnt + (1 - V_cnt(i,j,k))/G_m); 	
        G13_star(i,j,k)  = eta_star_3 / (V_cnt(i,j,k) / G13_cnt + (1 - V_cnt(i,j,k))/G_m); 	
        G23_star(i,j,k)  = eta_star_3 / (V_cnt(i,j,k) / G23_cnt + (1 - V_cnt(i,j,k))/G_m); 
        nu21(i,j,k) = nu12(i,j,k) * E_star_22(i,j,k) / E_star_11(i,j,k);	
        nu31(i,j,k) = nu13(i,j,k) * E_star_33(i,j,k) / E_star_11(i,j,k);	
        nu32(i,j,k) = nu23(i,j,k) * E_star_33(i,j,k) / E_star_22(i,j,k);	

        switch(shapes[l].material.POROUS_type)
        {
          case Material::POROUS_TYPE::AA: 
            {
              G12(i,j,k)= G12_star(i,j,k) *(1-e0*cos(pi*(z(k)-h/2)/h));
              G23(i,j,k)= G23_star(i,j,k) *(1-e0*cos(pi*(z(k)-h/2)/h));
              G13(i,j,k)= G13_star(i,j,k) *(1-e0*cos(pi*(z(k)-h/2)/h));
              rho[l](i,j,k)= ro_star(i,j,k) *(1-em*cos(pi*(z(k)-h/2)/h));
              E11(i,j,k)= E_star_11(i,j,k)*(1-e0*cos(pi*(z(k)-h/2)/h));
              E22(i,j,k)= E_star_22(i,j,k)*(1-e0*cos(pi*(z(k)-h/2)/h));
              E33(i,j,k)= E_star_33(i,j,k)*(1-e0*cos(pi*(z(k)-h/2)/h));

            }break;
          case Material::POROUS_TYPE::BB:
            {
              G12(i,j,k)= G12_star(i,j,k) *(1-e0*cos(pi*z(k)/(2*h)+pi/4));
                              G23(i,j,k)= G23_star(i,j,k) *(1-e0*cos(pi*z(k)/(2*h)+pi/4));
                            G13(i,j,k)= G13_star(i,j,k) *(1-e0*cos(pi*z(k)/(2*h)+pi/4));
                              rho[l](i,j,k)=  ro_star(i,j,k) *(1-em*cos(pi*z(k)/(2*h)+pi/4));
                              E11(i,j,k)= E_star_11(i,j,k)*(1-e0*cos(pi*z(k)/(2*h)+pi/4));
                            E22(i,j,k)= E_star_22(i,j,k)*(1-e0*cos(pi*z(k)/(2*h)+pi/4));
                              E33(i,j,k)= E_star_33(i,j,k)*(1-e0*cos(pi*z(k)/(2*h)+pi/4));
            }break;
          case Material::POROUS_TYPE::CC:
            {
                              G12(i,j,k)= G12_star(i,j,k) *alpha;
                                G23(i,j,k)= G13_star(i,j,k) *alpha;
                              G13(i,j,k)= G23_star(i,j,k) *alpha;                    
                    rho[l](i,j,k)= ro_star(i,j,k) *alpha_prime;
              E11(i,j,k)= E_star_11(i,j,k)*alpha;
              E22(i,j,k)= E_star_22(i,j,k)*alpha;
              E33(i,j,k)= E_star_33(i,j,k)*alpha;
            }break;
        }

        delta(i,j,k)=(1-nu12(i,j,k)*nu21(i,j,k)-nu23(i,j,k)*nu32(i,j,k)-nu31(i,j,k)*nu13(i,j,k)-2*nu21(i,j,k)*nu32(i,j,k)*nu13(i,j,k))/E11(i,j,k)/E22(i,j,k)/E33(i,j,k);
        Q11(i,j,k)=(1-nu23(i,j,k)*nu32(i,j,k))/E22(i,j,k)/E33(i,j,k)/delta(i,j,k);
        Q22(i,j,k)=(1-nu13(i,j,k)*nu31(i,j,k))/E11(i,j,k)/E33(i,j,k)/delta(i,j,k);
        Q33(i,j,k)=(1-nu12(i,j,k)*nu21(i,j,k))/E22(i,j,k)/E11(i,j,k)/delta(i,j,k);
        Q12(i,j,k)=(nu21(i,j,k)+nu31(i,j,k)*nu23(i,j,k))/E22(i,j,k)/E33(i,j,k)/delta(i,j,k);
        Q13(i,j,k)=(nu31(i,j,k)+nu21(i,j,k)*nu32(i,j,k))/E22(i,j,k)/E33(i,j,k)/delta(i,j,k);
        Q23(i,j,k)=(nu32(i,j,k)+nu12(i,j,k)*nu31(i,j,k))/E11(i,j,k)/E33(i,j,k)/delta(i,j,k);
        Q44(i,j,k)=G12(i,j,k);
        Q55(i,j,k)=G23(i,j,k);
        Q66(i,j,k)=G13(i,j,k);

      }
    }
  }

    
  Q11T[l]=Q11*pow(cos(theta),4)+2*(Q12+2*Q44)*(pow(cos(theta),2)*pow(sin(theta),2))+Q22*pow(sin(theta),4);
  
  Q12T[l]=Q12*pow(cos(theta),4)+(Q11+Q22-4*Q44)*(pow(cos(theta),2)*pow(sin(theta),2))+Q12*pow(sin(theta),4);
  
  Q13T[l]=Q13*pow(cos(theta),2)+Q23*pow(sin(theta),2);

  Q14T[l]=(Q11-Q12-2*Q44)*(pow(cos(theta),3)*sin(theta))+(2*Q44+Q12-Q22)*(cos(theta)*pow(sin(theta),3));

  Q22T[l]=Q22*pow(cos(theta),4)+2*(Q12+2*Q44)*(pow(cos(theta),2)*pow(sin(theta),2))+Q11*pow(sin(theta),4);

  Q23T[l]=Q23*pow(cos(theta),2)+Q13*pow(sin(theta),2);

  Q24T[l]=(Q12-Q22+2*Q44)*(pow(cos(theta),3)*sin(theta))+(Q11-Q12-2*Q44)*(cos(theta)*pow(sin(theta),3));

  Q33T[l]=Q33;

  Q34T[l]=(Q13-Q23)*(cos(theta)*sin(theta));

  Q44T[l]=(Q11+Q22-2*Q12-2*Q44)*(pow(cos(theta),2)*pow(sin(theta),2))+Q44*(pow(cos(theta),4)+pow(sin(theta),4));

  Q55T[l]=Q55*pow(cos(theta),2)+Q66*pow(sin(theta),2);

  Q56T[l]=(Q66-Q55)*cos(theta)*sin(theta);

  Q66T[l]=Q66*pow(cos(theta),2)+Q55*pow(sin(theta),2);

}






void FGM::FG_var_MT_honeycomb(unsigned int l)
{
  VectorXd &x = shapes[l].spaces[0].s;
  VectorXd &y = shapes[l].spaces[1].s;
  VectorXd &z = shapes[l].spaces[2].s;
  
  double nu12 = shapes[l].material.poissons[0]; 
  double nu23 = shapes[l].material.poissons[1]; 
  double nu13 = shapes[l].material.poissons[2];

  double e11 = shapes[l].material.elasticity[0];	
  double e22 = shapes[l].material.elasticity[1];	
  double e33 = shapes[l].material.elasticity[2];	
  
  double g12 = shapes[l].material.sheer_elasticity[0];	
  double g13 = shapes[l].material.sheer_elasticity[1];	
  double g23 = shapes[l].material.sheer_elasticity[2];	


  double nu21 = nu12 * e22 / e11;
  double nu32 = nu23 * e33 / e22;
  double nu31 = nu13 * e33 / e11;

  double delta = (1-nu12*nu21-nu23*nu32-nu31*nu13-2*nu21*nu32*nu13)/e11/e22/e33;
  
  double ro_h = 24.94;

  Tensor<double,3> Q11 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q11.setZero();
  Tensor<double,3> Q12 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q12.setZero();
  Tensor<double,3> Q13 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q13.setZero();
  Tensor<double,3> Q22 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q22.setZero();
  Tensor<double,3> Q23 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q23.setZero();
  Tensor<double,3> Q33 = Tensor<double,3>(x.size(), y.size(), z.size());
        Q33.setZero();
  Tensor<double,3> Q44 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q44.setZero();
  Tensor<double,3> Q55 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q55.setZero();
  Tensor<double,3> Q66 = Tensor<double,3>(x.size(), y.size(), z.size());
  Q66.setZero();

  for(int i = 0; i<x.size(); i++){
    for(int j = 0; j<y.size(); j++){
      for(int k=0; k<z.size(); k++){
        Q11T[l](i,j,k) = (1-nu23*nu32)/e22/e33/delta;
        Q22T[l](i,j,k) = (1-nu13*nu31)/e11/e33/delta;
        Q33T[l](i,j,k) = (1-nu12*nu21)/e22/e11/delta;
        Q12T[l](i,j,k) = (nu21+nu31*nu23)/e22/e33/delta;
        Q13T[l](i,j,k) = (nu31+nu21*nu32)/e22/e33/delta;
        Q23T[l](i,j,k) = (nu32+nu12*nu31)/e11/e33/delta;
        Q44T[l](i,j,k) = g12;
        Q55T[l](i,j,k) = g23;
        Q66T[l](i,j,k) = g13;
        rho[l](i,j,k) = ro_h;
      }
    }
  }
}




void FGM::tensor3(VectorXd &v_d3Nt, MatrixXd &Sst, int n,
				Tensor<double, 3> &X)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      for (int k = 0; k < n; k++)
      {
        double xijk = 0;
        for (int l = 0; l < 3 * n; l++)
        {
          xijk += v_d3Nt(l) * Sst(l, i) * Sst(l, j) * Sst(l, k);
        }
        X(i, j, k) = xijk;
      }
    }
  }
}



void FGM::inner_helper(Tensor<double, 3> &Axyz, Tensor<double, 3> &Xadl, Tensor<double, 3> &Ybem, Tensor<double, 3> &Zcfn,
				MatrixXd &VD, unsigned int l)
{
  double ****Xadmn;
  alloc4D(Xadmn, np[l][0], np[l][0], np[l][1], np[l][2]);
  for (int i = 0; i < np[l][0]; i++)
  {
    for (int j = 0; j < np[l][0]; j++)
    {
      for (int k = 0; k < np[l][1]; k++)
      {
        for (int ll = 0; ll < np[l][2]; ll++)
        {
          double sum = 0;
          for (int m = 0; m < np[l][0]; m++)
          {
            sum += Xadl(i, j, m) * Axyz(m, k, ll);
          }
          Xadmn[i][j][k][ll] = sum;
        }
      }
    }
  }

  double *****Gamma_adnbe;
  alloc5D(Gamma_adnbe, np[l][0], np[l][0], np[l][2], np[l][1], np[l][1]);
  for (int i = 0; i < np[l][0]; i++)
  {
    for (int j = 0; j < np[l][0]; j++)
    {
      for (int k = 0; k < np[l][2]; k++)
      {
        for (int ll = 0; ll < np[l][1]; ll++)
        {
          for (int m = 0; m < np[l][1]; m++)
          {
            double sum = 0;
            for (int o = 0; o < np[l][1]; o++)
            {
              sum += Xadmn[i][j][o][k] * Ybem(o, ll, m);
            }
            Gamma_adnbe[i][j][k][ll][m] = sum;
          }
        }
      }
    }
  }

  for (int i = 0; i < np[l][0]; i++)
  {
    for (int j = 0; j < np[l][0]; j++)
    {
      for (int k = 0; k < np[l][1]; k++)
      {
        for (int ll = 0; ll < np[l][1]; ll++)
        {
          for (int m = 0; m < np[l][2]; m++)
          {
            for (int o = 0; o < np[l][2]; o++)
            {
              double sum = 0;
              for (int v = 0; v < np[l][2]; v++)
              {
                sum += Gamma_adnbe[i][j][v][k][ll] * Zcfn(v, m, o);
              }
              int row = (i)*np[l][1] * np[l][2] + (k)*np[l][2] + m;
              int col = (j)*np[l][1] * np[l][2] + (ll)*np[l][2] + o;

              VD(row, col) += sum;
            }
          }
        }
      }
    }
  }

  free4D(Xadmn, np[l][0], np[l][0], np[l][1], np[l][2]);
  free5D(Gamma_adnbe, np[l][0], np[l][0], np[l][2], np[l][1], np[l][1]);
}





void FGM::inner_product_honeycomb(unsigned int l)
{
  MatrixXd IFT[3][3][2];
  for (int i = 0; i < 3; i++)
  { //xyz loop
    for (int j = 0; j < 3; j++)
    { //123 loop
      int sz = (j + 1) * np[l][i];
      IFT[i][j][0] = MatrixXd::Zero(sz, sz);
      IFT[i][j][1] = MatrixXd::Zero(sz, sz);
      cheb(sz, IFT[i][j][0], IFT[i][j][1]);
    }
  }

  VectorXd v_d3N[3];
  for (int i = 0; i < 3; i++)
  {
    VectorXd temp = cheb_int(shapes[l].spaces[i].start, shapes[l].spaces[i].end, 3 * np[l][i]);
    v_d3N[i] = (temp.transpose() * IFT[i][2][1]).transpose();
  }

  MatrixXd Ss[3];
  for (int i = 0; i < 3; i++)
  {
    MatrixXd I = MatrixXd::Identity(np[l][i], np[l][i]);
    MatrixXd Z = MatrixXd::Zero(np[l][i], np[l][i]);
    MatrixXd C(3 * np[l][i], np[l][i]);
    C << I, Z, Z;
    Ss[i] = IFT[i][2][0] * C * IFT[i][0][1];
  }

  Tensor<double, 3> Xadl(np[l][0], np[l][0], np[l][0]);
  Xadl.setZero();
  tensor3(v_d3N[0], Ss[0], np[l][0], Xadl);
  Tensor<double, 3> Ybem(np[l][1], np[l][1], np[l][1]);
  Ybem.setZero();
  tensor3(v_d3N[1], Ss[1], np[l][1], Ybem);
  Tensor<double, 3> Zcfn(np[l][2], np[l][2], np[l][2]);
  Zcfn.setZero();
  tensor3(v_d3N[2], Ss[2], np[l][2], Zcfn);

  Tensor<double,3> p11 = Q11T[l] * JAC[l];
  inner_helper(p11, Xadl, Ybem, Zcfn, VD_lame11[l], l);
  
  Tensor<double,3> p12 = Q12T[l] * JAC[l];
  inner_helper(p12, Xadl, Ybem, Zcfn, VD_lame12[l], l);

  Tensor<double,3> p13 = Q13T[l] * JAC[l];
  inner_helper(p13, Xadl, Ybem, Zcfn, VD_lame13[l], l);

  Tensor<double,3> p22 = Q22T[l] * JAC[l];
  inner_helper(p22, Xadl, Ybem, Zcfn, VD_lame22[l], l);

  Tensor<double,3> p33 = Q33T[l] * JAC[l];
  inner_helper(p33, Xadl, Ybem, Zcfn, VD_lame33[l], l);
  
  Tensor<double,3> p23 = Q23T[l] * JAC[l];
  inner_helper(p23, Xadl, Ybem, Zcfn, VD_lame23[l], l);

  Tensor<double,3> p44 = Q44T[l] * JAC[l];
  inner_helper(p44, Xadl, Ybem, Zcfn, VD_lame44[l], l);

  Tensor<double,3> p55 = Q55T[l] * JAC[l];
  inner_helper(p55, Xadl, Ybem, Zcfn, VD_lame55[l], l);

  Tensor<double,3> p66 = Q66T[l] * JAC[l];
  inner_helper(p66, Xadl, Ybem, Zcfn, VD_lame66[l], l);

  Tensor<double,3> prho = rho[l] * JAC[l];
  inner_helper(prho, Xadl, Ybem, Zcfn, VD_ro[l], l);	
}




void FGM::inner_product(unsigned int l)
{
  MatrixXd IFT[3][3][2];
  for (int i = 0; i < 3; i++)
  { //xyz loop
    for (int j = 0; j < 3; j++)
    { //123 loop
      int sz = (j + 1) * np[l][i];
      IFT[i][j][0] = MatrixXd::Zero(sz, sz);
      IFT[i][j][1] = MatrixXd::Zero(sz, sz);
      cheb(sz, IFT[i][j][0], IFT[i][j][1]);
    }
  }

  VectorXd v_d3N[3];
  for (int i = 0; i < 3; i++)
  {
    VectorXd temp = cheb_int(shapes[l].spaces[i].start, shapes[l].spaces[i].end, 3 * np[l][i]);
    v_d3N[i] = (temp.transpose() * IFT[i][2][1]).transpose();
  }

  MatrixXd Ss[3];
  for (int i = 0; i < 3; i++)
  {
    MatrixXd I = MatrixXd::Identity(np[l][i], np[l][i]);
    MatrixXd Z = MatrixXd::Zero(np[l][i], np[l][i]);
    MatrixXd C(3 * np[l][i], np[l][i]);
    C << I, Z, Z;
    Ss[i] = IFT[i][2][0] * C * IFT[i][0][1];
  }

  Tensor<double, 3> Xadl(np[l][0], np[l][0], np[l][0]);
  Xadl.setZero();
  tensor3(v_d3N[0], Ss[0], np[l][0], Xadl);
  Tensor<double, 3> Ybem(np[l][1], np[l][1], np[l][1]);
  Ybem.setZero();
  tensor3(v_d3N[1], Ss[1], np[l][1], Ybem);
  Tensor<double, 3> Zcfn(np[l][2], np[l][2], np[l][2]);
  Zcfn.setZero();
  tensor3(v_d3N[2], Ss[2], np[l][2], Zcfn);
  Tensor<double,3> p11 = Q11T[l] * JAC[l];
  inner_helper(p11, Xadl, Ybem, Zcfn, VD_lame11[l], l);
  
  Tensor<double,3> p12 = Q12T[l] * JAC[l];
  inner_helper(p12, Xadl, Ybem, Zcfn, VD_lame12[l], l);

  Tensor<double,3> p13 = Q13T[l] * JAC[l];
  inner_helper(p13, Xadl, Ybem, Zcfn, VD_lame13[l], l);

  Tensor<double,3> p22 = Q22T[l] * JAC[l];
  inner_helper(p22, Xadl, Ybem, Zcfn, VD_lame22[l], l);

  Tensor<double,3> p33 = Q33T[l] * JAC[l];
  inner_helper(p33, Xadl, Ybem, Zcfn, VD_lame33[l], l);
  
  Tensor<double,3> p23 = Q23T[l] * JAC[l];
  inner_helper(p23, Xadl, Ybem, Zcfn, VD_lame23[l], l);

  Tensor<double,3> p44 = Q44T[l] * JAC[l];
  inner_helper(p44, Xadl, Ybem, Zcfn, VD_lame44[l], l);

  Tensor<double,3> p55 = Q55T[l] * JAC[l];
  inner_helper(p55, Xadl, Ybem, Zcfn, VD_lame55[l], l);

  Tensor<double,3> p66 = Q66T[l] * JAC[l];
  inner_helper(p66, Xadl, Ybem, Zcfn, VD_lame66[l], l);

  
  Tensor<double,3> p14 = Q14T[l] * JAC[l];
  inner_helper(p14, Xadl, Ybem, Zcfn, VD_lame14[l], l);
  Tensor<double,3> p24 = Q24T[l] * JAC[l];
  inner_helper(p24, Xadl, Ybem, Zcfn, VD_lame24[l], l);
  Tensor<double,3> p34 = Q34T[l] * JAC[l];
  inner_helper(p34, Xadl, Ybem, Zcfn, VD_lame34[l], l);
  Tensor<double,3> p56 = Q56T[l] * JAC[l];
  inner_helper(p56, Xadl, Ybem, Zcfn, VD_lame56[l], l);
  

  Tensor<double,3> prho = rho[l] * JAC[l];
  inner_helper(prho, Xadl, Ybem, Zcfn, VD_ro[l], l);	

}





MatrixXd FGM::boundary_condition_3d(int xyz, int ol, unsigned int l)
{
  double bc[3] = {1, 1, 1};
  RowVectorXd e(np[l][xyz]);
  for (int i = 0; i < np[l][xyz]; i++)
    e(i) = 0;
  if (ol == 0)
  {
    e(0) = 1.0;
  }
  else
  {
    e(np[l][xyz] - 1) = 1.0;
  }
  MatrixXd BC(3, np[l][xyz]);
  BC << (bc[0] * e), (bc[1] * e), (bc[2] * e);
  return BC;
}
