#pragma once
#include <iostream>
#define EIGEN_USE_MKL_ALL
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Spectra/GenEigsRealShiftSolver.h>
#include <Spectra/MatOp/DenseGenRealShiftSolve.h>
#include <Eigen/SVD>
#include <math.h>
#include "consts.h"

using namespace Eigen;
using namespace Spectra;
using namespace std;

void alloc4D(dtype**** &ptr, int n1, int n2, int n3, int n4) {
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

void free4D(dtype**** &ptr, int n1, int n2, int n3, int n4) {
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

void alloc5D(dtype***** &ptr, int n1, int n2, int n3, int n4, int n5) {
  ptr = new dtype****[n1];
  for(int i = 0; i < n1; i++) {
    alloc4D(ptr[i], n2, n3, n4, n5);
  }
}

void free5D(dtype***** &ptr, int n1, int n2, int n3, int n4, int n5) {
  for(int i = 0; i < n1; i++) {
    free4D(ptr[i], n2, n3, n4, n5);
  }
  delete [] ptr;
}

void alloc6D(dtype****** &ptr, int n1, int n2, int n3, int n4, int n5, int n6) {
  ptr = new dtype*****[n1];
  for(int i = 0; i < n1; i++) {
    alloc5D(ptr[i], n2, n3, n4, n5, n6);
  }
}

void free6D(dtype****** &ptr, int n1, int n2, int n3, int n4, int n5, int n6) {
  for(int i = 0; i < n1; i++) {
    free5D(ptr[i], n2, n3, n4, n5, n6);
  }
  delete [] ptr;
}


VectorXd cheb_di(dtype start, dtype end, int N) {
  VectorXd v_di(N);
  v_di.setZero();
  for(int i = 1; i <= N; i += 2) {
    v_di(i-1) = (end - start) / (1.0 - ((i-1) * (i-1)));
  }
  return v_di;
}

VectorXd cheb_int(dtype start, dtype end, int N) {
  dtype scale = 2 / (end - start);  
  VectorXd vecint(N);
  vecint.setZero();
  for(int k = 0; k < N; k += 2) {
    vecint(k) = 2 / ((1 - (k * k)) * scale);
  }
  return vecint;
}

void cheb(int N, 
	  MatrixXd &IT, MatrixXd &FT) {
  for(int i = 1; i <= N; i++) {
    for(int j = 1; j <= N; j++) {
      IT(i-1,j-1) = cos((j-1) * pi * (N - i)/(N - 1));
    } 
  }  
  FT = IT.inverse();
}

void derivative(dtype start, dtype end, int N,
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
      for(int j = 1; j < DN + odd; j++) { //SOR burada odd gerekli mi, bu caseler birbirinin ayni mi yoksa
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

  //  D = D.triangularView<Eigen::Upper>() / scale;
}

VectorXd lobat(int N) {
  VectorXd x(N);
  int nm1 = N - 1;
  for(int i = 0; i < N; i++) {
    x(i) = sin((pi * ((-1 * nm1) + (2.0 * i))) / (2.0 * nm1));
  }
  return x;
}

void slobat(dtype start, dtype end, int N, 
	    VectorXd& s) {
  VectorXd lbt = lobat(N);
  dtype alpha = (end - start) / 2;
  dtype beta = (end + start) / 2;
  VectorXd z = VectorXd::Constant(N, 1);; 
  s  = alpha * lbt + beta * z;
}

void inner_product_helper(dtype start, dtype end, int N,
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
  
