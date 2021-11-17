#include "Shape.h"


Shape::Shape(dtype x_dim, dtype y_dim, dtype z_dim,
      int x_sample, int y_sample, int z_sample, 
      Material mat, double ctrl_y, double ctrl_z,double _theta, 
      dtype xcurve, dtype ycurve,double zeta_add) : dim{x_dim, y_dim, z_dim}, curve{xcurve, ycurve},
    is_curved(~(xcurve == 0 && ycurve == 0)),
    spaces{Space(-x_dim / 2, x_dim / 2, x_sample), Space(-y_dim / 2, y_dim / 2, y_sample), Space(0, z_dim, z_sample)},
    xyz(x_sample * y_sample * z_sample),
    VD(xyz, xyz),
    QDx(xyz, xyz), QDy(xyz, xyz), QDz(xyz, xyz), material(mat), 
    ctrl_y(ctrl_y), ctrl_z(ctrl_z),zeta_add(zeta_add), theta(_theta)
    {
      QDx.setZero();
      QDy.setZero();
      QDz.setZero();
      shapeX = Tensor<double,3>(x_sample, y_sample, z_sample);
      shapeY = Tensor<double,3>(x_sample, y_sample, z_sample);
      shapeZ = Tensor<double,3>(x_sample, y_sample, z_sample);
      shapeX.setZero();
      shapeY.setZero();
      shapeZ.setZero();
      
      jac = Tensor<double,3>(x_sample,y_sample,z_sample);
      fillShapeTensors();
      double alpha = curve[0]*2*pi/dim[0];
      double beta = curve[1]*2*pi/dim[1];
      vector_map_jac_curvature(alpha,beta);				
    }



void Shape::fillShapeTensors(){	
  for(int i = 0; i < spaces[0].no_points; i++){
    for(int j = 0; j < spaces[1].no_points; j++){
      for(int k = 0; k < spaces[2].no_points; k++){
        shapeX(i,j,k) = spaces[0].s(i);
        shapeY(i,j,k) = spaces[1].s(j);
        shapeZ(i,j,k) = spaces[2].s(k) + zeta_add;
      }
    }
  }
}



void Shape::vector_map_jac_curvature(double alpha, double beta){

  int x_sample = spaces[0].no_points;
  int y_sample = spaces[1].no_points;
  int z_sample = spaces[2].no_points;


        
  Tensor<double,3> tempx = Tensor<double,3>(x_sample, y_sample, z_sample);
  Tensor<double,3> tempy = Tensor<double,3>(x_sample, y_sample, z_sample);
  Tensor<double,3> tempz = Tensor<double,3>(x_sample, y_sample, z_sample);
  
  tempx.setZero();
  tempy.setZero();
  tempz.setZero();
  
  Tensor<double,3> temp2x = Tensor<double,3>(x_sample, y_sample, z_sample);
  Tensor<double,3> temp2y = Tensor<double,3>(x_sample, y_sample, z_sample);
  Tensor<double,3> temp2z = Tensor<double,3>(x_sample, y_sample, z_sample);

  temp2x.setZero();
  temp2y.setZero();
  temp2z.setZero();
  if(alpha != 0){
    //loop over temp vars
    for(int i = 0; i<x_sample; i++){
      for(int j = 0; j<y_sample; j++){
        for(int k = 0; k<z_sample; k++){
          tempx(i,j,k) = (cos(alpha*shapeX(i,j,k)) * ( (1/alpha) * sin(alpha*shapeX(i,j,k))))
            - (sin(alpha*shapeX(i,j,k)) * (shapeZ(i,j,k) - ((1/alpha) * (1-cos(alpha*shapeX(i,j,k))))));
          tempy(i,j,k) = shapeY(i,j,k);
          tempz(i,j,k) = (sin(alpha * shapeX(i,j,k)) * ( (1/alpha) * sin(alpha * shapeX(i,j,k))))
                  + (cos(alpha * shapeX(i,j,k)) * (shapeZ(i,j,k) - ((1/alpha) * (1-cos(alpha*shapeX(i,j,k))))));	
        }
      }
    }
  }else{
    tempx = shapeX;
    tempy = shapeY;
    tempz = shapeZ;
  }

  if(beta != 0){
    //loop over temp2 vars
    for(int i = 0; i<x_sample; i++){
      for(int j = 0; j<y_sample; j++){
        for(int k = 0; k<z_sample; k++){
          temp2x(i,j,k) = tempx(i,j,k);	

          temp2y(i,j,k) = (cos(beta*tempy(i,j,k)) * ( (1/beta) * sin(beta*tempx(i,j,k))))
            - (sin(beta*tempx(i,j,k)) * (tempz(i,j,k) - ((1/beta) * (1-cos(beta*tempx(i,j,k))))));
          
          temp2z(i,j,k) = (sin(beta*tempy(i,j,k)) * ( (1/beta) * sin(beta*tempy(i,j,k))))
            + (cos(beta*tempy(i,j,k)) * (tempz(i,j,k) - ((1/beta) * (1-cos(beta*tempy(i,j,k))))));
        }
      }
    }

  }else{
    temp2x = tempx;
    temp2y = tempy;
    temp2z = tempz;
  }


  MatrixXd Dijk = MatrixXd(3,3);
  MatrixXd Bijk = MatrixXd(3,3);
  MatrixXd Eijk = MatrixXd(3,3);

  double dxdxb2 = 0;
  double dxdyb2 = 0;
  double dxdzb2 = 0;

  double dydxb2 = 0;
  double dydyb2 = 0;
  double dydzb2 = 0;

  double dzdxb2 = 0;
  double dzdyb2 = 0;
  double dzdzb2 = 0;

  double dxdxb = 0;
  double dxdyb = 0;
  double dxdzb = 0;

  double dydxb = 0;
  double dydyb = 0;
  double dydzb = 0;

  double dzdxb = 0;
  double dzdyb = 0;
  double dzdzb = 0;



  Tensor<double,3> dxidx = Tensor<double,3>(x_sample,y_sample,z_sample);
  dxidx.setZero();
  Tensor<double,3> detadx = Tensor<double,3>(x_sample,y_sample,z_sample);
  detadx.setZero();
  Tensor<double,3> dzetadx = Tensor<double,3>(x_sample,y_sample,z_sample);
  dzetadx.setZero();


  Tensor<double,3> dxidy = Tensor<double,3>(x_sample,y_sample,z_sample);
  dxidy.setZero();
  Tensor<double,3> detady = Tensor<double,3>(x_sample,y_sample,z_sample);
  detady.setZero();
  Tensor<double,3> dzetady = Tensor<double,3>(x_sample,y_sample,z_sample);
  dzetady.setZero();


  Tensor<double,3> dxidz = Tensor<double,3>(x_sample,y_sample,z_sample);
  dxidz.setZero();
  Tensor<double,3> detadz = Tensor<double,3>(x_sample,y_sample,z_sample);
  detadz.setZero();
  Tensor<double,3> dzetadz = Tensor<double,3>(x_sample,y_sample,z_sample);
  dzetadz.setZero();


  for(int i = 0; i < x_sample; i++){
    for(int j = 0; j < y_sample; j++){
      for(int k = 0; k < z_sample; k++){
        dxdxb2 = cos(alpha * shapeX(i,j,k)) - (alpha * cos(alpha * shapeX(i,j,k)) * shapeZ(i,j,k));
        dxdyb2 = 0;
        dxdzb2 = -1 * sin(alpha * shapeX(i,j,k));
        dydxb2 = 0;
        dydyb2 = 1;
        dydzb2 = 0;

        dzdxb2 = sin(alpha * shapeX(i,j,k)) - (shapeZ(i,j,k) * alpha * sin(alpha * shapeX(i,j,k)));
        dzdyb2 = 0;
        dzdzb2 = cos(alpha * shapeX(i,j,k));

        dxdxb = 1;
        dxdyb = 0;
        dxdzb = 0;

        dydxb = 0;
        dydyb = cos((beta * tempy(i,j,k)) - (beta * cos((beta * tempy(i,j,k)) * tempz(i,j,k))));
        dydzb = -1 * sin(beta * tempy(i,j,k));

        dzdxb = 0;
        dzdyb = sin(beta * tempy(i,j,k)) - (tempz(i,j,k) * beta * sin(beta * tempy(i,j,k)));
        dzdzb = cos(beta * tempy(i,j,k));

        Dijk<<dxdxb2, dydxb2, dzdxb2,
              dxdyb2, dydyb2, dzdyb2,
              dxdzb2, dydzb2, dzdzb2;

        Bijk<<dxdxb, dydxb, dzdxb,
              dxdyb, dydyb, dzdyb,
              dxdzb, dydzb, dzdzb;
        
        jac(i,j,k) = (Dijk * Bijk).determinant();				
        Eijk = (Dijk * Bijk).inverse();

        dxidx(i,j,k) = Eijk(0,0);
        detadx(i,j,k) = Eijk(0,1);
        dzetadx(i,j,k) = Eijk(0,2);

        dxidy(i,j,k) = Eijk(1,0);
        detady(i,j,k) = Eijk(1,1);
        dzetady(i,j,k) = Eijk(1,2);

        dxidz(i,j,k) = Eijk(2,0);
        detadz(i,j,k) = Eijk(2,1);
        dzetadz(i,j,k) = Eijk(2,2);
        
      }
    }
  }
  //Vector mapping
  int xyz = x_sample * y_sample * z_sample;
  MatrixXd QDxi_dxidx(xyz,xyz);
  MatrixXd QDxi_dxidy(xyz,xyz);
  MatrixXd QDxi_dxidz(xyz,xyz);

  MatrixXd QDeta_detadx(xyz,xyz);
  MatrixXd QDeta_detady(xyz,xyz);
  MatrixXd QDeta_detadz(xyz,xyz);

  
  MatrixXd QDzeta_dzetadx(xyz,xyz);
  MatrixXd QDzeta_dzetady(xyz,xyz);
  MatrixXd QDzeta_dzetadz(xyz,xyz);

  QDxi_dxidx.setZero();
  QDxi_dxidy.setZero();
  QDxi_dxidz.setZero();

  QDeta_detadx.setZero();
  QDeta_detady.setZero();
  QDeta_detadz.setZero();

  QDzeta_dzetadx.setZero();
  QDzeta_dzetady.setZero();
  QDzeta_dzetadz.setZero();
  
  for(int i = 1; i <= x_sample; i++){
    for(int j = 1; j <= y_sample; j++){
      for(int k = 1; k <= z_sample; k++){

        int I = ((i - 1) * y_sample * z_sample) + ((j - 1) * z_sample) + k;
        for(int l = 1; l<= x_sample; l++){
          
          int J = ((l - 1) * y_sample * z_sample) + ((j - 1) * z_sample) + k;
          
          QDxi_dxidx(J - 1, I - 1) += dxidx(l-1,j-1,k-1)*spaces[0].Q1(l - 1, i - 1);
          QDxi_dxidy(J - 1, I - 1) += dxidy(l-1,j-1,k-1)*spaces[0].Q1(l - 1, i - 1);
          QDxi_dxidz(J - 1, I - 1) += dxidz(l-1,j-1,k-1)*spaces[0].Q1(l - 1, i - 1);
        }

        for(int l = 1; l<=y_sample; l++){
          int J = ((i - 1) * y_sample * z_sample) + ((l - 1) * z_sample) + k;
          QDeta_detadx(J-1, I-1) += detadx(l-1,j-1,k-1)*spaces[1].Q1(l-1,j-1);
          QDeta_detady(J-1, I-1) += detady(l-1,j-1,k-1)*spaces[1].Q1(l-1,j-1);
          QDeta_detadz(J-1, I-1) += detadz(l-1,j-1,k-1)*spaces[1].Q1(l-1,j-1);
        }

        for(int l = 1; l<=z_sample; l++){
          int J = ((i - 1) * y_sample * z_sample) + ((j - 1) * z_sample) + l;
          QDzeta_dzetadx(J-1, I-1) += dzetadx(i-1,j-1,l-1)*spaces[2].Q1(l-1,k-1);
          QDzeta_dzetady(J-1, I-1) += dzetady(i-1,j-1,l-1)*spaces[2].Q1(l-1,k-1);
          QDzeta_dzetadz(J-1, I-1) += dzetadz(i-1,j-1,l-1)*spaces[2].Q1(l-1,k-1);
        }
      }
    }
  }

  QDx = (QDxi_dxidx+QDeta_detadx+QDzeta_dzetadx);
  QDy = (QDxi_dxidy+QDeta_detady+QDzeta_dzetady);
  QDz = (QDxi_dxidz+QDeta_detadz+QDzeta_dzetadz);

}


void Shape::operator=(const Shape& s){
  for(unsigned int i = 0; i < 3; i++){
    dim[i] = s.dim[i];
    spaces[i] = s.spaces[i];
    if(i < 2){
      curve[i] = s.curve[i];
    }
  }
  is_curved = s.is_curved;
  xyz = s.xyz;
  VD = s.VD;
  VD = s.VD;
  QDx = s.QDx;
  QDy = s.QDy;
  QDz = s.QDz;
  jac = s.jac;
  shapeX = s.shapeX;
  shapeY = s.shapeY;
  shapeZ = s.shapeZ;
  material = s.material;
  ctrl_y = s.ctrl_y;
  ctrl_z = s.ctrl_z;
  theta = s.theta;
}

void Shape::vector_map_nojac()
{
  int npx = spaces[0].no_points;
  int npy = spaces[1].no_points;
  int npz = spaces[2].no_points;

  int xyz = npx * npy * npz;
  MatrixXd VDx = MatrixXd::Zero(xyz, xyz);
  MatrixXd VDy = MatrixXd::Zero(xyz, xyz);
  MatrixXd VDz = MatrixXd::Zero(xyz, xyz);
  for (int i = 1; i <= npx; i++)
  {
    for (int j = 1; j <= npy; j++)
    {
      for (int k = 1; k <= npz; k++)
      {

        int I = ((i - 1) * npy * npz) + ((j - 1) * npz) + k;
        for (int l = 1; l <= npx; l++)
        {
          int J = ((l - 1) * npy * npz) + ((j - 1) * npz) + k;
          VDx(J - 1, I - 1) += spaces[0].V(l - 1, i - 1);
          VDx(J - 1, I - 1) += spaces[0].V(l - 1, i - 1);
          QDx(J - 1, I - 1) += spaces[0].Q1(l - 1, i - 1);
        }

        for (int l = 1; l <= npy; l++)
        {
          int J = ((i - 1) * npy * npz) + ((l - 1) * npz) + k;
          VDy(J - 1, I - 1) += spaces[1].V(l - 1, j - 1);
          QDy(J - 1, I - 1) += spaces[1].Q1(l - 1, j - 1);
        }

        for (int l = 1; l <= npz; l++)
        {
          int J = ((i - 1) * npy * npz) + ((j - 1) * npz) + l;
          VDz(J - 1, I - 1) += spaces[2].V(l - 1, k - 1);
          QDz(J - 1, I - 1) += spaces[2].Q1(l - 1, k - 1);
        }
      }
    }
  }

  VD = VDx * VDy * VDz;
}
