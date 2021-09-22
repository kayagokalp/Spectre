#include <math.h>
#include <chrono>
#include <limits>
#include <algorithm>
#include <vector>

#include "omp.h"
#include "consts.h"
#include "helpers.h"
#include "mkl.h"

void debug() {};
void printMatrix(const Ref<const Matrix3d> t){
	cout << "rows: " << t.rows() << " cols: " << t.cols() << endl;
	cout << t.coeff(0, 0) << endl;
	cout << "ea" << endl;
}
#define CPUID 0
#define GPUID 1

class FGM;

#ifdef SMART
#define MAX_TASKS 4
#define MAX_GPUS 4
#define MAX_CPUS 32
#define PADDING 32
#define SAMPLE_SIZE 2

double gloads[MAX_GPUS];
double cloads[MAX_CPUS];

omp_lock_t glocks[MAX_GPUS];
omp_lock_t clocks[MAX_CPUS];

double gcosts[PADDING * MAX_GPUS][MAX_TASKS]{}; //= {{0.6,1,0.74,1}, {1,1,1,1}, {1,1,1,1}, {1,1,1,1}};
//std::fill(*gcosts, *gcosts + M*N, 1);
double ccosts[MAX_TASKS] = {4.32, 0.95, 5.35, 0.52}; //can also be extended to costs per CPU

#define GPU_MULT 8
#endif

#ifdef GPU
#define MAX_THREADS_NO 128
cublasHandle_t handle[MAX_THREADS_NO];
cudaStream_t stream[MAX_THREADS_NO];
cusolverDnHandle_t dnhandle[MAX_THREADS_NO];

int ttt;
#pragma omp threadprivate(ttt)

struct rinfo
{
	double *gpu_mem = nullptr;
	int gpu_id;
	int no_elements, no_bytes;

	rinfo(int gpu_id, int x1, int y1, int z1, int x2, int y2, int z2, int x3, int y3, int z3)  : gpu_id(gpu_id)
	{
		int nnxyz = (3 * x1 * y1 * z1) + (3 * x2 * y2 * z2) + (3 * x3 * y3 * z3);
		no_elements = 10 * nnxyz * nnxyz;
       		no_bytes = no_elements * sizeof(double);		
		gpuErrchk(cudaMalloc((void**)&gpu_mem, no_bytes));
	}

	//rinfo(int gpu_id, int x, int y, int z) : gpu_id(gpu_id)
	//{
	//	int nxyz = x * y * z;
	//	no_elements = 75 * nxyz * nxyz;
	//	no_bytes = no_elements * sizeof(double);
	//	gpuErrchk(cudaMalloc((void **)&gpu_mem, no_bytes));
	//}
};

rinfo **rinfos;
#endif

struct Space
{
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

		Space() {}
		Space(dtype start, dtype end, int no_points)
			: start(start), end(end), no_points(no_points),
			IT(no_points, no_points), FT(no_points, no_points), D(no_points, no_points), s(no_points)
	{
		IT.setZero();
		FT.setZero();
		D.setZero();
		s.setZero();
		V.setZero();
		Q1.setZero();
		discretize();
	}

		void discretize()
		{
			cheb(no_points, IT, FT);
			DBG(cout << "IT\n"
					<< IT << endl;);
			DBG(cout << "FT\n"
					<< FT << endl;);
			derivative(start, end, no_points, D);
			DBG(cout << "D\n"
					<< D << endl;);
			slobat(start, end, no_points, s);
			DBG(cout << "s\n"
					<< s << endl;);
			inner_product_helper(start, end, no_points, V);
			DBG(cout << "V\n"
					<< V << endl;);
			Q1 = IT * D * FT;
			DBG(cout << "Q1\n"
					<< Q1 << endl;);
			 //cout << "IT" << " Sum: " << IT.sum() << " Max: " << IT.maxCoeff() << " Min: " << IT.minCoeff() << endl;
			 //cout << "FT" << " Sum: " << FT.sum() << " Max: " << FT.maxCoeff() << " Min: " << FT.minCoeff() << endl;
			 //cout << "D" << " Sum: " << D.sum() << " Max: " << D.maxCoeff() << " Min: " << D.minCoeff() << endl;
			// cout << "s" << " Sum: " << s.sum() << " Max: " << s.maxCoeff() << " Min: " << s.minCoeff() << endl;
			// cout << "V" << " Sum: " << V.sum() << " Max: " << V.maxCoeff() << " Min: " << V.minCoeff() << endl;
			// cout << "Q1" << " Sum: " << Q1.sum() << " Max: " << Q1.maxCoeff() << " Min: " << Q1.minCoeff() << endl;
		}
};

class Material
{
	public:
		Material(): mod_elasticity(0), poisson_ratio(0), density(0) {}
		Material(dtype _mod_elasticity,
				dtype _poisson_ratio,
				dtype _density)
			: mod_elasticity(_mod_elasticity),
			poisson_ratio(_poisson_ratio),
			density(_density) {}

		void operator=(const Material& s){
			mod_elasticity = s.mod_elasticity;
			poisson_ratio = s.poisson_ratio;
			density = s.density;
		}
		//member variables
		dtype mod_elasticity;
		dtype poisson_ratio;
		dtype density;
};

class Shape
{
	public:

		bool is_curved;
		dtype curve[2];
		dtype dim[3];
		
		int debug_count = 0;

		Space spaces[3];
		Material material;

		double ctrl_y;
		double ctrl_z;

		int xyz;
		MatrixXd VD;
		MatrixXd QDx;
		MatrixXd QDy;
		MatrixXd QDz;


		Tensor<double,3> shapeX; 
		Tensor<double,3> shapeY;
		Tensor<double,3> shapeZ;

		Tensor<double,3> jac;

		Shape() : dim{0, 0, 0}, is_curved(false), curve{0, 0}, xyz(0) {}
		Shape(dtype x_dim, dtype y_dim, dtype z_dim,
				int x_sample, int y_sample, int z_sample, 
				Material mat, double ctrl_y, double ctrl_z,
				dtype xcurve = 0, dtype ycurve = 0) : dim{x_dim, y_dim, z_dim}, curve{xcurve, ycurve},
			is_curved(~(xcurve == 0 && ycurve == 0)),
			spaces{Space(-x_dim / 2, x_dim / 2, x_sample), Space(-y_dim / 2, y_dim / 2, y_sample), Space(0, z_dim, z_sample)},
			xyz(x_sample * y_sample * z_sample),
			VD(xyz, xyz),
			QDx(xyz, xyz), QDy(xyz, xyz), QDz(xyz, xyz), material(mat), 
			ctrl_y(ctrl_y), ctrl_z(ctrl_z)
			{
				QDx.setZero();
				QDy.setZero();
				QDz.setZero();
				//vector_map_nojac();	
				shapeX = Tensor<double,3>(z_sample, y_sample, z_sample);
				shapeY = Tensor<double,3>(z_sample, y_sample, z_sample);
				shapeZ = Tensor<double,3>(z_sample, y_sample, z_sample);
				shapeX.setZero();
		 		shapeY.setZero();
				shapeZ.setZero();
				
				jac = Tensor<double,3>(x_sample,y_sample,z_sample);
				fillShapeTensors();
				double alpha = -1.0472; //NOTE(KAYA) : for testing...
				double beta = 0;
				vector_map_jac_curvature(alpha,beta);				
				// cout << "VD" << " Sum: " << VD.sum() << " Max: " << VD.maxCoeff() << " Min: " << VD.minCoeff() << endl;
				// cout << "QDx" << " Sum: " << QDx.sum() << " Max: " << QDx.maxCoeff() << " Min: " << QDx.minCoeff() << endl;
				// cout << "QDy" << " Sum: " << QDy.sum() << " Max: " << QDy.maxCoeff() << " Min: " << QDy.minCoeff() << endl;
				// cout << "QDz" << " Sum: " << QDz.sum() << " Max: " << QDz.maxCoeff() << " Min: " << QDz.minCoeff() << endl;
			}

		void fillShapeTensors(){	
			for(int i = 0; i < spaces[0].no_points; i++){
				for(int j = 0; j < spaces[1].no_points; j++){
					for(int k = 0; k < spaces[2].no_points; k++){
						shapeX(i,j,k) = spaces[0].s(i);
						shapeY(i,j,k) = spaces[1].s(j);
						shapeZ(i,j,k) = spaces[2].s(k);
					}
				}
			}
		}
		void vector_map_jac_curvature(double alpha, double beta){

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

			cout<<"SHAPEX SUM = "<<shapeX.sum()<<endl;
			cout<<"SHAPEY SUM = "<<shapeY.sum()<<endl;
			cout<<"SHAPEZ SUM = "<<shapeZ.sum()<<endl;

			cout<<"TEMPX SUM = "<<tempx.sum()<<endl;
			cout<<"TEMPY SUM = "<<tempy.sum()<<endl;
			cout<<"TEMPZ SUM = "<<tempz.sum()<<endl;

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
						dydyb = cos(beta * tempy(i,j,k)) - (beta * cos(beta * tempy(i,j,k)) * tempz(i,j,k));
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

		//cout<<"dxdxb2 SUM "<<dxdxb2.sum()<<endl;
		//cout<<"dxdyb2 SUM "<<dxdyb2.sum()<<endl;
		//cout<<"dxdzb2 SUM "<<dxdzb2.sum()<<endl;

		//cout<<"dydxb2 SUM "<<dydxb2.sum()<<endl;
		//cout<<"dydyb2 SUM "<<dydyb2.sum()<<endl;
		//cout<<"dydzb2 SUM "<<dydzb2.sum()<<endl;

		//cout<<"dzdxb2 SUM "<<dzdxb2.sum()<<endl;
		//cout<<"dzdyb2 SUM "<<dzdyb2.sum()<<endl;
		//cout<<"dzdzb2 SUM "<<dzdzb2.sum()<<endl;

		cout<<"dxidx SUM: " << dxidx.sum()<<endl;
		cout<<"dxidy SUM: " << dxidy.sum()<<endl;
		cout<<"dxidz SUM: " << dxidz.sum()<<endl;

		cout<<"detadx SUM: " << detadx.sum()<<endl;
		cout<<"detady SUM: " << detady.sum()<<endl;
		cout<<"detadz SUM: " << detadz.sum()<<endl;

		cout<<"dzetadx SUM: " << dzetadx.sum()<<endl;
		cout<<"dzetady SUM: " << dzetady.sum()<<endl;
		cout<<"dzetadz SUM: " << dzetadz.sum()<<endl;

//		cout<<"jac SUM: "<< jac.sum() <<endl;	
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

	
			cout<<"QDxi_dxidx MAX: " << QDxi_dxidx.maxCoeff() << " SUM: "<< QDxi_dxidx.sum()<<endl;
			cout<<"QDxi_dxidy MAX: " << QDxi_dxidy.maxCoeff() << " SUM: "<< QDxi_dxidy.sum()<<endl;
			cout<<"QDxi_dxidz MAX: " << QDxi_dxidz.maxCoeff() << " SUM: "<< QDxi_dxidz.sum()<<endl;

//			cout<<"QDeta_detadx MAX: " << QDeta_detadx.maxCoeff() << " SUM: "<< QDeta_detadx.sum()<<endl;
//			cout<<"QDeta_detady MAX: " << QDeta_detady.maxCoeff() << " SUM: "<< QDeta_detady.sum()<<endl;
//			cout<<"QDeta_detadz MAX: " << QDeta_detadz.maxCoeff() << " SUM: "<< QDeta_detadz.sum()<<endl;


//			cout<<"QDzeta_dzetadx MAX: " << QDzeta_dzetadx.maxCoeff() << " SUM: "<< QDzeta_dzetadx.sum()<<endl;
//			cout<<"QDzeta_dzetady MAX: " << QDzeta_dzetady.maxCoeff() << " SUM: "<< QDzeta_dzetady.sum()<<endl;
//			cout<<"QDzeta_dzetadz MAX: " << QDzeta_dzetadz.maxCoeff() << " SUM: "<< QDzeta_dzetadz.sum()<<endl;

			QDx = QDxi_dxidx+QDeta_detadx+QDzeta_dzetadx;
			QDy = QDxi_dxidy+QDeta_detady+QDzeta_dzetady;
			QDz = QDxi_dxidz+QDeta_detadz+QDzeta_dzetadz;

			//cout<<QDxi_dxidx<<endl;
		//	cout<<"QDx MAX: " << QDx.maxCoeff() << " SUM: "<< QDx.sum()<<endl;
		//	cout<<"QDy MAX: " << QDy.maxCoeff() << " SUM: "<< QDy.sum()<<endl;
		//	cout<<"QDz MAX: " << QDz.maxCoeff() << " SUM: "<< QDz.sum()<<endl;

//			cout<<"Q1xi MAX: " << spaces[0].Q1.maxCoeff() << " SUM: "<< spaces[0].Q1.sum()<<endl;
//			cout<<"Q2xi MAX: " << spaces[1].Q1.maxCoeff() << " SUM: "<< spaces[1].Q1.sum()<<endl;
//			cout<<"Q3xi MAX: " << spaces[2].Q1.maxCoeff() << " SUM: "<< spaces[2].Q1.sum()<<endl;
		}

		void operator=(const Shape& s){
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
		}

		void vector_map_nojac()
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
};

//Functionally graded material
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
		Tensor<double, 3> * mu;
		Tensor<double, 3> * lame;
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
		FGM(unsigned int n_shapes, Shape * shps) 
		{
			num_shapes = n_shapes;
			shapes = shps;
			np = new unsigned int * [num_shapes];
			nxyz = new int[num_shapes];
			nnxyz = 0;
			mu = new Tensor<double, 3>[num_shapes];
			lame = new Tensor<double, 3>[num_shapes];
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
			//jac = shps.jac;
			for(unsigned int i = 0; i < n_shapes; i++){
				np[i] = new unsigned int[3];
				for(unsigned int j = 0; j < 3; j++){
					np[i][j] = shapes[i].spaces[j].no_points;
				}

				JAC[i] = shapes[i].jac;
				nxyz[i] = np[i][0] * np[i][1] * np[i][2];
				nnxyz += nxyz[i];
				mu[i] = Tensor<double, 3>(np[i][0], np[i][1], np[i][2]);
				mu[i].setZero();
				lame[i] = Tensor<double, 3>(np[i][0], np[i][1], np[i][2]);
				lame[i].setZero();
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
				FG_var_MT_CNT(i);
				//FG_var_MT(i);
				// cout << rho[i](0, 0, 0) << endl;
				// cout << rho[i](0, 0, 1) << endl;
				// cout << rho[i](0, 0, 2) << endl;
				// cout << rho[i](0, 0, 3) << endl;
				// cout << rho[i](0, 0, 4) << endl;
				// cout << rho[i](0, 0, 5) << endl;
				// cout << rho[i](0, 0, 6) << endl;
				// cout << "rho" << i << " Sum: " << rho[i].sum() << " Max: " << rho[i].maximum() << " Min: " << rho[i].minimum() << endl;
				// cout << "***********************" << endl;
				// cout << mu[i](0, 0, 0) << endl;
				// cout << mu[i](0, 0, 1) << endl;
				// cout << mu[i](0, 0, 2) << endl;
				// cout << mu[i](0, 0, 3) << endl;
				// cout << mu[i](0, 0, 4) << endl;
				// cout << mu[i](, 0, 5) << endl;
				// cout << mu[i](0, 0, 6) << endl;
				// cout << "***********************" << endl;
				// cout << lame[i](0, 0, 0) << endl;
				// cout << lame[i](0, 0, 1) << endl;
				// cout << lame[i](0, 0, 2) << endl;
				// cout << lame[i](0, 0, 3) << endl;
				// cout << lame[i](0, 0, 4) << endl;
				// cout << lame[i](0, 0, 5) << endl;
				// cout << lame[i](0, 0, 6) << endl;
				inner_product(i);
				// cout << "VD_mu" << i << " Sum: " << VD_mu[i].sum() << " Max: " << VD_mu[i].maxCoeff() << " Min: " << VD_mu[i].minCoeff() << endl;
				// cout << "VD_lame" << i << " Sum: " << VD_lame[i].sum() << " Max: " << VD_lame[i].maxCoeff() << " Min: " << VD_lame[i].minCoeff() << endl;
				// cout << "VD_rho" << i << " Sum: " << VD_rho[i].sum() << " Max: " << VD_rho[i].maxCoeff() << " Min: " << VD_rho[i].minCoeff() << endl;
			}

			//initialize MM and KK
			MM = MatrixXd(M[0].rows() + M[1].rows() + M[2].rows(), M[0].cols() + M[1].cols() + M[2].cols());
			MM.setZero();    
			KK = MatrixXd(K[0].rows() + K[1].rows() + K[2].rows(), K[0].cols() + K[1].cols() + K[2].cols());
			KK.setZero();
			//
		}

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
		//   inner_product();
		// }

#ifdef GPU
		void T1_system_matrices_GPU(unsigned int l)
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

			const double tu = 2.0;
			const double van = 1.0;
			const double ziro = 0.0;

			double *d_VD_lame, *d_VD_mu, *d_epx, *d_epy, *d_epz, *d_gammaxy, *d_gammayz, *d_gammaxz, *d_epxyz, *d_temp_K, *d_K, *d_temp, *gpu_mem;
			gpu_mem = rinfos[ttt]->gpu_mem;

			int nc = epx.cols();
			d_VD_lame = gpu_mem;                    //1
			d_VD_mu = d_VD_lame + VD_lame[l].size();   //1
			d_epx = d_VD_mu + VD_mu[l].size();         //3
			d_epy = d_epx + epx.size();             //3
			d_epz = d_epy + epy.size();             //3
			d_gammaxy = d_epz + epz.size();         //3
			d_gammaxz = d_gammaxy + gammaxy.size(); //3
			d_gammayz = d_gammaxz + gammaxz.size(); //3
			d_epxyz = d_gammayz + gammayz.size();   //3
			d_temp_K = d_epxyz + epx.size();        //9
			d_K = d_temp_K + (nc * nc);             //9
			d_temp = d_K + (nc * nc);               //3

			cudaMemcpy(d_VD_lame, VD_lame[l].data(), VD_lame[l].size() * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_VD_mu, VD_mu[l].data(), VD_mu[l].size() * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_epx, epx.data(), epx.size() * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_epy, epy.data(), epy.size() * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_epz, epz.data(), epz.size() * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_gammaxy, gammaxy.data(), gammaxy.size() * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_gammaxz, gammaxz.data(), gammaxz.size() * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_gammayz, gammayz.data(), gammayz.size() * sizeof(double), cudaMemcpyHostToDevice);

			cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, epx.rows(), epx.cols(), &van, d_epx, epx.rows(), &van, d_epy, epy.rows(), d_epxyz, epx.rows());
			cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, epx.rows(), epx.cols(), &van, d_epxyz, epx.rows(), &van, d_epz, epz.rows(), d_epxyz, epx.rows()); //(x + y + z)

			cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, VD_lame[l].rows(), nc, VD_lame[l].cols(), &van, d_VD_lame, VD_lame[l].rows(), d_epxyz, epx.rows(), &ziro, d_temp, VD_lame[l].rows()); //VD_lame * epxyz
			cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epx.rows(), &van, d_epxyz, epx.rows(), d_temp, VD_lame[l].rows(), &ziro, d_temp_K, nc);                              //epxyzT * VD_lame * epxyz
			cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &ziro, d_K, nc, &van, d_temp_K, nc, d_K, nc);

			cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, VD_mu[l].rows(), nc, VD_mu[l].cols(), &van, d_VD_mu, VD_mu[l].rows(), d_epx, epx.rows(), &ziro, d_temp, VD_mu[l].rows()); //VD_mu * epx
			cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epx.rows(), &van, d_epx, epx.rows(), d_temp, VD_mu[l].rows(), &ziro, d_temp_K, nc);                      //epxT * VD_mu * epx
			cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &tu, d_temp_K, nc, d_K, nc);

			cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, VD_mu[l].rows(), nc, VD_mu[l].cols(), &van, d_VD_mu, VD_mu[l].rows(), d_epy, epy.rows(), &ziro, d_temp, VD_mu[l].rows()); //VD_mu * epy
			cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epy.rows(), &van, d_epy, epy.rows(), d_temp, VD_mu[l].rows(), &ziro, d_temp_K, nc);                      //epyT * VD_mu * epy
			cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &tu, d_temp_K, nc, d_K, nc);

			cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, VD_mu[l].rows(), nc, VD_mu[l].cols(), &van, d_VD_mu, VD_mu[l].rows(), d_epz, epz.rows(), &ziro, d_temp, VD_mu[l].rows()); //VD_mu * epz
			cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, epz.rows(), &van, d_epz, epz.rows(), d_temp, VD_mu[l].rows(), &ziro, d_temp_K, nc);                      //epzT * VD_mu * epz
			cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &tu, d_temp_K, nc, d_K, nc);

			cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, VD_mu[l].rows(), nc, VD_mu[l].cols(), &van, d_VD_mu, VD_mu[l].rows(), d_gammaxy, gammaxy.rows(), &ziro, d_temp, VD_mu[l].rows()); //VD_mu * gammaxy
			cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammaxy.rows(), &van, d_gammaxy, gammaxy.rows(), d_temp, VD_mu[l].rows(), &ziro, d_temp_K, nc);                  //gammaxy * VD_mu * gammaxy
			cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc);

			cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, VD_mu[l].rows(), nc, VD_mu[l].cols(), &van, d_VD_mu, VD_mu[l].rows(), d_gammaxz, gammaxz.rows(), &ziro, d_temp, VD_mu[l].rows()); //VD_mu * gammaxz
			cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammaxz.rows(), &van, d_gammaxz, gammaxz.rows(), d_temp, VD_mu[l].rows(), &ziro, d_temp_K, nc);                  //gammaxz * VD_mu * gammaxz
			cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc);

			cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, VD_mu[l].rows(), nc, VD_mu[l].cols(), &van, d_VD_mu, VD_mu[l].rows(), d_gammayz, gammayz.rows(), &ziro, d_temp, VD_mu[l].rows()); //VD_mu * gammayz
			cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, gammayz.rows(), &van, d_gammayz, gammayz.rows(), d_temp, VD_mu[l].rows(), &ziro, d_temp_K, nc);                  //gammayz * VD_mu * gammayz
			cublasDgeam(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, &van, d_K, nc, &van, d_temp_K, nc, d_K, nc);
			cudaStreamSynchronize(stream[ttt]);
			K[l] = MatrixXd::Zero(nc, nc);
			cudaMemcpy(K[l].data(), d_K, nc * nc * sizeof(double), cudaMemcpyDeviceToHost);
		}
#endif

		void T1_system_matrices_CPU(unsigned int l)
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

			//MatrixXd epxyz = epx + epy + epz;

			MatrixXd sigx = (VD_lame11[l] * epx) + (VD_lame12[l] * epy) + (VD_lame13[l] * epz) + (VD_lame14[l] * gammaxy);
			MatrixXd sigy = (VD_lame12[l] * epx) + (VD_lame22[l] * epy) + (VD_lame23[l] * epz) + (VD_lame24[l] * gammaxy);
			MatrixXd sigz = (VD_lame13[l] * epx) + (VD_lame23[l] * epy) + (VD_lame33[l] * epz) + (VD_lame34[l] * gammaxy);
			MatrixXd tauxy = (VD_lame44[l] * gammaxy) + (VD_lame14[l] * epx) + (VD_lame24[l] * epy) + (VD_lame34[l] * epz);
			MatrixXd tauyz = (VD_lame55[l] * gammayz) + (VD_lame56[l] * gammaxz);
			MatrixXd tauxz = (VD_lame66[l] * gammaxz) + (VD_lame56[l] * gammayz);
			cout<< " QDX " << shapes[l].QDx.sum() << endl;
			cout<< " QDY " << shapes[l].QDy.sum() << endl;
			cout<< " QDZ " << shapes[l].QDz.sum() << endl;
			cout<< " SIGx "<< sigx.sum() << "MAX: "<< sigx.maxCoeff()<<endl;
			cout<< " SIGy "<< sigy.sum() << "MAX: "<< sigy.maxCoeff()<<endl;
			cout<< " SIGz "<< sigz.sum() <<	"MAX: "<< sigz.maxCoeff()<<endl;
			cout<< " tauxy "<< tauxy.sum() << "MAX: "<< tauxy.maxCoeff()<<endl;
			cout<< " tauyz "<< tauyz.sum() << "MAX: "<< tauyz.maxCoeff()<<endl;
			cout<< " tauxz "<< tauxz.sum() << "MAX: "<< tauxz.maxCoeff()<<endl;
		        cout<< " epx " << epx.sum() << endl;
			cout<< " epy " << epy.sum() << endl;
			cout<< " epz " << epz.sum() << endl;
			cout<< " gammaxy " << gammaxy.sum() << endl;	

			K[l] = (epx.transpose() * sigx) + (epy.transpose() * sigy) + (epz.transpose() * sigz) + (gammaxy.transpose() * tauxy) + (gammaxz.transpose() * tauxz) + (gammayz.transpose() * tauyz);

			cout<< "K " << K[l].sum() << " MAX: "<< K[l].maxCoeff()<<endl;

//			K[l] = (epxyz.transpose() * (VD_lame[l] * epxyz)) +
//				2 * ((epx.transpose() * (VD_mu[l] * epx)) +
//						(epy.transpose() * (VD_mu[l] * epy)) +
//						(epz.transpose() * (VD_mu[l] * epz))) +
//				(gammaxy.transpose() * (VD_mu[l] * gammaxy)) +
//				(gammaxz.transpose() * (VD_mu[l] * gammaxz)) +
//				(gammayz.transpose() * (VD_mu[l] * gammayz));
		}

		bool T1_system_matrices(unsigned int l)
		{
#ifdef SMART
			int decision = GPUID;
			int tid = omp_get_thread_num();
			int cid = sched_getcpu();
			int gid = rinfos[tid]->gpu_id;

			if (gloads[gid] > (cloads[cid] + ccosts[0]) * GPU_MULT)
			{
				decision = CPUID;
			}

			if (decision == GPUID)
			{
				//cout << tid << " decision 1 GPU - " << gloads[gid] << " " << cloads[cid] << endl;

				omp_set_lock(&glocks[gid]);
				gloads[gid] += gcosts[PADDING * gid][0];
				omp_unset_lock(&glocks[gid]);

				T1_system_matrices_GPU(l);

				omp_set_lock(&glocks[gid]);
				gloads[gid] -= gcosts[PADDING * gid][0];
				omp_unset_lock(&glocks[gid]);
				return true;
			}
			else
			{
				//cout << tid << " decision 1 CPU - " << gloads[gid] << " " << cloads[cid] << endl;

				omp_set_lock(&clocks[cid]);
				cloads[cid] += ccosts[0];
				omp_unset_lock(&clocks[cid]);

				T1_system_matrices_CPU(l);

				omp_set_lock(&clocks[cid]);
				cloads[cid] -= ccosts[0];
				omp_unset_lock(&clocks[cid]);
				return false;
			}
#elif defined GPU
			T1_system_matrices_GPU(l);
			return true;
#else
			T1_system_matrices_CPU(l);
			return false;
#endif
		}


		void T2_svd_CPU(MatrixXd &BC, MatrixXd &V)
		{
			JacobiSVD<MatrixXd> svd(BC, ComputeFullV);
			V = svd.matrixV();
		}

#ifdef GPU
		void T2_svd_GPU(MatrixXd &BC, MatrixXd &V)
		{
			double * gpu_mem = rinfos[ttt]->gpu_mem;
			//dnhandle[ttt] = NULL;
			//stream[ttt] ;
			//nu13.setZero();= NULL;
			//cusolverDnHandle_t cusolverH = NULL;	
			gesvdjInfo_t gesvdj_params = NULL;

			cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
			cudaError_t cudaStat1 = cudaSuccess;
			cudaError_t cudaStat2 = cudaSuccess;
			cudaError_t cudaStat3 = cudaSuccess;
			cudaError_t cudaStat4 = cudaSuccess;
			cudaError_t cudaStat5 = cudaSuccess;
			const int m = BC.rows();
			const int n = BC.cols();

			//double A[m*n]; 
			//double U[m*m];
			//double V[n*n];
			//double S[n];

			double *d_A = gpu_mem;
			double *d_S = d_A + m * n;
			double *d_U = d_S + n;
			double *d_V = d_U + m * m;
			int *d_info = (int *)(d_V + n * n);
			int lwork = 0;
			double *d_work = (double *)(d_info + n);
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
			status = cusolverDnDgesvdj_bufferSize(dnhandle[ttt], jobz, econ, m, n, d_A, m, d_S, d_U, m, d_V, n, &lwork, gesvdj_params);
			assert(CUSOLVER_STATUS_SUCCESS == status);

			cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
			assert(cudaSuccess == cudaStat1);
			
			//Step 5: compute SVD
			status = cusolverDnDgesvdj(dnhandle[ttt], jobz, econ, m, n, d_A, m, d_S, d_U, m, d_V, n, d_work, lwork, d_info, gesvdj_params);
			cudaStat1 = cudaDeviceSynchronize(); // ???
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

			//free resources ??
			//if(d_A) cudaFree(d_A);
			//if(d_S) cudaFree(d_S);
			//if(d_U) cudaFree(d_U);
			//if(d_V) cudaFree(d_V);
			//if(d_info) cudaFree(d_info);
			//if(d_work) cudaFree(d_work);
			//
		}
#endif

		bool T2_svd(MatrixXd &BC, MatrixXd &V)
		{
#ifdef SMART
			int decision = GPUID;
			int tid = omp_get_thread_num();
			int cid = sched_getcpu();
			int gid = rinfos[tid]->gpu_id;

			if (gloads[gid] > (cloads[cid] + ccosts[1]) * GPU_MULT)
			{
				decision = CPUID;
			}
	
			if (decision == GPUID)
			{
				omp_set_lock(&glocks[gid]);
				gloads[gid] += gcosts[PADDING * gid][1];
				omp_unset_lock(&glocks[gid]);

				T2_svd_GPU(BC, V);

				omp_set_lock(&glocks[gid]);
				gloads[gid] -= gcosts[PADDING * gid][1];
				omp_unset_lock(&glocks[gid]);
				return true;
			}
			else
			{
				omp_set_lock(&clocks[cid]);
				cloads[cid] += ccosts[1];
				omp_unset_lock(&clocks[cid]);

				T2_svd_CPU(BC, V);

				omp_set_lock(&clocks[cid]);
				cloads[cid] -= ccosts[1];
				omp_unset_lock(&clocks[cid]);
				return false;
			}
#elif defined GPU
			T2_svd_GPU(BC, V);
			return true;
#else
			T2_svd_CPU(BC, V);
			return false;
#endif
		}

#ifdef GPU
		void T3_mul_inv_GPU(MatrixXd &a0, MatrixXd &P)
		{
//cout << "Before r: " << a0.rows() << " c: " << a0.cols() << " s: " << a0.sum() << endl;
			const double van = 1.0;
			const double ziro = 0.0;
//debug();
			double *gpu_mem = rinfos[ttt]->gpu_mem;
			double *d_K, *d_M, *d_P, *d_K_phy, *d_M_phy, *d_a0, *d_temp, *d_M_phy_i, *d_work;
			int *d_pivot, *d_info, Lwork;

			int nc = P.cols();
			d_K = gpu_mem;                            //9
			//d_M = d_K + (9 * nnxyz * nnxyz);            //9
			d_M = d_K + (KK.rows()*KK.cols());            //9
			//d_P = d_M + (9 * nnxyz * nnxyz);            //max 9
			d_P = d_M + (MM.rows()*MM.cols());            //max 9
			//d_K_phy = d_P + (3 * nnxyz * nc);          //max 9
			d_K_phy = d_P + (P.rows()*P.cols());          //max 9
			d_M_phy = d_K_phy + (nc * nc);            //max 9
			d_a0 = d_M_phy + (nc * nc);               //max 9
			d_temp = d_a0 + (nc * nc);                //max 9
			//d_M_phy_i = d_temp + (3 * nnxyz * nc);     //max 9
			d_M_phy_i = d_temp + (3 * KK.rows() * nc);     //max 9
			d_pivot = (int *)(d_M_phy_i + (nc * nc)); //max 1
			d_info = d_pivot + nc;                    //max 1
			d_work = (double *)(d_info + nc);         //max 1.

			cudaMemcpy(d_K, KK.data(), KK.size() * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_M, MM.data(), MM.size() * sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_P, P.data(), P.size() * sizeof(double), cudaMemcpyHostToDevice);
			MatrixXd Id = MatrixXd::Identity(nc, nc);
			cudaMemcpy(d_M_phy_i, Id.data(), Id.size() * sizeof(double), cudaMemcpyHostToDevice);

			cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, KK.rows(), nc, KK.cols(), &van, d_K, KK.rows(), d_P, P.rows(), &ziro, d_temp, KK.rows()); //alpha * K * P + beta * K
			cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, P.rows(), &van, d_P, P.rows(), d_temp, KK.rows(), &ziro, d_K_phy, P.cols());   //Pt * K * P
			cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, MM.rows(), nc, MM.cols(), &van, d_M, MM.rows(), d_P, P.rows(), &ziro, d_temp, MM.rows()); //M * P
			cublasDgemm(handle[ttt], CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, P.rows(), &van, d_P, P.rows(), d_temp, MM.rows(), &ziro, d_M_phy, P.cols());   //Pt * M * P
			cusolverDnDgetrf_bufferSize(dnhandle[ttt], nc, nc, d_M_phy, nc, &Lwork);
			cusolverDnDgetrf(dnhandle[ttt], nc, nc, d_M_phy, nc, d_work, d_pivot, d_info);
			cusolverDnDgetrs(dnhandle[ttt], CUBLAS_OP_N, nc, nc, d_M_phy, nc, d_pivot, d_M_phy_i, nc, d_info);
			cublasDgemm(handle[ttt], CUBLAS_OP_N, CUBLAS_OP_N, nc, nc, nc, &van, d_M_phy_i, nc, d_K_phy, nc, &ziro, d_a0, nc); //M_phy_i * K_phy
			cudaStreamSynchronize(stream[ttt]);

			//MatrixXd at(nc, nc);
			//a0 = at;
			cudaMemcpy(a0.data(), d_a0, nc * nc * sizeof(double), cudaMemcpyDeviceToHost);
		}
#endif

		void T3_mul_inv_CPU(MatrixXd &a0, MatrixXd &P)
		{
			MatrixXd K_phy = P.transpose() * (KK * P);
			MatrixXd M_phy = P.transpose() * (MM * P);
			a0 = M_phy.inverse() * K_phy;
		}

		bool T3_mul_inv(MatrixXd &a0, MatrixXd &P)
		{
#ifdef SMART
			int decision = GPUID;
			int tid = omp_get_thread_num();
			int cid = sched_getcpu();
			int gid = rinfos[tid]->gpu_id;

			if (gloads[gid] > (cloads[cid] + ccosts[2]) * GPU_MULT)
			{
				decision = CPUID;
			}

			if (decision == GPUID)
			{
				omp_set_lock(&glocks[gid]);
				gloads[gid] += gcosts[PADDING * gid][2];
				omp_unset_lock(&glocks[gid]);

				T3_mul_inv_GPU(a0, P);

				omp_set_lock(&glocks[gid]);
				gloads[gid] -= gcosts[PADDING * gid][2];
				omp_unset_lock(&glocks[gid]);
				return true;
			}
			else
			{
				omp_set_lock(&clocks[cid]);
				cloads[cid] += ccosts[2];
				omp_unset_lock(&clocks[cid]);

				T3_mul_inv_CPU(a0, P);

				omp_set_lock(&clocks[cid]);
				cloads[cid] -= ccosts[2];
				omp_unset_lock(&clocks[cid]);
				return false;
			}
#elif defined GPU
			T3_mul_inv_GPU(a0, P);
			return true;
#else
			T3_mul_inv_CPU(a0, P);
			return false;
#endif
		}

		void T4_eigen(MatrixXd &a0, int &nconv, double &small_eig)
		{
			MatrixXd MMM = a0;
			DenseGenRealShiftSolve<double> op(MMM);
			GenEigsRealShiftSolver<DenseGenRealShiftSolve<double>> eigs(op, 10, 50, 0);

			eigs.init();
			nconv = eigs.compute();

			Eigen::VectorXcd evalues;
			if (eigs.info() == CompInfo::Successful)
			{
				evalues = eigs.eigenvalues();
				small_eig = evalues(nconv - 1).real();
			}
		}

		void compute_gpu_costs(const int noeigs, const int ncv, int &nconv, double &small_eig, const double shift = 0.01, const int max_iter = -1, const double tol = -1, int g_id = 0, int sample_size = 1)
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

		void compute_cpu_costs(const int noeigs, const int ncv, int &nconv, double &small_eig,
				const double shift = 0.01, const int max_iter = -1, const double tol = -1, const int sample_size = 1)
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

		void removeDuplicateRows(MatrixXd &mat){
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

		MatrixXd removeZeroRows(MatrixXd &mat){
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

		void prepareBoundaryCondition(MatrixXd &mat, unsigned int l){
			for(int i = 0; i < 3*np[l][0]; i++){
				if(i == 0){
					mat(seq(0, 0), seq(0, mat.cols()-1)).setZero(); 
				}
				if(i < 3*np[l][0]-1){
					mat(seq((i+1)*np[l][1], (i+1)*np[l][1]), seq(0, mat.cols()-1)).setZero(); 
				}
				mat(seq((i+1)*np[l][1]-1, (i+1)*np[l][1]-1), seq(0, mat.cols()-1)).setZero(); 
			}
			mat(seq(0, np[l][1]-1), seq(0, mat.cols()-1)).setZero();
			mat(seq(np[l][1]*(np[l][0]-1), np[l][1]*np[l][0]-1), seq(0, mat.cols()-1)).setZero();
			mat(seq(np[l][1]*np[l][0], np[l][1]*(np[l][0]+1)-1), seq(0, mat.cols()-1)).setZero();
			mat(seq(np[l][1]*(np[l][0]+np[l][1]-1),np[l][1]*(np[l][0]+np[l][1])-1), seq(0, mat.cols()-1)).setZero();	
			mat(seq(np[l][1]*(np[l][0]+np[l][1]-1), np[l][1]*(np[l][0]+np[l][1]+1)-1), seq(0, mat.cols()-1)).setZero();
			mat(seq(np[l][1]*(np[l][0]+np[l][1]+np[l][1]-1), np[l][1]*(np[l][0]+np[l][1]+np[l][1])-1), seq(0, mat.cols()-1)).setZero();
		}
		void compute(const int noeigs, const int ncv, int &nconv, double &small_eig,
				const double shift = 0.01, const int max_iter = -1, const double tol = -1)
		{
			int tid = omp_get_thread_num();
			int copyStartX = 0;
			int copyStartY = 0;

			for(int i = 0; i < num_shapes; i++){
				double t1t = omp_get_wtime();
				bool gpu_load = T1_system_matrices(i);
				double cost = omp_get_wtime() - t1t;
				//cout << "M" << i << " Rows: " << M[i].rows() << " Cols: " << M[i].cols() << " Sum: " << M[i].sum() << " Max: " << M[i].maxCoeff() << " Min: " << M[i].minCoeff() << endl;
				//cout << "K" << i << " Rows: " << K[i].rows() << " Cols: " << K[i].cols() << " Sum: " << K[i].sum() << " Max: " << K[i].maxCoeff() << " Min: " << K[i].minCoeff() << endl;
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
				cout << "T1 (system-matrices) => GPU: " << gpu_load << " Cost: " << cost << " secs " << endl;
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
// cout << "r: " << BC_I_U.rows() << " c: " << BC_I_U.cols() << " s: " << BC_I_U.sum() << endl;
// cout << "r: " << BC_II_B.rows() << " c: " << BC_II_B.cols() << " s: " << BC_II_B.sum() << endl;
// cout << "r: " << BC_III_U.rows() << " c: " << BC_III_U.cols() << " s: " << BC_III_U.sum() << endl;
// cout << "r: " << BC_IV_B.rows() << " c: " << BC_IV_B.cols() << " s: " << BC_IV_B.sum() << endl;
// cout << "r: " << BC_V13.rows() << " c: " << BC_V13.cols() << " s: " << BC_V13.sum() << endl;
// cout << "r: " << BC_V2.rows() << " c: " << BC_V2.cols() << " s: " << BC_V2.sum() << endl;
// cout << "r: " << BC_VI13.rows() << " c: " << BC_VI13.cols() << " s: " << BC_VI13.sum() << endl;
// cout << "r: " << BC_VI2.rows() << " c: " << BC_VI2.cols() << " s: " << BC_VI2.sum() << endl;
// cout << "r: " << BC_VII13.rows() << " c: " << BC_VII13.cols() << " s: " << BC_VII13.sum() << endl;
// cout << "r: " << BC_VII2.rows() << " c: " << BC_VII2.cols() << " s: " << BC_VII2.sum() << endl;
// cout << "r: " << BC_VIII13.rows() << " c: " << BC_VIII13.cols() << " s: " << BC_VIII13.sum() << endl;
// cout << "r: " << BC_VIII2.rows() << " c: " << BC_VIII2.cols() << " s: " << BC_VIII2.sum() << endl;
//
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

			MatrixXd V(BC.cols(), BC.cols());
			double t2t = omp_get_wtime();
			bool ranOnGPU = T2_svd(BC, V);
			double cost = omp_get_wtime() - t2t;
			cout <<  "T2 (svd) => GPU: " << ranOnGPU << " Cost: " << cost << " secs" << endl;
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
//debug();
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
			cout <<  "T3 (Mul-and-Inv) => GPU: " << gpu_load << " Cost: " << cost << " secs" << endl;
			double t4t = omp_get_wtime();
			T4_eigen(a0, nconv, small_eig);
			cost = omp_get_wtime() - t4t;
#ifdef SMART
			ccosts[3] = cost;
#endif
			cout << "T4 (Eigen) => Cost: " << cost << " secs - nconv = " << nconv << endl;
		}

		MatrixXd beta_matrix_3d(MatrixXd &BC_3D, int xyz, unsigned int l)
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

		void FG_var_MT_CNT(unsigned int l){
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
	

			int current_selection = 3; //This will be removed added for testing switch in the matlab code
			double V_star_cnt = 0.14; //This will be removed added for testing.
			double ro_cnt = 1400; //This will be removed added for testing.
			double ro_m = 1150; //This will be removed added for testing.

			double h = 0.15; //w_over_h1 this will be removed added for testing.

			double nu12_cnt = 0.175; //This will be removed added for testing.
			double nu13_cnt = 0.175; //This will be removed added for testing.
			double nu23_cnt = 0.175; //This will be removed added for testing.
			double nu_m = 0.34; //This will be removed added for testing.

			double eta_star_1 = 0.15; //This will be removed added for testing.
			double eta_star_2 = 0.941; //This will be removed added for testing.
			double eta_star_3 = 0.941; //This will be removed added for testing.
			double E11_cnt = 7.08e+12; //This will be removed added for testing.
			double E22_cnt = 7.08e+12; //This will be removed added for testing.
			double E33_cnt = 7.08e+12; //This will be removed added for testing.
			double E_m = 2.1e+9; //This will be removed added for testing.


			double G12_cnt = 1.9445e+12; //This will be removed added for testing.
			double G13_cnt = 1.9445e+12; //This will be removed added for testing.
			double G23_cnt = 3.0128e+12; //This will be removed added for testing.
			double G_m = 7.8358e+8; //This will be removed added for testing.


			int current_selection2 = 0;
			double e0 = 0.3; //This will be removed added for testing.
			double em = 0.1633; //This will be removed added for testing.	

			double alpha = 0.6733; //This will be removed added for testing.
			double alpha_prime = 0.8205; //This will be removed added for testing.
			double theta = 0; //This will be removed added for testing. 

			for(int i = 0; i < x.size(); i++){
				for(int j = 0; j < y.size(); j++){
					for(int k = 0; k < z.size(); k++){
						switch (current_selection)
						{
							case 0:
								{
									V_cnt(i,j,k) =V_star_cnt; 		
								}break;

							case 1:
								{
									V_cnt(i,j,k) = (1 + 2 * (z(k) - h/2)/h)*V_star_cnt;
								}break;
							case 2:
								{
									V_cnt(i,j,k) = (1 - 2 * (z(k) - h/2)/h)*V_star_cnt;
								}break;
							case 3:
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

						switch(current_selection2)
						{
							case 0: 
								{
									G12(i,j,k)= G12_star(i,j,k) *(1-e0*cos(pi*(z(k)-h/2)/h));
									G23(i,j,k)= G23_star(i,j,k) *(1-e0*cos(pi*(z(k)-h/2)/h));
									G13(i,j,k)= G13_star(i,j,k) *(1-e0*cos(pi*(z(k)-h/2)/h));
									rho[l](i,j,k)= ro_star(i,j,k) *(1-em*cos(pi*(z(k)-h/2)/h));
									E11(i,j,k)= E_star_11(i,j,k)*(1-e0*cos(pi*(z(k)-h/2)/h));
									E22(i,j,k)= E_star_22(i,j,k)*(1-e0*cos(pi*(z(k)-h/2)/h));
									E33(i,j,k)= E_star_33(i,j,k)*(1-e0*cos(pi*(z(k)-h/2)/h));

								}break;
							case 1:
								{
									G12(i,j,k)= G12_star(i,j,k) *(1-e0*cos(pi*z(k)/(2*h)+pi/4));
                    							G23(i,j,k)= G23_star(i,j,k) *(1-e0*cos(pi*z(k)/(2*h)+pi/4));
                  							G13(i,j,k)= G13_star(i,j,k) *(1-e0*cos(pi*z(k)/(2*h)+pi/4));
                    							rho[l](i,j,k)=  ro_star(i,j,k) *(1-em*cos(pi*z(k)/(2*h)+pi/4));
                    							E11(i,j,k)= E_star_11(i,j,k)*(1-e0*cos(pi*z(k)/(2*h)+pi/4));
                   							E22(i,j,k)= E_star_22(i,j,k)*(1-e0*cos(pi*z(k)/(2*h)+pi/4));
                    							E33(i,j,k)= E_star_33(i,j,k)*(1-e0*cos(pi*z(k)/(2*h)+pi/4));
								}break;
							case 2:
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
			/*
			cout<< " V_cnt "<< V_cnt.sum() << endl;

			cout<< " nu12 " << nu12.sum() << endl;
		        cout<< " nu21 " << nu21.sum() << endl;
			cout<< " nu23 " << nu23.sum() << endl;
			cout<< " nu32 " << nu32.sum() << endl;
			cout<< " nu31 " << nu31.sum() << endl;
			cout<< " nu13 " << nu13.sum() << endl;
			cout<< " E11 "  << E11.sum() << endl;
			cout<< " E22 " << E22.sum() << endl;
			cout<< " E33 " << E33.sum() << endl; 	


			cout << " delta "<< delta.sum() << endl; 
			cout << " Q11 "<< Q11.sum() << endl;
			cout << " Q12 "<< Q12.sum() << endl;
			cout << " Q13 "<< Q13.sum() << endl;
			cout << " Q22 "<< Q22.sum() << endl;
			cout << " Q23 "<< Q23.sum() << endl;
			cout << " Q33 "<< Q33.sum() << endl;
			cout << " Q44 "<< Q44.sum() << endl;
			cout << " Q55 "<< Q55.sum() << endl;
			cout << " Q66 "<< Q66.sum() << endl;
			*/

				
			Q11T[l]=Q11*pow(cos(theta),4)+2*(Q12+2*Q44)*(pow(cos(theta),2)*pow(sin(theta),2))+Q22*pow(sin(theta),4);
			//cout<< " Q11T "<< Q11T[l].sum() <<endl;
			
			Q12T[l]=Q12*pow(cos(theta),4)+(Q11+Q22-4*Q44)*(pow(cos(theta),2)*pow(sin(theta),2))+Q12*pow(sin(theta),4);
			//cout<< " Q12T "<< Q12T[l].sum() << endl;
			
			Q13T[l]=Q13*pow(cos(theta),2)+Q23*pow(sin(theta),2);
			//cout<< " Q13T "<< Q13T[l].sum() << endl;

			Q14T[l]=(Q11-Q12-2*Q44)*(pow(cos(theta),3)*sin(theta))+(2*Q44+Q12-Q22)*(cos(theta)*pow(sin(theta),3));
			//cout<< " Q14T "<< Q14T[l].sum() << endl;

			Q22T[l]=Q22*pow(cos(theta),4)+2*(Q12+2*Q44)*(pow(cos(theta),2)*pow(sin(theta),2))+Q11*pow(sin(theta),4);
			//cout<< " Q22T "<< Q22T[l].sum() << endl;

			Q23T[l]=Q23*pow(cos(theta),2)+Q13*pow(sin(theta),2);
			//cout<< " Q23T "<< Q23T[l].sum() << endl;

			Q24T[l]=(Q12-Q22+2*Q44)*(pow(cos(theta),3)*sin(theta))+(Q11-Q12-2*Q44)*(cos(theta)*pow(sin(theta),3));
			//cout<< " Q24T "<< Q24T[l].sum() << endl;

			Q33T[l]=Q33;
			//cout<< " Q33T "<< Q33T[l].sum() << endl;

			Q34T[l]=(Q13-Q23)*(cos(theta)*sin(theta));
			//cout<< " Q34T "<< Q34T[l].sum() << endl;

			Q44T[l]=(Q11+Q22-2*Q12-2*Q44)*(pow(cos(theta),2)*pow(sin(theta),2))+Q44*(pow(cos(theta),4)+pow(sin(theta),4));
			//cout<< " Q44T "<< Q44T[l].sum() << endl;

			Q55T[l]=Q55*pow(cos(theta),2)+Q66*pow(sin(theta),2);
			//cout<< " Q55T "<< Q55T[l].sum() << endl;

			Q56T[l]=(Q66-Q55)*cos(theta)*sin(theta);
			//cout<< " Q56T "<< Q56T[l].sum() << endl;

			Q66T[l]=Q66*pow(cos(theta),2)+Q55*pow(sin(theta),2);
			//cout<< " Q66T "<< Q66T[l].sum() << endl;

		}

		void FG_var_MT(unsigned int l) //problem
		{
			VectorXd &x = shapes[l].spaces[0].s;
			VectorXd &y = shapes[l].spaces[1].s;
			VectorXd &z = shapes[l].spaces[2].s;

			double K_m = (shapes[0].material.mod_elasticity / 3) / (1 - 2 * shapes[0].material.poisson_ratio);
			double G_m = (shapes[0].material.mod_elasticity / 2) / (1 + shapes[0].material.poisson_ratio);

			double K_c = (shapes[1].material.mod_elasticity / 3) / (1 - 2 * shapes[1].material.poisson_ratio);
			double G_c = (shapes[1].material.mod_elasticity / 2) / (1 + shapes[1].material.poisson_ratio);

			double V_min = 0;
			double V_max = 1;

			//for matlab conversion - can be removed later
			double c = shapes[l].dim[2];
			double b = shapes[l].dim[1];
			double p = shapes[l].ctrl_y;
			double q = shapes[l].ctrl_z;
			double rho_m = shapes[0].material.density;
			double rho_c = shapes[1].material.density;

			for (int j = 0; j < np[l][1]; j++)
			{
				for (int k = 0; k < np[l][2]; k++)
				{
					//bu satirda funtion pointer olabilir
					//double vcijk = V_min + (V_max-V_min) * pow((z(k)/c), p) * pow((0.5+y(j)/b), q);
					double vcijk = V_min + (V_max - V_min) * pow(1 - (z(k) / c), p) * pow(1 - (z(k) / c), q);
					double vmijk = 1 - vcijk;
					double rhotemp = (rho_c * vcijk) + (rho_m * vmijk);
					//double K = K_m + (K_c - K_m) * vcijk / (1 + (1 - vcijk) * (3 * (K_c - K_m) / (3*K_m + 4*G_m)));
					//double f1 = G_m*(9*K_m+8*G_m)/(6*(K_m+2*G_m));
					//double G = G_m + (G_c-G_m) * vcijk/(1 + (1- vcijk)*( (G_c-G_m)/(G_m+f1)));
					//double eijk = 9*K*G/(3*K+G);
					//double poisijk = (3*K-2*G)/(2*(3*K+G));
					double eijk = (shapes[1].material.mod_elasticity * vcijk) + (shapes[0].material.mod_elasticity * vmijk);
					double poisijk = (shapes[1].material.poisson_ratio * vcijk) + (shapes[0].material.poisson_ratio * vmijk);
					double mutemp = eijk / (2 * (1 + poisijk));
					double lametemp = (2 * mutemp * poisijk) / (1 - 2 * poisijk);

					for (int i = 0; i < np[l][0]; i++)
					{
						rho[l](i, j, k) = rhotemp;
						mu[l](i, j, k) = mutemp;
						lame[l](i, j, k) = lametemp;
					}
				}
			}
		}

		void tensor3(VectorXd &v_d3Nt, MatrixXd &Sst, int n,
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

		void inner_helper(Tensor<double, 3> &Axyz, Tensor<double, 3> &Xadl, Tensor<double, 3> &Ybem, Tensor<double, 3> &Zcfn,
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


		void inner_product(unsigned int l)
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
			// cout << "Xadl" << " Sum: " << Xadl.sum() << " Max: " << Xadl.maximum() << " Min: " << Xadl.minimum() << endl;
			// cout << "Ybem" << " Sum: " << Ybem.sum() << " Max: " << Ybem.maximum() << " Min: " << Ybem.minimum() << endl;
			// cout << "Zcfn" << " Sum: " << Zcfn.sum() << " Max: " << Zcfn.maximum() << " Min: " << Zcfn.minimum() << endl;
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
/*
			cout<< "VD_lame11 " << VD_lame11[l].sum()<<endl;
			cout<< "VD_lame12 " << VD_lame12[l].sum()<<endl;
			cout<< "VD_lame13 " << VD_lame13[l].sum()<<endl;
			cout<< "VD_lame22 " << VD_lame22[l].sum()<<endl;
			cout<< "VD_lame33 " << VD_lame33[l].sum()<<endl;
			cout<< "VD_lame23 " << VD_lame23[l].sum()<<endl;
			cout<< "VD_lame44 " << VD_lame44[l].sum()<<endl;
			cout<< "VD_lame55 " << VD_lame55[l].sum()<<endl;
			cout<< "VD_lame66 " << VD_lame66[l].sum()<<endl;
			cout<< "VD_lame14 " << VD_lame14[l].sum()<<endl;
			cout<< "VD_lame24 " << VD_lame24[l].sum()<<endl;
			cout<< "VD_lame34 " << VD_lame34[l].sum()<<endl;
			cout<< "VD_lame56 " << VD_lame56[l].sum()<<endl;
			cout<< "VD_rho " << VD_rho[l].sum()<<endl;
*/
		}

		MatrixXd boundary_condition_3d(int xyz, int ol, unsigned int l)
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

};

// -ostream &operator<<(ostream &os, const Space &spc)
// -{
// -  os << spc.start << "\t" << spc.end << "\t" << spc.no_points;
// -  return os;
// -}
// -
// -ostream &operator<<(ostream &os, const Material &mat)
// -{
// -  os << mat.mod_elasticity << "\t" << mat.poisson_ratio << "\t" << mat.density;
// -  return os;
// -}
// -
// -ostream &operator<<(ostream &os, const Shape &shp)
// -{
// -  os << "\tDims  : " << shp.dim[0] << "\t" << shp.dim[1] << "\t" << shp.dim[2] << "\n"
// -     << "\tCurved: " << shp.curve[0] << "\t" << shp.curve[1] << "\n"
// -     << "\t\tX-space: " << shp.spaces[0] << "\n"
// -     << "\t\tY-space: " << shp.spaces[1] << "\n"
// -     << "\t\tZ-space: " << shp.spaces[2] << "\n";
// -  return os;
// -}
// -
// -ostream &operator<<(ostream &os, const FGM &fgm)
// -{
// -  os << "Shape -------------------------------------------\n"
// -     << fgm.shape
// -     << "Materials ---------------------------------------\n"
// -     << "\tMat 1: " << fgm.mats[0] << "\n"
// -     << "\tMat 2: " << fgm.mats[1] << "\n"
// -     << "Parameters --------------------------------------\n"
// -     << "\tCtrl : " << fgm.ctrl_y << "\t" << fgm.ctrl_z << "\n";
// -  cout << "-------------------------------------------------\n";
// -  return os;
// -}

//Fiber induced composite
class FIC
{
};
//Laminated composite
class LCO
{
};

int main(int argc, char **argv)
{	

unsigned int xi1 = 0, xi2 = 0, xi3 = 0;
unsigned int eta1 = 0, eta2 = 0, eta3 = 0;
unsigned int zeta1 = 0, zeta2 = 0, zeta3 = 0;

#if defined(SMART) || defined(GPU)
	if (argc < 12)
	{
		cout << "Usage: " << argv[0] << " nthreads ngpus xi1 xi2 xi3 eta1 eta2 eta3 zeta1 zeta2 zeta3" << endl;
		return 0;
	}else{
		xi1 = atoi(argv[3]);
		xi2 = atoi(argv[4]);
		xi3 = atoi(argv[5]);
		eta1 = atoi(argv[6]);
		eta2 = atoi(argv[7]);
		eta3 = atoi(argv[8]);
		zeta1 = atoi(argv[9]);
		zeta2 = atoi(argv[10]);
		zeta3 = atoi(argv[11]);
	}
#else
	if (argc < 11)
	{
		cout << "Usage: " << argv[0] << " nthreads xi1 xi2 xi3 eta1 eta2 eta3 zeta1 zeta2 zeta3" << endl;
		return 0;
	}else{	
		xi1 = atoi(argv[2]);
		cout << xi1 << endl;
		xi2 = atoi(argv[3]);
		cout << xi2 << endl;
		xi3 = atoi(argv[4]);
		cout << xi3 << endl;
		eta1 = atoi(argv[5]);
		cout << eta1 << endl;
		eta2 = atoi(argv[6]);
		cout << eta2 << endl;
		eta3 = atoi(argv[7]);
		cout << eta3 << endl;
		zeta1 = atoi(argv[8]);
		cout << zeta1 << endl;
		zeta2 = atoi(argv[9]);
		cout << zeta2 << endl;
		zeta3 = atoi(argv[10]);
		cout << zeta3 << endl;
	}
#endif
/*
#ifndef GPU
	if (argc < 5)
	{
		cout << "Usage: " << argv[0] << " xdim ydim zdim nthreads " << endl;
		return 0;
	}
#else
	if (argc < 6)
	{
		cout << "Usage: " << argv[0] << " xdim ydim zdim nthreads ngpus" << endl;
		return 0;
	}
#endif
*/
	cout << "*******************************************************************" << endl;
	cout << "*******************************************************************" << endl;
	cout << "Starting Preproccesing..." << endl;
	double pstart = omp_get_wtime();
	Eigen::setNbThreads(1);

	mkl_set_num_threads_local(1);

//	int xdim = atoi(argv[1]);
//	int ydim = atoi(argv[2]);
//	int zdim = atoi(argv[3]);
	int nthreads = atoi(argv[1]);
	omp_set_num_threads(nthreads);

#ifdef SMART
	for (int i = 0; i < MAX_GPUS; i++)
		gloads[i] = 0;
	for (int i = 0; i < MAX_CPUS; i++)
		cloads[i] = 0;
	for (int i = 0; i < MAX_GPUS; i++)
		omp_init_lock(&glocks[i]);
	for (int i = 0; i < MAX_CPUS; i++)
		omp_init_lock(&clocks[i]);
#endif

	unsigned int num_layers = 3;
// 	unsigned int xi1 = 12, xi2 = 12, xi3 = 12;
// 	unsigned int eta1 = 12, eta2 = 12, eta3 = 12;
// 	unsigned int zeta1 = 16, zeta2 = 12, zeta3 = 16;
 

 	//unsigned int xi1 = 5, xi2 = 5, xi3 = 5;
 	//unsigned int eta1 = 5, eta2 = 5, eta3 = 5;
 	//unsigned int zeta1 = 7, zeta2 = 5, zeta3 = 5;	

	//unsigned int xi1 = 7, xi2 = 7, xi3 = 7;
 	//unsigned int eta1 = 7, eta2 = 7, eta3 = 7;
 	//unsigned int zeta1 = 9, zeta2 = 7, zeta3 = 9;

	//unsigned int xi1 = 13, xi2 = 13, xi3 = 13;
 	//unsigned int eta1 = 13, eta2 = 13, eta3 = 13;
 	//unsigned int zeta1 = 7, zeta2 = 5, zeta3 = 7;

// 	unsigned int xi1 = 4, xi2 = 4, xi3 = 4;
// 	unsigned int eta1 = 4, eta2 = 4, eta3 = 4;
// 	unsigned int zeta1 = 5, zeta2 = 4, zeta3 = 5;
	//temp vars for cost calculation
	Material tfirst(1, 0.3, 1);
	Material tsecond(200.0 / 70.0, 0.3, 5700.0 / 2702.0);
	Shape * shps = new Shape[num_layers];
	Shape tshape1(3, 3, 0.15, xi1, eta1, zeta1, tfirst, 0.1, 0);
	shps[0] = tshape1;
	Shape tshape2(3, 3, 0.15, xi1, eta1, zeta1, tsecond, 0, 0);
	shps[1] = tshape2;
	Shape tshape3(3, 3, 0.15, xi1, eta1, zeta1, tfirst, 0.1, 0);
	shps[2] = tshape3;
	//set CPU costs for each task
	FGM tfgm(num_layers, shps);
	int tnconv;
	double tmineig;
	//tfgm.compute_cpu_costs(10, 60, tnconv, tmineig, 0.01, 100, 0.01, SAMPLE_SIZE);
	//
#ifdef SMART
	cout << "CPU costs: " << endl;
	for (int i = 0; i < 4; i++)
	{
		cout << "T" << i + 1 << ": " << ccosts[i] << endl;
	}
#endif

#ifdef GPU
	rinfos = new rinfo *[nthreads];

	int no_gpus = atoi(argv[2]);

	bool failed = false;
#pragma omp parallel num_threads(nthreads)
	{
		ttt = omp_get_thread_num();
#pragma omp critical
		{
			gpuErrchk(cudaSetDevice((ttt % no_gpus)));
			//cudaSetDeviceFlags(cudaDeviceScheduleYield);
			cudaFree(0);

			int deviceID;
			cudaGetDevice(&deviceID);
			if (deviceID != (ttt % no_gpus))
			{
				cout << "device ID is not equal to the given one " << endl;
				failed = true;
			}
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, deviceID);
			cout << "GPU " << deviceID << ": " << prop.name << " is assigned to " << ttt << endl;

			if (cublasCreate(&handle[ttt]) != CUBLAS_STATUS_SUCCESS)
			{
				std::cout << "blas handler initialization failed" << std::endl;
				failed = true;
			}

			if (cusolverDnCreate(&dnhandle[ttt]) != CUSOLVER_STATUS_SUCCESS)
			{
				std::cout << "solver handler initialization failed" << std::endl;
				failed = true;
			}

			cudaStreamCreate(&stream[ttt]);
			cublasSetStream(handle[ttt], stream[ttt]);
			cusolverDnSetStream(dnhandle[ttt], stream[ttt]);
			rinfos[ttt] = new rinfo(ttt % no_gpus, xi1, eta1, zeta1, xi2, eta2, zeta2, xi3, eta3, zeta3);

#ifdef SMART
			//set costs for each task per gpu
			if (ttt / no_gpus < 1)
			{
				int i = ttt % no_gpus;
				//tfgm.compute_gpu_costs(11, 60, tnconv, tmineig, 0.01, 100, 0.01, i, SAMPLE_SIZE);
				cout << "GPU" << i << " costs: " << endl;
				for (int j = 0; j < 4; j++)
				{
					gcosts[PADDING * i][j] = 1;
					cout << "T" << j + 1 << ": " << gcosts[PADDING * i][j] << endl;
				}
			}
			//
#endif
		}
	}
	if (failed)
	{
		exit(1);
	}
#endif


	Material first(70000000000.0, 0.33, 2700);
	Material second(200000000000.0, 0.33, 5700.0);

	//geometric properties
	double asp_Rat1 = 3;
	double Lr1 = 3;

	double min_ctrl_y = 0.1, max_ctrl_y = 0.10, min_ctrl_z = 0.1, max_ctrl_z = 0.10;
	double interval = 0.1;
	vector<FGM > problems;
	for (double cy = min_ctrl_y; cy <= max_ctrl_y; cy += interval)
	{
		for (double cz = min_ctrl_z; cz <= max_ctrl_z; cz += interval)
		{
			for(double h_a = 0.01; h_a <= 0.01; h_a += 0.01)
			{
				for(double h1_h2 = 0.1; h1_h2 <= 0.1; h1_h2 += 0.1){
					Shape * shapes = new Shape[num_layers];
					double ht = h_a * Lr1;
					double w_over_h2 = ht / (h1_h2 + 2);
					double w_over_h1 = h1_h2 * w_over_h2;
					double w_over_h3 = w_over_h1;
//					shapes[0] = Shape(Lr1, asp_Rat1, w_over_h1, xi1, eta1, zeta1, first, cy, 0);
//					shapes[1] = Shape(Lr1, asp_Rat1, w_over_h2, xi2, eta2, zeta2, second, 0, 0);
//					shapes[2] = Shape(Lr1, asp_Rat1, w_over_h3, xi3, eta3, zeta3, first, cz, 0);
        			 	Shape tshape1(3, 3, 0.15, xi1, eta1, zeta1, tfirst, 0.1, 0);
       					shapes[0] = tshape1;
	       	 			Shape tshape2(3, 3, 0.15, xi1, eta1, zeta1, tsecond, 0, 0);
		       		 	shapes[1] = tshape2;
			        	Shape tshape3(3, 3, 0.15, xi1, eta1, zeta1, tfirst, 0.1, 0);
				        shapes[2] = tshape3;
					problems.push_back(FGM(num_layers, shapes));
					//break;
				}
				//break;
			}
			//break;
		}
		//break;
	}

	cout << "Preprocessing ended." << endl;
	cout << "Time spent for preprocessing is " << omp_get_wtime() - pstart << endl;
	cout << "*******************************************************************" << endl;
	cout << "*******************************************************************" << endl;
	cout << endl;

	cout << "*******************************************************************" << endl;
	cout << "*******************************************************************" << endl;
	cout << "Starting Computation..." << endl;
	omp_set_num_threads(nthreads);
	cout << "No problems: " << problems.size() << endl;

	double smallest_mineig = std::numeric_limits<double>::max();
	double best_y = -1, best_z = -1;
	double ostart = omp_get_wtime();

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
	//{
	for (int i = 0; i < problems.size(); i++)
	{
		double start = omp_get_wtime();
		// FGM fgm(shape, first, second, problems[i].first, problems[i].second);
		int nconv;
		double mineig;
		FGM fgm = problems[i];
		fgm.compute(10, 60, nconv, mineig, 0.01, 100, 0.01);
		double end = omp_get_wtime();

#pragma omp critical
{
		if (nconv > 0)
		{
			cout << i << " " << fgm.shapes[0].ctrl_y << " " << fgm.shapes[0].ctrl_z << " " << mineig << " " << end - start << endl;
			if (mineig < smallest_mineig)
			{
				smallest_mineig = mineig;
				best_y = fgm.shapes[0].ctrl_y;
				best_z = fgm.shapes[0].ctrl_z;
			}
		}
		else
		{
			cout << "No eigen: " << fgm.shapes[0].ctrl_y << " " << fgm.shapes[0].ctrl_z << " " << end - start << endl;
		}
}
	}
	cout << endl
		<< "Result:" << endl;
	cout << "Smallest eig: " << smallest_mineig << " - (" << best_y << ", " << best_z << ")" << endl;
	double oend = omp_get_wtime();
	cout << "Time spent on computation: " << oend - ostart << endl;
	cout << "*******************************************************************" << endl;
	cout << "Total time: " << oend - pstart << endl;
	cout << "*******************************************************************" << endl;
#ifdef GPU
	for (int i = 0; i < nthreads; i++)
	{
		cudaStreamDestroy(stream[i]);
		cublasDestroy(handle[i]);
	}
#endif

	return 0;
}

