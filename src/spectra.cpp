#include <math.h>
#include <chrono>
#include <limits>
#include <algorithm>
#include <vector>

#include "omp.h"
#include "consts.h"
#include "helpers.h"
#include "mkl.h"
#include "FGM.h"
#include "Shape.h"
#include "Space.h"
#include "Material.h"

void debug() {};
void printMatrix(const Ref<const Matrix3d> t){
	cout << "rows: " << t.rows() << " cols: " << t.cols() << endl;
	cout << t.coeff(0, 0) << endl;
	cout << "ea" << endl;
}
#define CPUID 0
#define GPUID 1



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

#endif
                                                                                                                                                                                 
int ttt;
#pragma omp threadprivate(ttt)

struct DesignParameters
{
	double theta1 = -1;
	double theta2 = -1;
	DesignParameters(double _theta, double _theta2)
	{
		theta1 = _theta;
		theta2 = _theta2;
	}
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
		xi2 = atoi(argv[3]);
		xi3 = atoi(argv[4]);
		eta1 = atoi(argv[5]);
		eta2 = atoi(argv[6]);
		eta3 = atoi(argv[7]);
		zeta1 = atoi(argv[8]);
		zeta2 = atoi(argv[9]);
		zeta3 = atoi(argv[10]);
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
	mkl_set_num_threads(1);
//	omp_set_nested(1);

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
	Material tfirst(2.1e+9, 0.34, 7.8358e+8, 7.08e+12, 7.08e+12, 7.08e+12, 1.9445e+12, 1.9445e+12, 3.0128e+12,0.175,0.175,0.175,Material::CNT_TYPE::UD,Material::POROUS_TYPE::AA);
	Material tsecond(2.1e+9, 0.34, 7.0e+10, 7.0e+10, 7.0e+10, 7.0e+10, 2.6119e+10, 2.6119e+10, 2.6119e+10,0.34,0.34,0.34,Material::CNT_TYPE::UD,Material::POROUS_TYPE::AA);
	Material tthird(2.1e+9, 0.34, 7.8358e+8, 7.08e+12, 7.08e+12, 7.08e+12, 2.6119e+10, 2.6119e+10, 3.0128e+12,0.175,0.175,0.175,Material::CNT_TYPE::UD,Material::POROUS_TYPE::AA);
	Shape *shps = new Shape[num_layers];
	cout<<num_layers<<endl;
//        Shape tshape1(3, 3, 0.15, xi1, eta1, zeta1, tfirst, 0.1, 0,0.5236,-0.25,0, 0);
//	return 0;
	shps[0] = Shape(3,3,0.15,xi1,eta1,zeta1,tfirst, 0.1, 0, 0.5236,-0.25,0,0);

//	Shape tshape2(3, 3, 0.15, xi2, eta2, zeta2, tsecond, 0, 0,0,0,0,0.15);
	shps[1] = Shape (3, 3, 0.15, xi2, eta2, zeta2, tsecond, 0, 0,0,0,0,0.15);
	//Shape tshape3(3, 3, 0.15, xi3, eta3, zeta3, tthird, 0.1, 0,1.0472,-0.25,0,0.45);
	shps[2] = Shape (3, 3, 0.15, xi3, eta3, zeta3, tthird, 0.1, 0,1.0472,-0.25,0,0.45);
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
	
	TaskMemoryReq task_memory_reqs;
	string fileName = "TASK.config";//Take this as input maybe?
	read_config(task_memory_reqs, fileName);
	int no_gpus = atoi(argv[2]);
	GPUManager gpu_mans[no_gpus];
	for (int i = 0; i<no_gpus; i++){
		gpuErrchk(cudaSetDevice(i));
		size_t remaining_memory, total_memory = 0;
		cudaMemGetInfo(&remaining_memory, &total_memory);		
		int deviceID;
		cudaGetDevice(&deviceID);
		if(deviceID != i){
			cout<<"device id is not equal to the ith index" <<endl;
			exit(1);
		}
		gpu_mans[i] = GPUManager(remaining_memory,deviceID); 
		#pragma omp parallel num_threads(nthreads)
		{
			ttt = omp_get_thread_num();
			gpu_mans[i].create_cuda_stream(ttt);	
			gpu_mans[i].create_cublas_handle(ttt);
			gpu_mans[i].create_cusolver_handle(ttt);
		}
		cout << "GPU "<< deviceID << " manager is created"<<endl;
	}
	bool failed = false;
#endif


	Material first(2.1e+9, 0.34, 7.8358e+8, 5.6466e+12, 7.08e+12, 7.08e+12, 1.9445e+12, 1.9445e+12, 2.9030e+12,0.175,0.175,0.2194,Material::CNT_TYPE::FGO,Material::POROUS_TYPE::AA, 0.17, 1400, 1150, 0.149, 1.381, 1.381, 0.6733, 0.3);
	Material second(0, 0, 0, 0.0354e+6, 0.0354e+6, 655.87e+6, 0.0266e+6, 92.463e+6, 141.12e+6,0.999856,0,0,Material::CNT_TYPE::UD,Material::POROUS_TYPE::AA,0.17,1400,1150,0.149,1.381,1.381,0.6733,0.3);
	Material third(2.1e+9, 0.34, 7.8358e+8, 5.6466e+12, 7.08e+12, 7.08e+12, 1.9445e+12, 1.9445e+12, 2.9030e+12,0.175,0.175,0.2194,Material::CNT_TYPE::FGO,Material::POROUS_TYPE::AA,0.17,1400,1150,0.149,1.381,1.381,0.6733,0.3);

	//geometric properties
	double thickness = 0.1;
	double curvature = 0.0;
	double asp_Rat1 = 2;
	double Lr1 = 1;

	int count = 0;
	vector<DesignParameters> problems;
	
	for(double theta = -90; theta<=90; theta+=10)
	{
		for(double theta2 = -90; theta2<=90; theta2+=10)
		{
			problems.push_back(DesignParameters(theta * pi/180 ,theta2*pi/180));
		}
	}


			//problems.push_back(DesignParameters(-60 * pi/180 ,60*pi/180));

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
	double best_theta1 = -1, best_theta2 = -1;
	double ostart = omp_get_wtime();

#pragma omp parallel for schedule(dynamic,1) num_threads(nthreads) 
	for (int i = 0; i < problems.size(); i++)
	{
		Shape *shapes = new Shape[num_layers];
		double start = omp_get_wtime();
		int nconv = 0;
		double mineig = 0;
        	Shape shape1(Lr1, asp_Rat1, thickness/4, xi1, eta1, zeta1, first, 0.1, 0,problems[i].theta1,curvature, 0,0);
        	shapes[0] = shape1;	
     	 	Shape shape2(Lr1, asp_Rat1, thickness/2, xi2, eta2, zeta2, second, 0, 0,0,curvature,0,thickness/4);
		shapes[1] = shape2;
		Shape shape3(Lr1, asp_Rat1, thickness/4, xi3, eta3, zeta3, third, 0.1, 0,problems[i].theta2,curvature,0,3*thickness/4);
		shapes[2] = shape3;
		#ifdef GPU
		FGM fgm = FGM(num_layers, shapes, gpu_mans, no_gpus,task_memory_reqs);
		#endif
		#ifndef GPU
		FGM fgm = FGM(num_layers, shapes);
		#endif
		fgm.compute(10, 60, nconv, mineig, 0.01, 100, 0.01);
		double end = omp_get_wtime();

#pragma omp critical
{
		if (nconv > 0)
		{
			cout << i << " " << fgm.shapes[0].theta << " " << fgm.shapes[2].theta << " " << mineig << " " << end - start << endl;
			if (mineig < smallest_mineig)
			{
				smallest_mineig = mineig;
				best_theta1 = fgm.shapes[0].theta;
				best_theta2 = fgm.shapes[2].theta;
			}
		}
		else
		{
			cout << "No eigen: " << fgm.shapes[0].ctrl_y << " " << fgm.shapes[0].ctrl_z << " " << end - start << endl;
		}
		delete [] shapes;
}
	}
	
	double shift = 0.001;
	double nat_freq = sqrt(smallest_mineig-shift)/2/pi;
	cout << endl
		<< "Result:" << endl;
	cout << "Smallest eig: " << smallest_mineig << " nat freq "<< nat_freq<<" - (" << best_theta1 << ", " << best_theta2 << ")" << endl;
	double oend = omp_get_wtime();
	cout << "Time spent on computation: " << oend - ostart << endl;
	cout << "*******************************************************************" << endl;
	cout << "Total time: " << oend - pstart << endl;
	cout << "*******************************************************************" << endl;

	#pragma omp parallel num_threads(nthreads)
	{
		for(int i = 0; i<no_gpus; i++){
			gpu_mans[i].destroy_cublas_handle(ttt);
			gpu_mans[i].destroy_cusolver_handle(ttt);
			gpu_mans[i].destroy_cuda_stream(ttt);
		}
	}
	return 0;
}

