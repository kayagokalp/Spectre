#pragma once
#include "helpers.h"
#include "Material.h"
#include "Space.h"




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

		double zeta_add;

		int xyz;
		MatrixXd VD;
		MatrixXd QDx;
		MatrixXd QDy;
		MatrixXd QDz;


		Tensor<double,3> shapeX; 
		Tensor<double,3> shapeY;
		Tensor<double,3> shapeZ;

		Tensor<double,3> jac;
		double theta;

		Shape() : dim{0, 0, 0}, is_curved(false), curve{0, 0}, xyz(0),zeta_add(0), theta(0) {}
		
		Shape(dtype x_dim, dtype y_dim, dtype z_dim,
				int x_sample, int y_sample, int z_sample, 
				Material mat, double ctrl_y, double ctrl_z,double _theta, 
				dtype xcurve = 0, dtype ycurve = 0,double zeta_add=0);

		void fillShapeTensors();

		void vector_map_jac_curvature(double alpha, double beta);

		void operator=(const Shape& s);

		void vector_map_nojac();
};

