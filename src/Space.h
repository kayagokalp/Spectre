#pragma once
#include "helpers.h"



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
		Space(dtype start, dtype end, int no_points);

		void discretize();
};
