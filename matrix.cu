#include "matrix.h"

Matrix::Matrix(int x,int y)
{
	dim_x = x;
	dim_y = y;
	
	//M = new float[dim_x*dim_y];
	cudaMallocManaged(&M,dim_x*dim_y*sizeof(float));
}
