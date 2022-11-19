#ifndef MATRIX_UTIL_H
#define MATRIX_UTIL_H

class Matrix
{

	//float* M = nullptr;//new float[dim_x*dim_y];

	public : 
	int dim_x=0;	
	int dim_y=0;	
	float* M = nullptr;//new float[dim_x*dim_y];

	Matrix(int,int);
	~Matrix();
	Matrix& operator=(const Matrix& other);
	void print_matrix();
	void print_size();
};

__global__ void matrix_multiply(float*, float*, float*,int,int,int,int,int,int);
__global__ void matrix_multiply_add(float*, float*,float*,float*,int,int,int,int,int,int,int,int);
__global__ void matrix_transpose(float*, float*,int,int);
__global__ void matrix_add(float*, float*, float*,int,int,int,int,int,int);
__global__ void matrix_hadamard_product(float*, float*, float*,int,int);
__global__ void matrix_scalar_product(float*,int,int,float);

void matrix_multiply_add_cpu(Matrix *, Matrix *, Matrix *, Matrix *);
void matrix_multiply_gpu(Matrix *, Matrix *, Matrix *);
void matrix_hadamard_product_gpu(Matrix *, Matrix *, Matrix *);
void matrix_multiply_add_gpu(Matrix *, Matrix *,Matrix *, Matrix *);
void matrix_transpose_gpu(Matrix *, Matrix *);
void matrix_add_gpu(Matrix *, Matrix *, Matrix *);
void matrix_scalar_product_gpu(Matrix *,float);

#endif //MATRIX_UTIL_H
