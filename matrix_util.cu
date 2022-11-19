#include "matrix_util.h"
#include<iostream>


using namespace std;

Matrix::Matrix(int x,int y)
{
	dim_x = x;
	dim_y = y;
	//M = new float[dim_x*dim_y];
	cudaMallocManaged(&M,dim_x*dim_y*sizeof(float));
}
Matrix::~Matrix()
{
	cudaFree(M);
}
Matrix& Matrix::operator=(const Matrix& other)
{
	//std::cout<<"here ? \n";
	if(this == &other)
		return *this;
	if((other.dim_x == this->dim_x) and (other.dim_y == this->dim_y))
	{
		// maybe a cuda kernel should be called here 
		cudaMemcpy(this->M,other.M,dim_x*dim_y*sizeof(float),cudaMemcpyDeviceToDevice);
	}
	else
	{
		cout<<"you have a problem in copy \n";
	}
	return *this;
}
void Matrix::print_matrix()
{
	for(int i=0;i<dim_x;i++)  
	{
		for(int j=0;j<dim_y;j++)
		{
			cout<<i<<"\t"<<j<<"\t"<<M[j*dim_x+i]<<"\n";
		}
	}
}

void Matrix::print_size()
{
	cout<<"("<<dim_x<<","<<dim_y<<")\n";
}
/* WRAPPER FUNCTIONS */

void matrix_add_gpu(Matrix* A, Matrix* B, Matrix* C)
{
	int block_size = 32;

	if(!((A->dim_x == B->dim_x) and (A->dim_y == B->dim_y) and (B->dim_x == C->dim_x)))
	{
		cout<<"error dimension miss match in add \n"<<"\n";
		cout<<A->dim_x<<"\t"<<A->dim_y<<"\n";
		cout<<B->dim_x<<"\t"<<B->dim_y<<"\n";
		cout<<C->dim_x<<"\t"<<C->dim_y<<"\n";
	}

	int n_blocks_x=(C->dim_x+block_size-1)/block_size; 
	int n_blocks_y=(C->dim_y+block_size-1)/block_size; 
	//cout<<n_blocks_x<<"\t"<<n_blocks_y<<"\n";

	dim3 n_blocks(n_blocks_x,n_blocks_y);
	dim3 n_threads(block_size,block_size);
	//cout<<"time\t"<<time(0)<<"\n";

	matrix_add <<< n_blocks,n_threads >>> (A->M,B->M,C->M,A->dim_x,A->dim_y,B->dim_x,B->dim_y,C->dim_x,C->dim_y);
	cudaDeviceSynchronize();
}
void matrix_multiply_gpu(Matrix* A, Matrix* B, Matrix* C)
{
	int block_size = 32;

	if(!((A->dim_x == B->dim_y) and (A->dim_y == C->dim_y) and (B->dim_x == C->dim_x)))
	{
		cout<<"error dimension miss match in multiply \n"<<"\n";
		cout<<A->dim_x<<"\t"<<A->dim_y<<"\n";
		cout<<B->dim_x<<"\t"<<B->dim_y<<"\n";
		cout<<C->dim_x<<"\t"<<C->dim_y<<"\n";
	}

	int n_blocks_x=(C->dim_x+block_size-1)/block_size; 
	int n_blocks_y=(C->dim_y+block_size-1)/block_size; 
	//cout<<n_blocks_x<<"\t"<<n_blocks_y<<"\n";

	dim3 n_blocks(n_blocks_x,n_blocks_y);
	dim3 n_threads(block_size,block_size);
	//cout<<"time\t"<<time(0)<<"\n";

	matrix_multiply <<< n_blocks,n_threads >>> (A->M,B->M,C->M,A->dim_x,A->dim_y,B->dim_x,B->dim_y,C->dim_x,C->dim_y);
	cudaDeviceSynchronize();

}
void matrix_scalar_product_gpu(Matrix* A, float alpha)
{
	/* this is just a rescaling of the matrix elements with alpha */
	int block_size = 32;


	int n_blocks_x=(A->dim_x+block_size-1)/block_size; 
	int n_blocks_y=(A->dim_y+block_size-1)/block_size; 
	//cout<<n_blocks_x<<"\t"<<n_blocks_y<<"\n";

	dim3 n_blocks(n_blocks_x,n_blocks_y);
	dim3 n_threads(block_size,block_size);
	//cout<<"time\t"<<time(0)<<"\n";

	matrix_scalar_product<<< n_blocks,n_threads >>> (A->M,A->dim_x,A->dim_y,alpha); 
	cudaDeviceSynchronize();

}
void matrix_hadamard_product_gpu(Matrix* A, Matrix* B, Matrix* C)
{
	/* this is just element wise product
	   A,B and C has to have the same dimensions. */
	int block_size = 32;

	if(!((A->dim_x == B->dim_x) and (A->dim_y == B->dim_y) and (B->dim_x == C->dim_x) and (B->dim_y == C->dim_y)))
	{
		cout<<"error dimension miss match hadamard \n"<<"\n";
	}

	int n_blocks_x=(C->dim_x+block_size-1)/block_size; 
	int n_blocks_y=(C->dim_y+block_size-1)/block_size; 
	//cout<<n_blocks_x<<"\t"<<n_blocks_y<<"\n";

	dim3 n_blocks(n_blocks_x,n_blocks_y);
	dim3 n_threads(block_size,block_size);
	//cout<<"time\t"<<time(0)<<"\n";

	matrix_hadamard_product<<< n_blocks,n_threads >>> (A->M,B->M,C->M,A->dim_x,A->dim_y); 
	cudaDeviceSynchronize();

}
void matrix_multiply_add_gpu(Matrix* A, Matrix* B,Matrix* C, Matrix* D)
{
	int block_size = 32;
	/* 
                 dim_x
	  	 _____
		|     |
		|     |
	dim_y	|     |
		|     |
		|_____|

	*/
	if(!((A->dim_x == B->dim_y) and (A->dim_y == D->dim_y) and (B->dim_x == D->dim_x)))
	{
		cout<<"error dimension miss match multiply add \n"<<"\n";
	}
	if(!((C->dim_x == D->dim_x) and (C->dim_y == D->dim_y)))
	{
		cout<<"error dimension miss match 2 \n"<<"\n";
	}

	int n_blocks_x=(C->dim_x+block_size-1)/block_size; 
	int n_blocks_y=(C->dim_y+block_size-1)/block_size; 
	//cout<<n_blocks_x<<"\t"<<n_blocks_y<<"\n";

	dim3 n_blocks(n_blocks_x,n_blocks_y);
	dim3 n_threads(block_size,block_size);
	//cout<<"time\t"<<time(0)<<"\n";

	matrix_multiply_add <<< n_blocks,n_threads >>> (A->M,B->M,C->M,D->M,A->dim_x,A->dim_y,B->dim_x,B->dim_y,C->dim_x,C->dim_y,D->dim_x,D->dim_y);
	cudaDeviceSynchronize();
}
void matrix_transpose_gpu(Matrix* A, Matrix* A_transpose)
{
	int block_size = 32;
	/* 
	   MATRIX A            MATRIX A_TRANSPOSE 
                 dim_x                    dim_y
	  	 _____        		 _________
		|     |     		|         |
		|     |     	 dim_x	|         |
	dim_y	|     |         	|_________|
		|     |     
		|_____|     

	*/
	int n_blocks_x=(A->dim_x+block_size-1)/block_size; 
	int n_blocks_y=(A->dim_y+block_size-1)/block_size; 
	//cout<<n_blocks_x<<"\t"<<n_blocks_y<<"\n";

	if(!(A->dim_x == A_transpose->dim_y) or !(A->dim_y == A_transpose->dim_x))
	{
		cout<<"error dimension miss match \n"<<"\n";
	}
	dim3 n_blocks(n_blocks_x,n_blocks_y);
	dim3 n_threads(block_size,block_size);
	//cout<<"time\t"<<time(0)<<"\n";

	matrix_transpose <<< n_blocks,n_threads >>> (A->M,A_transpose->M,A->dim_x,A->dim_y);
	cudaDeviceSynchronize();
}



/* GPU CODES */


__global__
void matrix_add(float *A, float *B, float *C,  int A_dim_x, int A_dim_y, int B_dim_x, int B_dim_y, int C_dim_x, int C_dim_y)
{
	int col = blockDim.x*blockIdx.x+threadIdx.x;
	int row = blockDim.y*blockIdx.y+threadIdx.y;
	if(col < C_dim_x and row < C_dim_y)
	{
		C[row*C_dim_x+col] = A[row*A_dim_x+col]+B[row*A_dim_x+col];
	}

}

__global__
void matrix_scalar_product(float * A, int A_dim_x, int A_dim_y, float alpha)
{
	int col = blockDim.x*blockIdx.x+threadIdx.x;
	int row = blockDim.y*blockIdx.y+threadIdx.y;
	// this function should do C = A*B 
	if(col < A_dim_x and row < A_dim_y)
	{
		A[row*A_dim_x+col] = alpha*A[row*A_dim_x+col];
	}
}
__global__
void matrix_hadamard_product(float * A, float * B, float *C, int A_dim_x, int A_dim_y)
{
	int col = blockDim.x*blockIdx.x+threadIdx.x;
	int row = blockDim.y*blockIdx.y+threadIdx.y;
	// this function should do C = A*B 
	if(col < A_dim_x and row < A_dim_y)
	{
			C[row*A_dim_x+col]  = A[row*A_dim_x+col]*B[row*A_dim_x+col];
	}
}
__global__
void matrix_multiply(float * A, float * B, float *C, int A_dim_x, int A_dim_y, int B_dim_x, int B_dim_y, int C_dim_x, int C_dim_y)
{
	int col = blockDim.x*blockIdx.x+threadIdx.x;
	int row = blockDim.y*blockIdx.y+threadIdx.y;
	// this function should do C = A*B 
	if(col < C_dim_x and row < C_dim_y)
	{
		float sum =0. ;
		for(int k=0;k<A_dim_x;k++) // A->dim_x should be the same as B->dim_y
		{
			 sum += A[row*A_dim_x+k]*B[k*B_dim_x+col];
		}

		C[row*C_dim_x+col] = sum;
	}
}
__global__
void matrix_multiply_add(float * A, float * B, float *C, float * D, int A_dim_x, int A_dim_y, int B_dim_x, int B_dim_y, int C_dim_x, int C_dim_y,int D_dim_x, int D_dim_y)
{
	int col = blockDim.x*blockIdx.x+threadIdx.x;
	int row = blockDim.y*blockIdx.y+threadIdx.y;
	// this function should do D = A*B+C 
	// D and C should have the same dimensions.
	//row=col=1;
	if(col < C_dim_x and row < C_dim_y)
	{
		float sum =0. ;
		for(int k=0;k<A_dim_x;k++) // A->dim_x should be the same as B->dim_y
		{
			 sum += A[row*A_dim_x+k]*B[k*B_dim_x+col];
		}

		D[row*C_dim_x+col] = sum+C[row*D_dim_x+col];
	}
}
__global__
void matrix_transpose(float * A, float * A_transpose, int A_dim_x, int A_dim_y)
{
	int col = blockDim.x*blockIdx.x+threadIdx.x;
	int row = blockDim.y*blockIdx.y+threadIdx.y;
	if(col < A_dim_x and row < A_dim_y)
	{
		A_transpose[col*A_dim_y+row] = A[row*A_dim_x+col];
	}
}

/* This following function is only for checking 
   */
void matrix_multiply_add_cpu(Matrix* A, Matrix* B, Matrix* C, Matrix* D)
{
	//int col = blockDim.x*blockIdx.x+threadIdx.x;
	//int row = blockDim.y*blockIdx.y+threadIdx.y;
	//if(col < C->dim_x and row < C->dim_y)
	for(int col=0;col<D->dim_x;col++)
	{
		for(int row=0;row<D->dim_y;row++)
		{
			for(int k=0;k<A->dim_x;k++) // A->dim_x should be the same as B->dim_y
			{
				D->M[row*D->dim_x+col] += A->M[row*A->dim_x+k]*B->M[k*B->dim_x+col];
				//cout<<row<<"\t"<<col<<"\t"<<k<<"\t"<<A->M[row*A->dim_x+k]*B->M[k*B->dim_x+col]<<"\t"<<A->M[row*A->dim_x+k]<<"\t"<<B->M[k*B->dim_x+col]<<"\n";
			}
			D->M[row*D->dim_x+col] += C->M[row*C->dim_x+col] ; 
		}
	}
}
