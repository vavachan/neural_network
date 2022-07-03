#include "neural_network.h"
#include <cmath>
#include <iostream>

layer::layer(int input_S, int output_S)
{
	inp_size = input_S;
	out_size = output_S;
	Weight  = new Matrix(inp_size,out_size);
	dWeight = new Matrix(inp_size,out_size);
	Bias    = new Matrix(1,out_size);
	dBias   = new Matrix(1,out_size);


	/* for the time being we will initialize all the 
	   weigths and biases to zero, see what happens. */

	for(int i=0;i<Weight->dim_x;i++)  
	{
		for(int j=0;j<Weight->dim_y;j++)
		{
			Weight->M[j*Weight->dim_x+i] = 1.;//rand()%10;
			//B->M[j*sizeX+i] = rand()%10;
		}
	}
	for(int i=0;i<Bias->dim_x;i++)  
	{
		for(int j=0;j<Bias->dim_y;j++)
		{
			Bias->M[j*Bias->dim_x+i] = 2.;//rand()%10;
			//B.M[j*sizeX+i] = rand()%10;
		}
	}
}
void layer::forward(Matrix* input, Matrix* output)
{
	/* the dimension of the  input vector is input->dim_y ( basically a 1dim column vector)
	   the  dimension of the  out vector is correspondingly output->dim_y. 
   	   These vectors are given to the layer, the only job of the  layer is to 
	   compute output[i] = Weight[i][j]*input[j]+Bias[i] 
	   Therefore the Weight->dim_x has to match input->dim_y 
	   Such checks are  done here ? // why o why
	 */
	matrix_multiply_add_gpu(Weight,input,Bias,output);
}

sigmoid_layer::sigmoid_layer(int input_S, int output_S) : layer(input_S, output_S)
{
	;//layer::layer(int input_S, int output_S);
}


void sigmoid_layer::forward(Matrix* input, Matrix* output)
{
	layer::forward(input,output);

}

void sigmoid_layer_forward_gpu(Matrix* input, Matrix* output)
{
	int block_size = 32;

	if(!((input->dim_x == output->dim_x) and (input->dim_y == output->dim_y)))
	{
		std::cout<<"error dimension miss match \n"<<"\n";
	}

	int n_blocks_x=(input->dim_x+block_size-1)/block_size; 
	int n_blocks_y=(input->dim_y+block_size-1)/block_size; 
	//cout<<n_blocks_x<<"\t"<<n_blocks_y<<"\n";

	dim3 n_blocks(n_blocks_x,n_blocks_y);
	dim3 n_threads(block_size,block_size);
	//cout<<"time\t"<<time(0)<<"\n";

	sigmoid_function <<< n_blocks,n_threads >>> (input->M,output->M,input->dim_x,input->dim_y,output->dim_x,output->dim_y);
	cudaDeviceSynchronize();
}

__global__
void sigmoid_function(float *input, float *output, int input_dim_x, int input_dim_y, int output_dim_x, int output_dim_y)
{
	int col = blockDim.x*blockIdx.x+threadIdx.x;
	int row = blockDim.y*blockIdx.y+threadIdx.y;
	if(col < output_dim_x and row < output_dim_y)
	{
		output[row*output_dim_x+col] = 1/(1+exp(input[row*input_dim_x+col]));
	}

}

__global__ 
void mean_squared_error_2d(float *predictions, float *target, int size_x, int size_y, float *error)
{
	int col = blockDim.x*blockIdx.x+threadIdx.x;
	int row = blockDim.y*blockIdx.y+threadIdx.y;
	if(col < size_x and row < size_y)
	{
		//output[row*output_dim_x+col] = 1/(1+exp(input[row*input_dim_x+col]));
		atomicAdd(error,fdividef(powf(predictions[row*size_x+col]-target[row*size_x+col],2)))
	}

}

//
//void layer::backward(Matrix* delta,Matrix* input)
//{
//	/* This is backward propogation part of the layer. 
//	   Here we find the derivative of the cost function 
//	   with respect to the weights associated with this layer. 
//	   This will have contribution due to the derivatives 
//	   of all the previous layers. This is given to the function 
//	   as delta. The only thing this function is supposed to do 
//	   is multiply the transpose of weights of this layer with delta. 
//	   And this is the new delta which will be passed on to the  
//	   previous layer after multiplying the elements with 
//	   the derivative of the activation function etc. To find the derivative of cost function 
//	   with respect to the weights of this layer, we only 
//	   need the delta and the input value from the previous layer.  
//	 */
//	
//
//
//}
