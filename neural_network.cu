#include "neural_network.h"
#include <cmath>
#include <iostream>


layer::layer(int input_S, int output_S)
{
	inp_size = input_S;
	out_size = output_S;
	activations = new Matrix(1,inp_size);
	activationsT = new Matrix(inp_size,1);
	Weight  = new Matrix(inp_size,out_size);
	WeightT  = new Matrix(out_size,inp_size);
	dWeight = new Matrix(inp_size,out_size);
	Bias    = new Matrix(1,out_size);
	dBias   = new Matrix(1,out_size);


	/* for the time being we will initialize all the 
	   weigths and biases to zero, see what happens. */

	for(int i=0;i<Weight->dim_x;i++)  
	{
		for(int j=0;j<Weight->dim_y;j++)
		{
			Weight->M[j*Weight->dim_x+i] = 0.;//rand()%10;
			//B->M[j*sizeX+i] = rand()%10;
		}
	}
	for(int i=0;i<Bias->dim_x;i++)  
	{
		for(int j=0;j<Bias->dim_y;j++)
		{
			Bias->M[j*Bias->dim_x+i] = 0.;//rand()%10;
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
	*activations = *input;	
	matrix_multiply_add_gpu(Weight,input,Bias,output);
}

void layer::backward(Matrix* delta_n,Matrix* delta_n_minus_one)
{
	/* This is backward propogation part of the layer. 
	   Here we find the derivative of the cost function 
	   with respect to the weights associated with this layer. 
	   This will have contribution due to the derivatives 
	   of all the previous layers. This is given to the function 
	   as delta. The only thing this function is supposed to do 
	   is multiply the transpose of weights of this layer with delta. 
	   And this is the new delta which will be passed on to the  
	   previous layer after multiplying the elements with 
	   the derivative of the activation function etc. To find the derivative of cost function 
	   with respect to the weights of this layer, we only 
	   need the delta and the input value from the previous layer.  
	 */

	// First we need to take the transpose of the weight matrix
	// We need to use this matrix to generate \delta_l

	matrix_transpose_gpu(Weight,WeightT);
	matrix_transpose_gpu(activations,activationsT);
	matrix_multiply_gpu(delta_n,activationsT,dWeight);
	*dBias = *delta_n;
	matrix_multiply_gpu(WeightT,delta_n,delta_n_minus_one);
}
void layer::update(float learning_rate)
{
	matrix_scalar_product_gpu(dWeight,learning_rate);	
	matrix_scalar_product_gpu(dBias,learning_rate);	
	
	//matrix_add_gpu(Weight,dWeight,Weight);
	matrix_add_gpu(Bias,dBias,Bias);

}
sigmoid_layer::sigmoid_layer(int input_S, int output_S) : layer(input_S, output_S)
{
	/* currently lets keep this as the activations from corresponding 
	   linear layer ( z_n = a_n-1 * W_nn-1 + b_n ) */

	/* While sigmoid layer can accept inp_size and out_size 
	   of different size, then the linear layer 
	   within the sigmoid function has to take care of the 
	   size change using the weight matrix. */
	sigmoid_activations = new Matrix(1,output_S);  // this activations should be of the output size 
}


void sigmoid_layer::forward(Matrix* input, Matrix* output)
{
	Matrix* tmp = nullptr;
	tmp = new Matrix(output->dim_x,output->dim_y);

	layer::forward(input,tmp);

	*sigmoid_activations = *tmp; // we will use this for calculating the derivatives. 
	//std::cout<<sigmoid_activations<<"\t"<<sigmoid_activations->dim_y<<"\n";
	sigmoid_layer_forward_gpu(tmp,output);

	delete tmp;
	//std::cout<<sigmoid_activations<<"\t"<<sigmoid_activations->dim_y<<"\n";
}

void sigmoid_layer::backward(Matrix* delta_n,Matrix* delta_n_minus_one)
{
	/* for the backward prop of the sigmoid layer, 
	   first the derivative of the sigmoid function is multiplied 
	   with the ``delta'' that this layer recieved from 
	   the next layer ( n -> n-1) [remember this is going backward]
	   */

	// the derivative of the sigmoid function measured at the activation values 
	// that were recieved by this layer during forward pass is multiplied 
	// with the gradients that will  
	Matrix* dsigma = nullptr;
	Matrix* dsigma_dot_delta = nullptr;
	dsigma = new Matrix(delta_n->dim_x,delta_n->dim_y);
	dsigma_dot_delta = new Matrix(delta_n->dim_x,delta_n->dim_y);
	
	//std::cout<<sigmoid_activations<<"\t"<<sigmoid_activations->dim_y<<" this is in the backward prop\n";
//	std::cout<<sigmoid_activations->dim_y<<"\t"<<dsigma->dim_y<<"\n";

	sigmoid_layer_backward_gpu(sigmoid_activations,dsigma);
	matrix_hadamard_product_gpu(dsigma,delta_n,dsigma_dot_delta);

	layer::backward(dsigma_dot_delta,delta_n_minus_one);

	delete dsigma;
	delete dsigma_dot_delta;

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

void sigmoid_layer_backward_gpu(Matrix* input, Matrix* output)
{
	int block_size = 32;
	if(!((input->dim_x == output->dim_x) and (input->dim_y == output->dim_y)))
	{
		std::cout<<"error dimension miss match in sigmoid backward\n"<<"\n";
		std::cout<<input->dim_x<<"\t"<<input->dim_y<<"\n";
		std::cout<<output->dim_x<<"\t"<<output->dim_y<<"\n";
	}

	int n_blocks_x=(input->dim_x+block_size-1)/block_size; 
	int n_blocks_y=(input->dim_y+block_size-1)/block_size; 
	//cout<<n_blocks_x<<"\t"<<n_blocks_y<<"\n";

	dim3 n_blocks(n_blocks_x,n_blocks_y);
	dim3 n_threads(block_size,block_size);
	//cout<<"time\t"<<time(0)<<"\n";

	dsigmoid_function <<< n_blocks,n_threads >>> (input->M,output->M,input->dim_x,input->dim_y,output->dim_x,output->dim_y);
	cudaDeviceSynchronize();
}
__global__
void sigmoid_function(float *input, float *output, int input_dim_x, int input_dim_y, int output_dim_x, int output_dim_y)
{
	int col = blockDim.x*blockIdx.x+threadIdx.x;
	int row = blockDim.y*blockIdx.y+threadIdx.y;
	// We will compute stupidly. 
	if(col < output_dim_x and row < output_dim_y)
	{
		output[row*output_dim_x+col] = exp(-1.*input[row*input_dim_x+col])/((1+exp(-1.*input[row*input_dim_x+col])*(1+exp(-1.*input[row*input_dim_x+col]))));
	}
}

__global__
void dsigmoid_function(float *input, float *output, int input_dim_x, int input_dim_y, int output_dim_x, int output_dim_y)
{
	int col = blockDim.x*blockIdx.x+threadIdx.x;
	int row = blockDim.y*blockIdx.y+threadIdx.y;
	if(col < output_dim_x and row < output_dim_y)
	{
		output[row*output_dim_x+col] = 1./(1+exp(-1.*input[row*input_dim_x+col]));
	}

}
void mean_squared_error_2d_gpu(Matrix* predictions, Matrix* target, Matrix *gradient, float *error)
{
	int block_size = 32;

	if(!((predictions->dim_x == target->dim_x) and (predictions->dim_y == target->dim_y)))
	{
		std::cout<<"error dimension miss match \n"<<"\n";
	}

	int n_blocks_x=(predictions->dim_x+block_size-1)/block_size; 
	int n_blocks_y=(predictions->dim_y+block_size-1)/block_size; 
	//cout<<n_blocks_x<<"\t"<<n_blocks_y<<"\n";

	dim3 n_blocks(n_blocks_x,n_blocks_y);
	dim3 n_threads(block_size,block_size);
	//cout<<"time\t"<<time(0)<<"\n";

	//std::cout<<*error<<"\t"<<predictions->dim_x<<"\t"<<predictions->dim_y<<"\n";
	mean_squared_error_2d <<< n_blocks,n_threads >>> (predictions->M,target->M,gradient->M,predictions->dim_x, predictions->dim_y, error);
	//float cost =0. ;
	//for(int i=0; i < predictions->dim_y; i++)
	//{
	//	cost = cost+(predictions->M[i]-target->M[i])*(predictions->M[i]-target->M[i]);
	//}
	//cost = cost/predictions->dim_y;
	////std::cout<<cost<<"\n";
	cudaDeviceSynchronize();
}
__global__ 
void mean_squared_error_2d(float *predictions, float *target, float *gradient, int size_x, int size_y, float *error)
{
	int col = blockDim.x*blockIdx.x+threadIdx.x;
	int row = blockDim.y*blockIdx.y+threadIdx.y;
	//atomicAdd(error,1);
	if(col < size_x and row < size_y)
	{
		//output[row*output_dim_x+col] = 1/(1+exp(input[row*input_dim_x+col]));
		atomicAdd(error,fdividef(powf(predictions[row*size_x+col]-target[row*size_x+col],2),size_x*size_y));
		gradient[row*size_x+col] = fdividef(2.*(predictions[row*size_x+col]-target[row*size_x+col]),size_x*size_y) ;
	}
}
