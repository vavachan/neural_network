
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "matrix_util.h"
class layer
{
	public:
		int inp_size;
		int out_size;
		Matrix * Weight;
	        Matrix * dWeight;
	        Matrix * Bias;
	        Matrix * dBias;
	        void forward(Matrix *, Matrix *);
		layer(int,int);
	////////void backward(Matrix *,Matrix *);
};


class sigmoid_layer : public layer
{
	// this is a sigmoid layer, this inherits the 
	// features of the linear layer
	// this will apply a sigmoid function to the output of 
	// linear layer. 
	sigmoid_layer(int ,int ); // I hope this constructor works.
	void forward(Matrix *, Matrix *);
};


void sigmoid_layer_forward_gpu(Matrix *input, Matrix *output);
__global__ void sigmoid_function(float *, float *, int,int, int,int);

void mean_squared_error_2d_gpu(Matrix *prediction, Matrix *target, float *error);
__global__ void mean_squared_error_2d(float *,float *, int, int, float *);

#endif // NEURAL_NETWORK_H
