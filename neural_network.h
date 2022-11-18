
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "matrix_util.h"
class layer
{
	/* the only parameters defined here is 
	 * how the layer should take an array of 
	 * inp_size and spit out another array of 
	 * out_size. The dimensions matching with 
	 * the subsequent layer is taken care elsewhere.
	 *
	 * We also store the input recieved by the 
	 * layer here. This is essentially the "activations" 
	 * recieved by this layer ( output of the previous layer).
	 *
	 * */
	public:
		int inp_size;
		int out_size;
		Matrix* activations;
		Matrix* activationsT;
		Matrix* Weight;
		Matrix* WeightT;
	        Matrix* dWeight;
	        Matrix* Bias;
	        Matrix* dBias;
	        virtual void forward(Matrix *, Matrix *);
		layer(int,int);
	        virtual void backward(Matrix *,Matrix *);
		void update(Matrix *, float);
};


class sigmoid_layer : public layer
{
	// this is a sigmoid layer, this inherits the 
	// features of the linear layer
	// this will apply a sigmoid function to the output of 
	// linear layer. 
	public :
		Matrix* sigmoid_activations; // This needs to be stored so one can compute the derivative 
					     // easily. 
		sigmoid_layer(int ,int ); // I hope this constructor works.
		void forward(Matrix *, Matrix *);
		void backward(Matrix *, Matrix *);
};


void sigmoid_layer_forward_gpu(Matrix *input, Matrix *output);
void sigmoid_layer_backward_gpu(Matrix *input, Matrix *output);
__global__ void sigmoid_function(float *, float *, int,int, int,int);
__global__ void dsigmoid_function(float *, float *, int,int, int,int);

void mean_squared_error_2d_gpu(Matrix *prediction, Matrix *target, Matrix *gradient, float *error);
__global__ void mean_squared_error_2d(float *prediction,float *target, float *gradient, int sizex, int sizey, float *error);

#endif // NEURAL_NETWORK_H
