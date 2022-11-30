#include "matrix_util.h"
#include "neural_network.h"
#include "sequential.h"
#include <iostream>

sequential_NN::sequential_NN(std::vector<layer*> network, int bs)
{
	layers = network;
	batch_size = bs ;
}

void sequential_NN::forward(Matrix *input, Matrix **output)
{
	sequential_forward_gpu(input,layers,output,batch_size);
}

void sequential_forward_gpu(Matrix *input, std::vector< layer*> layers, Matrix **output,int batch_size)
{
	int layer_output_size;

	Matrix *_input = nullptr;

	Matrix ** curr_out = new Matrix*[layers.size()];

	_input = input;

	for(int i =0;i<layers.size();i++)
	{
		layer *curr_layer = layers[i];

		layer_output_size = curr_layer->out_size;

		curr_out[i] = new Matrix(batch_size,layer_output_size);

		curr_layer->forward(_input,curr_out[i]);

		_input = curr_out[i];
		if(i)
		{
			delete curr_out[i-1];
		}
	}

	*output = curr_out[layers.size()-1];
}

void sequential_NN::backward(Matrix *delta_n)
{

	/* backward propagation works from the last.
	   delta_n is the gradient of the cost function 
	   as this is the only thing functions outside 
	   this class should know off. */
	Matrix** delta_n_minus_one = new Matrix*[layers.size()];


	int layer_input_size;
	for(int i= layers.size()-1; i>-1;i--)
	{
		layer *curr_layer = layers[i];

		layer_input_size = curr_layer->inp_size;
		//cout<<i<<"\t"<<layer_input_size<<"\n";
		delta_n_minus_one[i] = new Matrix(batch_size,layer_input_size);

		curr_layer->backward(delta_n,delta_n_minus_one[i]);

		delta_n = delta_n_minus_one[i];
		if(i<layers.size()-1)
		{
		    delete delta_n_minus_one[i+1];
		}
	}
}
void sequential_NN::update()
{
		//std::cout<<i<<"\t"<<layer_output_size<<"\t"<<_input->dim_y<<"\n";

	for(int i=0; i<layers.size(); i++)
	{
		layers[i]->update(-1.*0.01);
	}
}
