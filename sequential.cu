#include "matrix_util.h"
#include "neural_network.h"
#include "sequential.h"
#include <iostream>

sequential_NN::sequential_NN(std::vector<layer*> network)
{
	layers = network;
}

void sequential_NN::forward(Matrix *input, Matrix **output)
{
	sequential_forward_gpu(input,layers,output);
}

void sequential_forward_gpu(Matrix *input, std::vector< layer*> layers, Matrix **output)
{
	int layer_output_size;
	//input->print_matrix();
	Matrix *_input = nullptr;
	Matrix * curr_out = nullptr;
	_input = input;
	for(int i =0;i<layers.size();i++)
	{
		layer *curr_layer = layers[i];
		layer_output_size = curr_layer->out_size;
		//std::cout<<i<<"\t"<<layer_output_size<<"\t"<<_input->dim_y<<"\n";
		// treating all input and output as 1d matrices, will have to worry about 2D later?
		curr_out = new Matrix(1.,layer_output_size);
		//curr_out->print_matrix();
		//std::cout<<curr_out<<"\t"<<_input<<"\n";
		curr_layer->forward(_input,curr_out);
		//delete _input;
		_input = curr_out;
	}
	//curr_out->print_matrix();
	//cout<<output<<"\n";
	*output = curr_out;
	//cout<<output<<"\n";
	//delete curr_out;
}

void sequential_NN::backward(Matrix *delta_n)
{

	/* backward propagation works from the last.
	   delta_n is the gradient of the cost function 
	   as this is the only thing functions outside 
	   this class should know off. */
	Matrix* delta_n_minus_one = nullptr;
	int layer_input_size;
	for(int i= layers.size()-1; i>-1;i--)
	{
		layer *curr_layer = layers[i];

		layer_input_size = curr_layer->inp_size;
		//cout<<i<<"\t"<<layer_input_size<<"\n";
		delta_n_minus_one = new Matrix(1.,layer_input_size);

		curr_layer->backward(delta_n,delta_n_minus_one);

		delta_n = delta_n_minus_one;
	}
}
void sequential_NN::update()
{

	for(int i=0; i<layers.size(); i++)
	{
		layers[i]->update(-1.*0.5);
	}
}
