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
