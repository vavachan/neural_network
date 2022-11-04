#include "matrix_util.h"
#include "neural_network.h"
#include "sequential.h"

sequential_NN::sequential_NN(std::vector<layer*> network)
{
	layers = network;
}

void sequential_NN::forward(Matrix *input, Matrix *output)
{
	sequential_forward_gpu(input,layers,output);
}

void sequential_forward_gpu(Matrix *input, std::vector< layer*> layers, Matrix *output)
{
	int layer_output_size;
	for(int i =0;i<layers.size();i++)
	{
		layer *curr_layer = layers[i];
		layer_output_size = curr_layer->out_size;
		// treating all input and output as 1d matrices, will have to worry about 2D later? 
		Matrix curr_out(1.,layer_output_size);
		curr_layer->forward(input,&curr_out);
		input = &curr_out;
	}
}
