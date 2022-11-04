#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "neural_network.h"
#include "matrix_util.h"

#include <vector>

using namespace std;

class sequential_NN 
{
	public:
		vector<layer*> layers;
		sequential_NN(vector<layer*>);
		void forward(Matrix *,Matrix *);
		void update();
};
void sequential_forward_gpu(Matrix *input ,std::vector< layer*> layers, Matrix *output);
#endif
