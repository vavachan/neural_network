
#include<iostream>
#include "matrix_util.h"
#include "neural_network.h"
#include "sequential.h"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;
int reverseInt (int i) 
{
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float function(float x)
{
	// let us try sinx 
	// x is considered to be an angle in degrees 
	//x = x%360; // angle has to be between 0 and 360
	float x_radians = x*M_PI/180;
	return cos(x_radians);
}

float function_2d(float x,float y)
{
	return (x*x + y - 11)*(x*x + y - 11) + (x + y*y -7)*(x + y*y -7);

}

void make_batch(Matrix *X, Matrix *Y)
{
	for(int i=0;i<X->dim_x;i++)  
	{
		float x=5.*(2.*((double) rand() / (RAND_MAX))-1.);
		float y=5.*(2.*((double) rand() / (RAND_MAX))-1.);
		//cout<<x<<"\t"<<y<<"\n";

		X->M[0*X->dim_x+i] = x;
		X->M[1*X->dim_x+i] = y;

		Y->M[0*Y->dim_x+i]=function_2d(x,y);
	}
}

int main()
{
	/*This main will be used to test the neural network on a simple function 
	  We can see if the neural network can store a simple function. 
	  it could be an image or a 1d function like sinx. 
	  doing so will test backprop etc. 

	  update 19/11/2022 

	  sin(X) function can be learned:  update 30/11/22 so this was stupid.
	  now trying a 2D function so that further test of code is done. 
	*/


	// We need an x and y representing the function


	/* We have to create a 2D function. 
	   for this we need a set of (x,y) and f(x,y) 
	   the array of (x,y) will be the two dimensional 
	   input and f(x,y) will be ??!! */
	cudaSetDevice(1);            // Set device 0 as current

	std::ifstream file("/home/varghese/ACADS/CUDA/cuda_check/MINST/train-images-idx3-ubyte",std::ios::binary);
	int a;
	if(file.is_open())
	{
		int magic_number=0;
		int number_of_images=0;
		int n_rows=0;
		int n_cols=0;
		file.read((char*)&magic_number,sizeof(magic_number)); 
		std::cout<<magic_number<<"\n";//<<number_of_images<<"\n";
		magic_number= reverseInt(magic_number);
		file.read((char*)&number_of_images,sizeof(number_of_images));
		std::cout<<number_of_images<<"\n";
		number_of_images= reverseInt(number_of_images);
		file.read((char*)&n_rows,sizeof(n_rows));
		n_rows= reverseInt(n_rows);
		file.read((char*)&n_cols,sizeof(n_cols));
		n_cols= reverseInt(n_cols);
		std::cout<<magic_number<<"\n";//<<number_of_images<<"\n";
		for(int i=0;i<5;++i)
		{
			for(int r=0;r<n_rows;++r)
			{
				std::cout<<"\n";
				for(int c=0;c<n_cols;++c)
				{
					unsigned char temp=0;
					file.read((char*)&temp,sizeof(temp));
					if(temp&255)
						std::cout<<1;
					else
						std::cout<<0;

				}
			}
		}
	}
	int sample_size = 2000;

	Matrix X(sample_size,2);
	Matrix Y(sample_size,1);


	Matrix* Y_NN = nullptr; //(size_x,size_y);
//
  	sigmoid_layer SIGMOID(X.dim_y,150,sample_size);
  	layer LINLAYER(150,Y.dim_y,sample_size);

	vector<layer*> nn{&SIGMOID,&LINLAYER};

	sequential_NN neural_network(nn,sample_size);
//
  	float *cost;	
  	cudaMallocManaged(&cost,sizeof(float));
  	Matrix cost_gradient(sample_size,Y.dim_y);
//
//
  	for(int i=0; i<10000; i++)
  	{
  		*cost=0.;
		make_batch(&X,&Y);
  		// forward propogation
  		neural_network.forward(&X,&Y_NN);
		//SIGMOID.Weight->print_matrix();
		//Y_NN->print_matrix();
  		// calculate cost
  		mean_squared_error_2d_gpu(Y_NN,&Y,&cost_gradient,cost);
		//cost_gradient.print_matrix();
  		cout<<i<<"\t"<<*cost/sample_size<<"\n";
  		// Now we have to do backpropogation 
  		// We have the gradient of the cost function right now 
  		neural_network.backward(&cost_gradient);
		//LINLAYER.dWeight->print_matrix();
  		// update the weights
  		neural_network.update();
		delete Y_NN;
  	}

//	for(int i=0; i<sample_size;i++)
//	{
//		cout<<X.M[0*X.dim_x+i]<<"\t"<<X.M[X.dim_x+i]<<"\t"<<Y.M[i]<<"\t"<<Y_NN->M[i]<<"\n";
//	}


	cudaFree(cost);
	cudaDeviceReset();

	return 0;
}
