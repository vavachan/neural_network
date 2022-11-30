
#include<iostream>
#include "matrix_util.h"
#include "neural_network.h"
#include "sequential.h"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <cmath>
#include <vector>


using namespace std;

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

int main()
{
	/*This main will be used to test the neural network on a simple function 
	  We can see if the neural network can store a simple function. 
	  it could be an image or a 1d function like sinx. 
	  doing so will test backprop etc. 

	  update 19/11/2022 

	  sin(X) function can be learned: 
	  now trying a 2D function so that further test of code is done. 
	*/


	/* THIS IS FOR 1D FUNCTION : THIS WILL BE REMOVED DURING CLEANUP 
	// We need an x and y representing the function 


	float angle = 0.;
	float angles[100];
	float sin_function[100];
	for(int i=0; i<100; i++)
	{
		//cout<<angle<<"\t"<<function(angle)<<"\n";
		angles[i]=angle;
		sin_function[i] = function(angle);
		angle = angle + 360./100;	
	}
	int size_x = 1;
	int size_y = 100;
	Matrix X(size_x,size_y);
	Matrix Y(size_x,size_y);
	Matrix* Y_NN = nullptr; //(size_x,size_y);

	for(int i=0;i<X.dim_x;i++)  
	{
		for(int j=0;j<X.dim_y;j++)
		{
			X.M[j*X.dim_x+i] = angles[j];
			//B.M[j*sizeX+i] = rand()%10;
		}
	}
	for(int i=0;i<Y.dim_x;i++)  
	{
		for(int j=0;j<Y.dim_y;j++)
		{
			Y.M[j*Y.dim_x+i] = sin_function[j];
			//cout<<i<<"\t"<<j<<"\t"<<Y_NN.M[j*Y.dim_x+i]<<"\n";
			//B.M[j*sizeX+i] = rand()%10;
		}
	} */

	/* We have to create a 2D function. 
	   for this we need a set of (x,y) and f(x,y) 
	   the array of (x,y) will be the two dimensional 
	   input and f(x,y) will be ??!! */
	cudaSetDevice(1);            // Set device 0 as current

	float x_max = 5.0; 
	float x_min = -5.0; 

	float y_max = 5.0; 
	float y_min = -5.0; 

	int sample_size = 100;

	Matrix X(sample_size,2);
	Matrix Y(sample_size,1);

	for(int i=0;i<X.dim_x;i++)  
	{
		float x=5.*(2.*((double) rand() / (RAND_MAX))-1.);
		float y=5.*(2.*((double) rand() / (RAND_MAX))-1.);

		X.M[0*X.dim_x+i] = x;
		X.M[1*X.dim_x+i] = y;

		Y.M[0*Y.dim_x+i]=function_2d(x,y);
	}
	// now I have X matrix and Y matrix 
	// now I have to initialize a neural network
	// Random 1000 samples they are. 

	Matrix* Y_NN = nullptr; //(size_x,size_y);
//
  	sigmoid_layer SIGMOID(X.dim_y,50,sample_size);
  	layer LINLAYER(50,Y.dim_y,sample_size);

	vector<layer*> nn{&SIGMOID,&LINLAYER};

	sequential_NN neural_network(nn,sample_size);
//
  	float *cost;	
  	cudaMallocManaged(&cost,sizeof(float));
  	Matrix cost_gradient(sample_size,Y.dim_y);
//
//
  	for(int i=0; i<500000; i++)
  	{
  		*cost=0.;
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
  	}
	cudaFree(cost);
	cudaDeviceReset();
	return 0;
}
