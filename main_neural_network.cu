
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
	return sin(x_radians);
}

int main()
{
	/*This main will be used to test the neural network on a simple function 
	  We can see if the neural network can store a simple function. 
	  it could be an image or a 1d function like sinx. 
	  doing so will test backprop etc. 
	*/

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
	}
	// now I have X matrix and Y matrix 
	// now I have to initialize a neural network

	//layer LINLAYER_1(X.dim_y,Y.dim_y);
	//sigmoid_layer SIGMOID_1(X.dim_y,Y.dim_y);
	//layer LINLAYER_2(X.dim_y,Y.dim_y);

	

	layer LINLAYER_1(X.dim_y,1000);
	sigmoid_layer SIGMOID_1(X.dim_y,1000);
	layer LINLAYER_2(1000,Y.dim_y);


	vector<layer*> nn{&SIGMOID_1,&LINLAYER_2};
	//vector<layer*> nn{&LINLAYER_1};

	sequential_NN neural_network(nn);

	float *cost;	
	cudaMallocManaged(&cost,sizeof(float));
	Matrix cost_gradient(size_x,size_y);

	//cout<<Y_NN<<"\n";	
	//Y_NN->print_matrix();

	// forward propogation
	neural_network.forward(&X,&Y_NN);

	mean_squared_error_2d_gpu(Y_NN,&Y,&cost_gradient,cost);

	cout<<*cost<<"\n";
	// Now we have to do backpropogation 
	// We have the gradient of the cost function right now 
	Matrix* tmp_delta = nullptr;
	tmp_delta = new Matrix(1,1000);
	LINLAYER_2.backward(&cost_gradient,tmp_delta);
		
	return 0;
}
