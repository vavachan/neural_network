
#include<iostream>
#include "matrix_util.h"
#include "neural_network.h"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <cmath>

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
	Matrix Y_NN(size_x,size_y);

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
			//B.M[j*sizeX+i] = rand()%10;
		}
	}
	// now I have X matrix and Y matrix 
	// now I have to initialize a neural network

	// Consider a single linear layer 

	layer LINLAYER_1(X.dim_y,Y.dim_y);


	LINLAYER_1.forward(&X,&Y_NN);

	for(int i=0;i<Y_NN.dim_x;i++)  
	{
		for(int j=0;j<Y_NN.dim_y;j++)
		{
			//Y.M[j*Y.dim_x+i] = sin_function[j];
			cout<<j<<"\t"<<Y_NN.M[j*Y.dim_x+i]<<"\n";
			//B.M[j*sizeX+i] = rand()%10;
		}
	}
}
