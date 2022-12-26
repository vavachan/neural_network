
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
#include <algorithm>    // std::random_shuffle

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

int magic_number=0;
int number_of_images=0;
int number_of_test_images=0;
int n_rows=0;
int n_cols=0;
double **training_data=nullptr;
int * training_labels=nullptr;
int * indices=nullptr;


double **test_data=nullptr;
int *test_labels=nullptr;

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

void make_batch(Matrix *X, Matrix *Y,int begin)
{
	for(int i=0;i<X->dim_x;i++)  
	{
		//int random_image = int(((double) rand() / (RAND_MAX))*30000);
		int random_image=indices[begin+i];
		//cout<<random_image<<"\n";
		int y_index = 0;
		for(int row=0;row<n_rows;row++)
		{
			for(int col=0;col<n_cols;col++)
			{
				X->M[y_index*X->dim_x+i]=training_data[random_image][row*n_cols+col];
				y_index++;
			}
		}
		for(int label=0;label<10;label++)
		{
			if(training_labels[random_image]==label)
			{
				Y->M[label*Y->dim_x+i]=1;
			}
			else
			{
				Y->M[label*Y->dim_x+i]=0;
			}
		}
		//cout<<random_image<<"\n";
		//float x=5.*(2.*((double) rand() / (RAND_MAX))-1.);
		//float y=5.*(2.*((double) rand() / (RAND_MAX))-1.);
		////cout<<x<<"\t"<<y<<"\n";

		//X->M[0*X->dim_x+i] = x;
		//X->M[1*X->dim_x+i] = y;

		//Y->M[0*Y->dim_x+i]=function_2d(x,y);
	}

}

void make_test_batch(Matrix *X, Matrix *Y)
{
	for(int i=0;i<X->dim_x;i++)  
	{
		int random_image = int(((double) rand() / (RAND_MAX))*10000);
		//int random_image=indices[begin+i];
		//cout<<random_image<<"\n";
		int y_index = 0;
		for(int row=0;row<n_rows;row++)
		{
			for(int col=0;col<n_cols;col++)
			{
				X->M[y_index*X->dim_x+i]=test_data[random_image][row*n_cols+col];
				y_index++;
			}
		}
		for(int label=0;label<10;label++)
		{
			if(test_labels[random_image]==label)
			{
				Y->M[label*Y->dim_x+i]=1;
			}
			else
			{
				Y->M[label*Y->dim_x+i]=0;
			}
		}
		//cout<<random_image<<"\n";
		//float x=5.*(2.*((double) rand() / (RAND_MAX))-1.);
		//float y=5.*(2.*((double) rand() / (RAND_MAX))-1.);
		////cout<<x<<"\t"<<y<<"\n";

		//X->M[0*X->dim_x+i] = x;
		//X->M[1*X->dim_x+i] = y;

		//Y->M[0*Y->dim_x+i]=function_2d(x,y);
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

	  update 24/12/2022 // going ahead and training 
	  with MINST data set. I think this is good 
	  way to conclude this project. 
	*/


	// We need an x and y representing the function


	/* We have to create a 2D function. 
	   for this we need a set of (x,y) and f(x,y) 
	   the array of (x,y) will be the two dimensional 
	   input and f(x,y) will be ??!! */
	cudaSetDevice(1);            // Set device 0 as current

	std::ifstream file("/home/varghese/ACADS/CUDA/cuda_check/MINST/train-images-idx3-ubyte",std::ios::binary);
	if(file.is_open())
	{
		file.read((char*)&magic_number,sizeof(magic_number)); 
		magic_number= reverseInt(magic_number);
		file.read((char*)&number_of_images,sizeof(number_of_images));
		number_of_images= reverseInt(number_of_images);
		file.read((char*)&n_rows,sizeof(n_rows));
		n_rows= reverseInt(n_rows);
		file.read((char*)&n_cols,sizeof(n_cols));
		n_cols= reverseInt(n_cols);
		training_data = new double*[number_of_images];
		indices = new int[number_of_images];
		for(int image=0; image<number_of_images; image++)
		{
			training_data[image] = new double[n_rows*n_cols];
			indices[image]=image;
		}
		for(int i=0;i<number_of_images;++i)
		{
			int loc=0;
			for(int r=0;r<n_rows;++r)
			{
				for(int c=0;c<n_cols;++c)
				{
					unsigned char temp=0;
					file.read((char*)&temp,sizeof(temp));
					training_data[i][loc]=double((temp&255)/255.);
					loc++;

				}
			}
		}
	}
	// training data has been loaded. 

	std::ifstream lfile("/home/varghese/ACADS/CUDA/cuda_check/MINST/train-labels-idx1-ubyte",std::ios::binary); // training labels 
	training_labels = new int[number_of_images];
	if(lfile.is_open())
	{
		lfile.read((char*)&magic_number,sizeof(magic_number)); 
		magic_number= reverseInt(magic_number);
		lfile.read((char*)&number_of_images,sizeof(number_of_images));
		number_of_images= reverseInt(number_of_images);
		for(int i=0;i<number_of_images;++i)
		{
			unsigned char temp=0;
			lfile.read((char*)&temp,sizeof(temp));
			training_labels[i]=temp&255;
			//std::cout<<(temp&255)<<"\n";
			
		}
	}
/*###########################################################################################################################*/

	std::ifstream tfile("/home/varghese/ACADS/CUDA/cuda_check/MINST/t10k-images-idx3-ubyte",std::ios::binary);
	if(tfile.is_open())
	{
		tfile.read((char*)&magic_number,sizeof(magic_number)); 
		magic_number= reverseInt(magic_number);
		tfile.read((char*)&number_of_test_images,sizeof(number_of_test_images));
		number_of_test_images= reverseInt(number_of_test_images);
		tfile.read((char*)&n_rows,sizeof(n_rows));
		n_rows= reverseInt(n_rows);
		tfile.read((char*)&n_cols,sizeof(n_cols));
		n_cols= reverseInt(n_cols);
		test_data = new double*[number_of_test_images];
		for(int image=0; image<number_of_test_images; image++)
		{
			test_data[image] = new double[n_rows*n_cols];
		}
		for(int i=0;i<number_of_test_images;++i)
		{
			int loc=0;
			for(int r=0;r<n_rows;++r)
			{
				for(int c=0;c<n_cols;++c)
				{
					unsigned char temp=0;
					tfile.read((char*)&temp,sizeof(temp));
					test_data[i][loc]=double((temp&255)/255.);
					loc++;

				}
			}
		}
	}
	tfile.close();
	//// test data has been loaded. 

	std::ifstream ltfile("/home/varghese/ACADS/CUDA/cuda_check/MINST/t10k-labels-idx1-ubyte",std::ios::binary); // training labels 
	test_labels = new int[number_of_test_images];
	if(ltfile.is_open())
	{
		ltfile.read((char*)&magic_number,sizeof(magic_number)); 
		magic_number= reverseInt(magic_number);
		ltfile.read((char*)&number_of_test_images,sizeof(number_of_test_images));
		number_of_test_images= reverseInt(number_of_test_images);
		for(int i=0;i<number_of_test_images;++i)
		{
			unsigned char temp=0;
			ltfile.read((char*)&temp,sizeof(temp));
			test_labels[i]=temp&255;
			//std::cout<<(temp&255)<<"\n";
			
		}
	}
	ltfile.close();
	// training labels has been loaded as well.
	//int image_number=3245;
	//cout<<"\n"<<training_labels[image_number]<<"\n";
	//return 0;
	int sample_size = 10;

	Matrix X(sample_size,n_cols*n_rows);
	Matrix Y(sample_size,10);

	//make_batch(&X,&Y);
	Matrix* Y_NN = nullptr; //(size_x,size_y);
//
  	sigmoid_layer SIGMOID1(X.dim_y,30,sample_size);
  	sigmoid_layer SIGMOID2(30,Y.dim_y,sample_size);

	vector<layer*> nn{&SIGMOID1,&SIGMOID2};

	sequential_NN neural_network(nn,sample_size);
//
  	float *cost;	
  	cudaMallocManaged(&cost,sizeof(float));
  	Matrix cost_gradient(sample_size,Y.dim_y);
//
//
	//Y.print_matrix();
	//return 0;
	int number_of_epochs=30;
  	for(int epoch=0; epoch<number_of_epochs; epoch++)
  	{
		random_shuffle(&indices[0],&indices[number_of_images-1]);
		for(int batches=0;batches < number_of_images/sample_size; batches++)
		{
			*cost=0.;
			make_batch(&X,&Y,batches*sample_size);
			// forward propogation
			neural_network.forward(&X,&Y_NN);
			//SIGMOID.Weight->print_matrix();
			//Y_NN->print_matrix();
			//cout<<"cost\n";
			// calculate cost
			mean_squared_error_2d_gpu(Y_NN,&Y,&cost_gradient,cost);
			//cost_gradient.print_matrix();
			// Now we have to do backpropogation 
			// We have the gradient of the cost function right now 
			neural_network.backward(&cost_gradient);
			//LINLAYER.dWeight->print_matrix();
			// update the weights
			neural_network.update();
			delete Y_NN;
		}
		
		cout<<epoch<<"\t"<<*cost/sample_size<<"\n";
  	}
	for(int i=0;i<2;i++)
	{
		make_test_batch(&X,&Y);
		neural_network.forward(&X,&Y_NN);
		for(int i=0; i < sample_size;i++)
		{
			int y_index=0;
			for(int row=0;row<n_rows;row++)
			{
				for(int col=0;col<n_cols;col++)
				{
					if(X.M[y_index*X.dim_x+i])
					{
						cout<<1;
					}
					else
					{
						cout<<0;
					}
					y_index++;
				}
				cout<<"\n";
			}
			for(int label=0;label<10;label++)
			{
				printf("%d\t%.3f\t%d\n",label,Y_NN->M[label*Y_NN->dim_x+i],int(Y.M[label*Y.dim_x+i]));
				//cout<<label<<"\t"<<Y_NN->M[label*Y_NN->dim_x+i]<<"\t"<<Y.M[label*Y.dim_x+i]<<"\n";
			}
		}
		//Y_NN->print_matrix();
		//cout<<"correct\n";
		//Y.print_matrix();
	}
//	for(int i=0; i<sample_size;i++)
//	{
//		cout<<X.M[0*X.dim_x+i]<<"\t"<<X.M[X.dim_x+i]<<"\t"<<Y.M[i]<<"\t"<<Y_NN->M[i]<<"\n";
//	}


	cudaFree(cost);
	cudaDeviceReset();

	return 0;
}
