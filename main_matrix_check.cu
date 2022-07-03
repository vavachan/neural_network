#include<iostream>
#include "matrix_util.h"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
//#include "matrix.h"


using namespace std;

int main()
{
	int sizeX = 2;
	int sizeY = 2;
	Matrix A(sizeX,sizeY);
	Matrix B(sizeX,sizeY);
	Matrix C(sizeX,sizeY);
	Matrix C_cpu(sizeX,sizeY);
	srand(time(0));
	for(int i=0;i<sizeX;i++)
	{
		for(int j=0;j<sizeY;j++)
		{
			A.M[j*sizeX+i] = rand();
			B.M[j*sizeX+i] = rand();
		}
	}
	for(int i=0;i<sizeX;i++)
	{
		for(int j=0;j<sizeY;j++)
		{
			cout<<A.M[j*sizeX+i]<<"\t";
		}
		cout<<"\n";
	}
	cout<<"\n";
	for(int i=0;i<sizeX;i++)
	{
		for(int j=0;j<sizeY;j++)
		{
			cout<<B.M[j*sizeX+i]<<"\t";
		}
		cout<<"\n";
	}
	int block_size = 32;
	int n_blocks_x=(C.dim_x+block_size-1)/block_size; 
	int n_blocks_y=(C.dim_y+block_size-1)/block_size; 
	//cout<<n_blocks_x<<"\t"<<n_blocks_y<<"\n";

	dim3 n_blocks(n_blocks_x,n_blocks_y);
	dim3 n_threads(block_size,block_size);

	matrix_multiply <<< n_blocks,n_threads >>> (A.M,B.M,C.M,A.dim_y,A.dim_y,B.dim_y,B.dim_y,C.dim_y,C.dim_y);
	matrix_multiply_cpu(&A,&B,&C_cpu);



	cout<<"\n";
	for(int i=0;i<sizeX;i++)
	{
		for(int j=0;j<sizeY;j++)
		{
			cout<<C.M[j*sizeX+i]<<"\t";
		}
		cout<<"\n";
	}
	for(int i=0;i<sizeX;i++)
	{
		for(int j=0;j<sizeY;j++)
		{
			cout<<i<<"\t"<<j<<"\t"<<C.M[j*sizeX+i]-C_cpu.M[j*sizeX+i]<<"\n";
		}
	}
}

