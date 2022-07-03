#include<iostream>
#include "matrix_util.h"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>


using namespace std;

int main()
{
	int sizeX = 30;
	int sizeY = 20;
	Matrix A(sizeX,sizeY);
	sizeX = 20;
	sizeY = 30;
	Matrix B(sizeX,sizeY);
	sizeX = 20;
	sizeY = 20;
	Matrix C(sizeX,sizeY);
	Matrix C_cpu(sizeX,sizeY);
	srand(time(0));
	for(int i=0;i<A.dim_x;i++)  
	{
		for(int j=0;j<A.dim_y;j++)
		{
			A.M[j*A.dim_x+i] = rand()%10;
			//B.M[j*sizeX+i] = rand()%10;
		}
	}

  	for(int j=0;j<B.dim_y;j++)
  	{
  		for(int i=0;i<B.dim_x;i++)
  		{
  			//cout<<B.M[j*sizeX+i]<<"\t";
			B.M[j*B.dim_x+i] = rand()%10;
  		}
  	}

//	for(int j=0;j<A.dim_y;j++) // this selects the row
//	{
//		for(int i=0;i<A.dim_x;i++) // this selects the column
//		{
//			cout<<A.M[j*A.dim_x+i]<<"\t";
//		}
//		cout<<"\n";
//	}
//	cout<<"\n";
//	for(int i=0;i<A.dim_x*A.dim_y;i++)
//	{
//		cout<<A.M[i]<<"\t";
//	}
//	cout<<"\n";
//	cout<<"\n";
//	for(int j=0;j<B.dim_y;j++)
//	{
//		for(int i=0;i<B.dim_x;i++)
//		{
//			cout<<B.M[j*B.dim_x+i]<<"\t";
//		}
//		cout<<"\n";
//	}
//	cout<<"\n";
//	for(int i=0;i<B.dim_x*B.dim_y;i++)
//	{
//		cout<<B.M[i]<<"\t";
//	}
//	cout<<"\n";
//	cout<<"\n";
////////int block_size = 32;
////////int n_blocks_x=(C.dim_x+block_size-1)/block_size; 
////////int n_blocks_y=(C.dim_y+block_size-1)/block_size; 
//////////cout<<n_blocks_x<<"\t"<<n_blocks_y<<"\n";

////////dim3 n_blocks(n_blocks_x,n_blocks_y);
////////dim3 n_threads(block_size,block_size);
//////////cout<<"time\t"<<time(0)<<"\n";

////////matrix_multiply <<< n_blocks,n_threads >>> (A.M,B.M,C.M,A.dim_x,A.dim_y,B.dim_x,B.dim_y,C.dim_x,C.dim_y);
////////cudaDeviceSynchronize();
	matrix_multiply_gpu(&A,&B,&C);

////////cout<<"\n";
////////for(int j=0;j<C.dim_y;j++)
////////{
////////	for(int i=0;i<C.dim_x;i++)
////////	{
////////		cout<<C.M[j*C.dim_x+i]<<"\t";
////////	}
////////	cout<<"\n";
////////}
//
////////cout<<"\n";
////////for(int j=0;j<C_cpu.dim_y;j++)
////////{
////////	for(int i=0;i<C_cpu.dim_x;i++)
////////	{
////////		cout<<C_cpu.M[j*C_cpu.dim_x+i]<<"\t";
////////	}
////////	cout<<"\n";
////////}
//
////////for(int i=0;i<C.dim_x;i++)
////////	for(int j=0;j<C.dim_y;j++)
////////	{
////////		if(C.M[j*C.dim_x+i]-C_cpu.M[j*C.dim_x+i])
////////		{
////////			cout<<"error\t"<<C.M[j*C.dim_x+i]-C_cpu.M[j*C.dim_x+i]<<"\n";
//
////////		}
////////		//cout<<C.M[j*sizeX+i]-C_cpu.M[j*sizeX+i]<<"\n";
////////	}
}
