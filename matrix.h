using namespace std;

class Matrix
{

	//float* M = nullptr;//new float[dim_x*dim_y];

	public : 
	int dim_x=0;	
	int dim_y=0;	
	float* M = nullptr;//new float[dim_x*dim_y];

	Matrix(int dim_x,int dim_y);
};
