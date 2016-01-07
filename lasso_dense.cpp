#include<cstdio>
#include<algorithm>
#include<vector>
#include<iostream>
#include<string>
#include<sstream>
#include<cmath>

#define TOLERANCE 1e-3
#define ITER 100
#define NN 100000
#define TR_SIZE 4000
#define VAL_SIZE 1000
#define TE_SIZE 1000

using namespace std;

vector<float> split(const string &s, char delim) {
    vector<float> elems;
	stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(stof(item));
    }
    return elems;
}

vector< vector<float> > readInput(char *fname)
{
	vector< vector<float> > result;

	char str[NN];
	FILE *fp = fopen(fname,"r");
	if(!fp) return result;

	while(fscanf(fp," %[^\n]",str)!=EOF)
		result.push_back(split(string(str),','));
	fclose(fp);
	return result;
}

vector<float> readLabel(char *fname)
{
	float label = 0;
	vector<float> result;
	FILE *fp = fopen(fname,"r");
	if(!fp) return result;
	
	while(fscanf(fp,"%f",&label)!=EOF) result.push_back(label);
	fclose(fp);
	return result;
}

pair< vector< vector<float> >, vector<float> > splitData(vector< vector<float> > &input,vector<float> &labels,int start,int end)
{
	vector< float > label_result;
	vector< vector<float> > result;
	for(int i=start;i<end;i++) 
	{
		result.push_back(input[i]);
		label_result.push_back(labels[i]);
	}
	return make_pair(result,label_result);
}

vector<float> getLambda()
{
	vector<float> lambda;
	lambda.push_back(0.01);
	lambda.push_back(0.05);
	lambda.push_back(0.1);
	lambda.push_back(0.5);
	lambda.push_back(1.0);
	lambda.push_back(10.0);
	
	return lambda;
}

float vectorDotProduct(vector<float> &X,vector<float> &weights)
{
	float sum = 0.0;
	for(int i=0;i<X.size();i++) sum = sum + X[i]*weights[i];
	return sum;
}

vector<float> matrixDotProduct(vector< vector<float> > &X,vector<float> &weights)
{
	vector< float > result(X.size(),0.0);
	for(int i=0;i<X.size();i++) result[i] = vectorDotProduct(X[i],weights);
	return result;
}

float cost(vector< vector<float> > &X,vector< float > &y,vector<float> &weights,float lambda,float offset)
{
	float csum 	= 0.0;
	float tsum	= 0.0;
	for(int i=0;i<X.size();i++)
	{
		tsum	= vectorDotProduct(X[i],weights);
		tsum	= tsum - offset + y[i];
		csum	= csum + tsum*tsum;
	}
	
	tsum		= 0.0;
	for(int i=0;i<weights.size();i++) tsum = tsum + fabs(weights[i]);
	return csum + lambda*tsum;
}

float gradient(vector< vector<float> > &X,vector< float >  &y,vector<float> &weights,vector<float> &dotProduct,int feature,float offset,float lambda)
{
	float tsum	= 0.0;
	float csum	= 0.0;
	float res	= 0.0;

	for(int i=0;i<X.size();i++)
	{	
		tsum 	= dotProduct[i] - X[i][feature]*weights[feature];
		tsum	= y[i] - tsum - offset;
		res		= res + tsum*X[i][feature];
		csum	= csum+ (X[i][feature]*X[i][feature]);	
	}
	
	csum		= 2*csum;
	res			= 2*res;
	
	if(res > lambda) return (res-lambda)/csum;
	else if(res < -1.0*lambda) return (res+lambda)/csum;
	else return 0;
}

float getoffset(vector< vector<float> > &X,vector< float > &y,vector<float> &weights)
{
	float sum	= 0.0;
	for(int i=0;i<X.size();i++)
		sum = sum + y[i] - vectorDotProduct(X[i],weights);
	
	return sum/(int)X.size();
}

vector<float> runCoordinateDescent(vector< vector<float> > &X,vector< float > &y,float lambda)
{
	float c1		= 0.0;
	float c2		= 0.0;
	float offset	= 0.0;
	float nweight	= 0.0;		

	vector<float> weights(X[0].size(),0.0);
	for(int i=0;i<ITER;i++)
	{
		offset						= getoffset(X,y,weights);
		c1							= cost(X,y,weights,lambda,offset);
		vector<float> dotProduct 	= matrixDotProduct(X,weights);
		for(int j=0;j<X[0].size();j++)
		{
			nweight	= gradient(X,y,weights,dotProduct,j,offset,lambda);
			for(int k=0;k<X.size();k++) dotProduct[k] = dotProduct[k] + X[k][j] * (nweight - weights[j]);
			weights[j]		= nweight;
		}
		c2	= cost(X,y,weights,lambda,offset);
		if(c1-c2 < TOLERANCE) break;
	}
	return weights;
}

pair<float,float> normalize(vector<float> &labels)
{
	float mu 	= 0.0;
	float sigma = 0.0;
	float csum	= 0.0;
		
	for(int i=0;i<labels.size();i++) csum = csum + labels[i];
	mu	 = csum/(int)labels.size();
	
	csum		= 0.0;
	for(int i=0;i<labels.size();i++) csum = csum + (labels[i]-mu)*(labels[i]-mu);
	sigma		= sqrt(csum/(int)labels.size());
	
	for(int i=0;i<labels.size();i++) labels[i] = (labels[i]-mu)/sigma;
	return make_pair(mu,sigma);
}

float predictSingle(vector<float> &X,vector<float> &weights,pair<float,float> &norm_constants)
{
	float sum = 0.0;
	for(int i=0;i<X.size();i++) sum = sum + X[i]*weights[i];
	return sum*norm_constants.second + norm_constants.first;
}

vector<float> predict(vector< vector<float> > &X,vector<float> &weights,pair<float,float> &norm_constants)
{
	vector<float> result;
	for(int i=0;i<X.size();i++) 
		result.push_back(predictSingle(X[i],weights,norm_constants));
	return result;
}

float computeError(vector<float> actual,vector<float> predicted)
{
	float csum	= 0.0;
	for(int i=0;i<actual.size();i++) csum = csum + (actual[i]-predicted[i])*(actual[i]-predicted[i]);
	return sqrt(csum/(int)actual.size());
}

void writeOutput(char *fname,vector<float> &predictions)
{
	FILE *fp = fopen(fname,"w");
	if(!fp) return;
	
	for(int i=0;i<predictions.size();i++) fprintf(fp,"%.2f\n",predictions[i]);
	fclose(fp);	
}

int main(int argc,char **argv)
{
	vector< vector<float> > input 								= readInput(argv[1]);
	vector< float > labels										= readLabel(argv[2]);

	pair< vector< vector<float> >, vector<float> > train	  	= splitData(input,labels,0,TR_SIZE);
	pair< vector< vector<float> >, vector<float> > validation	= splitData(input,labels,TR_SIZE,TR_SIZE+VAL_SIZE);
	pair< vector< vector<float> >, vector<float> > test			= splitData(input,labels,TR_SIZE+VAL_SIZE,TR_SIZE+VAL_SIZE+TE_SIZE);
	
	input.clear();	
	labels.clear();

	pair<float,float> norm_constants 							= normalize(train.second);
	vector<float> lambda										= getLambda();
	float best_error											= 1e6;
	float best_lambda											= 0;
	
	vector<float> best_weights;

	for(int i=0;i<lambda.size();i++)
	{
		vector<float> weights			= runCoordinateDescent(train.first,train.second,lambda[i]);
		vector<float> predictions		= predict(validation.first,weights,norm_constants);
		float rmse						= computeError(validation.second,predictions);
		if(rmse < best_error)
		{
			best_error 					= rmse;
			best_lambda					= lambda[i];
			best_weights				= weights;
		}
	}
	
	printf("Best lambda %.4f\n",best_lambda);
	vector<float> predictions			= predict(test.first,best_weights,norm_constants);
	writeOutput(argv[3],predictions);
	return 0;
}
