#include<cstdio>
#include<algorithm>
#include<vector>
#include<iostream>
#include<string>
#include<sstream>
#include<cmath>

#define TOLERANCE 1e-6
#define ITER 100
#define TR_SIZE 30000
#define VAL_SIZE 10000
#define TE_SIZE 5000

using namespace std;
typedef struct csr
{
	vector<int> document;
	vector<int> feature;
	vector<float> values;
	vector<int> documentid;
	
	int maxDocuments;
	int maxFeatures;
	int nonzeros;

}csr;

typedef struct csc
{
	vector<int> document;
	vector<int> feature;
	vector<float> values;
	vector<int> documentid;

}csc;

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

csr readInput(char *fname)
{
	csr data;
	FILE *fp 			= fopen(fname,"r");
	if(!fp) return data;

	int documentid		= 0;
	int featureid		= 0;
	float values		= 0;
	int pre_document	= -1;

	fscanf(fp,"%d%d%d",&data.maxDocuments,&data.maxFeatures,&data.nonzeros);
	while(fscanf(fp,"%d%d%f",&documentid,&featureid,&values)!=EOF)
	{
		if(pre_document != documentid) 
		{
			data.documentid.push_back(documentid);
			data.document.push_back(data.feature.size());
		}
		data.feature.push_back(featureid);
		data.values.push_back(values);
		pre_document = documentid;
	}
	data.document.push_back(data.feature.size());
	return data;
}

csr split(csr &data,int start,int end)
{
	csr result;
	result.maxFeatures	= data.maxFeatures;
	for(int i=start;i<end;i++)
	{
		result.document.push_back(result.feature.size());
		result.documentid.push_back(data.documentid[i]);
		for(int j=data.document[i];j<data.document[i+1];j++)
		{
			result.feature.push_back(data.feature[j]);
			result.values.push_back(data.values[j]);	
		}
	}
	
	result.maxDocuments	= result.document.size();
	result.nonzeros		= result.feature.size();	
	result.document.push_back(result.feature.size());
	
	return result;
}

void normalize(csr &data)
{
	float csum		= 0.0;
	for(int i=0;i<data.document.size()-1;i++)
	{
		csum	= 0.0;
		for(int j=data.document[i];j<data.document[i+1];j++) csum = csum + data.values[j] * data.values[j];
		csum	= sqrt(csum);
		for(int j=data.document[i];j<data.document[i+1];j++) data.values[j] = data.values[j]/csum;
	}
}

csc constructCSC(csr &data)
{
	csc result;
	result.document.resize(data.nonzeros,0);
	result.documentid.resize(data.nonzeros,0);
	result.values.resize(data.nonzeros,0);

	vector<int> colptr(data.maxFeatures,0);
	for(int i=0;i<data.document.size()-1;i++)
	{
		for(int j=data.document[i];j<data.document[i+1];j++) colptr[data.feature[j]] = colptr[data.feature[j]] + 1;
	}

	int csum = 0;
	for(int i=0;i<colptr.size();i++)
	{
		result.feature.push_back(csum);
		csum = csum + colptr[i];
		colptr[i] = 0;
	}
	result.feature.push_back(csum);
	
	for(int i=0;i<colptr.size();i++) colptr[i] = result.feature[i];
	for(int i=0;i<data.document.size()-1;i++)
	{
		for(int j=data.document[i];j<data.document[i+1];j++)
		{
			result.document[colptr[data.feature[j]]] 	= i;
			result.documentid[colptr[data.feature[j]]]	= data.documentid[i];
			result.values[colptr[data.feature[j]]]		= data.values[j];
			colptr[data.feature[j]]						= colptr[data.feature[j]] + 1;
		}
	}	
	return result;
}

vector<float> readRatings(char *fname)
{
	vector<float> result;
	FILE * fp = fopen(fname,"r");
	if(!fp) return result;

	float rating				= 0.0;
	while(fscanf(fp,"%f",&rating)!=EOF) result.push_back(rating); 
	return result;
}

pair<float,float> normalizeDenseVector(vector<float> &labels)
{
	float mu    = 0.0;
    float sigma = 0.0;
    float csum  = 0.0;

    for(int i=0;i<labels.size();i++) csum = csum + labels[i];
    mu   = csum/(int)labels.size();

    csum        = 0.0;
    for(int i=0;i<labels.size();i++) csum = csum + (labels[i]-mu)*(labels[i]-mu);
    sigma       = sqrt(csum/(int)labels.size());

    for(int i=0;i<labels.size();i++) labels[i] = (labels[i]-mu)/sigma;
    return make_pair(mu,sigma);	
}

vector<float> splitY(vector<float> &y,int start,int end)
{
	vector<float> result;
	for(int i=start;i<end;i++) result.push_back(y[i]);
	return result;
}

float sparseDenseVectorDotProduct(csr &X,int document,vector<float> &weights)
{
	float sum = 0.0;
	for(int j=X.document[document];j<X.document[document+1];j++) 
		sum = sum + weights[X.feature[j]] * X.values[j]; 
	return sum;
}

float getoffset(csr &X,vector< float > &y,vector<float> &weights)
{
    float sum   = 0.0;
	for(int i=0;i<X.document.size()-1;i++) sum = sum + y[i] - sparseDenseVectorDotProduct(X,i,weights);
    return sum/(int)X.documentid.size();
}

vector<float> matrixDotProduct(csr &X,vector<float> &weights)
{
    vector< float > result(X.document.size()-1,0.0);
    for(int i=0;i<X.document.size()-1;i++) result[i] = sparseDenseVectorDotProduct(X,i,weights);
    return result;
}

float cost(csr &X,vector<float> &y,vector<float> &weights,float lambda,float offset)
{
	float csum  = 0.0;
    float tsum  = 0.0;
    for(int i=0;i<X.document.size()-1;i++)
    {
        tsum    = sparseDenseVectorDotProduct(X,i,weights);
		tsum    = tsum + offset - y[i];
        csum    = csum + tsum*tsum;
    }

    tsum        = 0.0;
    for(int i=0;i<weights.size();i++) tsum = tsum + fabs(weights[i]);
    return csum + lambda*tsum;
}

float gradient(csc &X,vector< float >  &y,vector<float> &weights,vector<float> &dotProduct,int feature,float offset,float lambda)
{
    float tsum  = 0.0;
    float csum  = 0.0;
    float res   = 0.0;

    for(int i=X.feature[feature];i<X.feature[feature+1];i++)
    {
        tsum    = dotProduct[X.document[i]] - X.values[i]*weights[feature];
        tsum    = y[X.document[i]] - tsum - offset;
        res     = res + tsum*X.values[i];
        csum    = csum+ (X.values[i])*(X.values[i]);
    }

    csum        = 2*csum;
    res         = 2*res;

    if(res > lambda) return (res-lambda)/csum;
    else if(res < -1.0*lambda) return (res+lambda)/csum;
    else return 0;
}

vector<float> runCoordinateDescent(csr &X,csc &Xt,vector<float> &y,float lambda)
{
	float c1        = 0.0;
    float c2        = 0.0;
    float offset    = 0.0;
    float nweight   = 0.0;

    vector<float> weights(X.maxFeatures,0.0);
    for(int i=0;i<ITER;i++)
    {
        offset                      = getoffset(X,y,weights);
        c1                          = cost(X,y,weights,lambda,offset);
        vector<float> dotProduct    = matrixDotProduct(X,weights);
        for(int j=0;j<X.maxFeatures;j++)
        {
            nweight = gradient(Xt,y,weights,dotProduct,j,offset,lambda);
            for(int k=Xt.feature[j];k<Xt.feature[j+1];k++) 
				dotProduct[Xt.document[k]] = dotProduct[Xt.document[k]] + Xt.values[k] * (nweight - weights[j]);
			weights[j]      = nweight;
		}
		c2  = cost(X,y,weights,lambda,offset);
        if(c1-c2 < TOLERANCE) break;
    }
    return weights;
}


float predictSingle(csr &X,int document,vector<float> &weights,pair<float,float> &norm_constants)
{
    float sum = 0.0;
    for(int i=X.document[document];i<X.document[document+1];i++) sum = sum + X.values[i]*weights[X.feature[i]];
    return sum*norm_constants.second + norm_constants.first;
}

vector<float> predict(csr &X,vector<float> &weights,pair<float,float> &norm_constants)
{
    vector<float> result;
    for(int i=0;i<X.document.size()-1;i++) result.push_back(predictSingle(X,i,weights,norm_constants));
    return result;
}

float computeError(vector<float> actual,vector<float> predicted)
{
    float csum  = 0.0;
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
	csr input							= readInput(argv[1]);
	normalize(input);

	csr train							= split(input,0,TR_SIZE);
	csr validation						= split(input,TR_SIZE,TR_SIZE+VAL_SIZE);
	csr test							= split(input,TR_SIZE+VAL_SIZE,input.document.size()-1);

	csc train_csc						= constructCSC(train);
	csc validation_csc					= constructCSC(validation);
	csc test_csc						= constructCSC(test);

	vector<float> y						= readRatings(argv[2]);
	vector<float> ytrain				= splitY(y,0,train.documentid.size());
	pair<float,float> norm				= normalizeDenseVector(ytrain);
	vector<float> yvalidation			= splitY(y,train.documentid.size(),train.documentid.size()+validation.documentid.size());
	vector<float> ytest					= splitY(y,train.documentid.size()+validation.documentid.size(),input.documentid.size());

	vector<float> lambda                = getLambda();
    float best_error                    = 1e6;
    float best_lambda                   = 0;

    vector<float> best_weights;

    for(int i=0;i<lambda.size();i++)
    {
        vector<float> weights           = runCoordinateDescent(train,train_csc,ytrain,lambda[i]);
		vector<float> predictions       = predict(validation,weights,norm);
        float rmse                      = computeError(yvalidation,predictions);
        if(rmse < best_error)
        {
            best_error                  = rmse;
            best_lambda                 = lambda[i];
            best_weights                = weights;
        }
    }

	printf("Best lambda %.4f\n",best_lambda);
    vector<float> predictions           = predict(test,best_weights,norm);
    writeOutput(argv[3],predictions);
	return 0;
}
