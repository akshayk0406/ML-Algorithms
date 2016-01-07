#include<cstdio>
#include<iostream>
#include<sstream>
#include<algorithm>
#include<cmath>
#include<vector>
#include<string>
#include<set>
#include<map>

using namespace std;

float gauss(float x,float m,float s)
{
	static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - m) / s;
    return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
}

std::vector<float> split(const std::string &s, char delim) {
    std::vector<float> elems;
    std::stringstream ss(s);
    std::string number;
    while(std::getline(ss, number, delim)) {
        elems.push_back(std::stof(number));
    }
    return elems;
}

void readInput(vector< vector<float> > &X,vector<int> &y,char *fname)
{
	FILE *fp = fopen(fname,"r");
	if(!fp) return;

	char str[4098];
	while(fscanf(fp," %[^\n]",str)!=EOF)
	{
		vector<float> features = split(string(str),',');
		y.push_back((int)features[features.size()-1]);
		features.pop_back();
		X.push_back(features);
	} 
	fclose(fp);
}

void computeStd(vector< vector<float> > &X,vector<int> &y,map<int, vector<float> > &class_means,map<int, vector<float> > &class_std)
{
	if(X.size()==0) return;	
	
	map<int,int> class_freq;
	vector<float> res(X[0].size(),0.0);
	for(int i=0;i<X.size();i++)
	{
		if(class_freq.find(y[i])==class_freq.end()) class_freq[y[i]] = 1;
		else class_freq[y[i]] = class_freq[y[i]] + 1;
		
		if(class_std.find(y[i])==class_std.end()) class_std[y[i]] = res;
		for(int j=0;j<X[i].size();j++)
			class_std[y[i]][j] = class_std[y[i]][j] + (X[i][j]-class_means[y[i]][j])*(X[i][j]-class_means[y[i]][j]);
	}
	
	for(map<int,vector<float> >::iterator it = class_std.begin(); it != class_std.end() ; it++ )
	{
		for(int i=0;i<it->second.size();i++) it->second[i] = sqrt(it->second[i]/class_freq[it->first]);
	}
}

void computeMean(vector< vector<float> > &X,vector<int> &y,map<int, vector<float> > &class_means)
{
	if(X.size() == 0) return;

	map<int,int> class_freq;
	vector<float> res(X[0].size(),0.0);
	for(int i=0;i<X.size();i++)
	{
		if(class_freq.find(y[i])==class_freq.end()) class_freq[y[i]] = 1;
		else class_freq[y[i]] = class_freq[y[i]] + 1;
		
		if(class_means.find(y[i])==class_means.end()) class_means[y[i]] = res;
		for(int j=0;j<X[i].size();j++)
			class_means[y[i]][j] = class_means[y[i]][j] + X[i][j];
	}
	
	for(map<int,vector<float> >::iterator it = class_means.begin(); it != class_means.end() ; it++ )
	{
		for(int i=0;i<it->second.size();i++) it->second[i] = it->second[i]/class_freq[it->first];
	}
}

int predict(vector<float> &input,map<int,vector<float> > &class_means,map<int,vector<float> > &class_std)
{
	float best_score = -1e8;
	int label = 0;

	map<int,vector<float> >::iterator it1 = class_means.begin();
	map<int,vector<float> >::iterator it2 = class_std.begin();
	
	while(it1 != class_means.end())
	{
		float score = 0;
		for(int i=0;i<it1->second.size();i++) 
			score = score + log(gauss(input[i],it1->second[i],it2->second[i]));
		if(score>best_score)
		{
			best_score = score;
			label = it1->first;
		}
		it1++;
		it2++;	
	}
	return label;
}

int main(int argc,char **argv)
{
	vector< vector<float> > Xtrain,Xtest;
	vector< int > Ytrain,Ytest;
	
	readInput(Xtrain,Ytrain,argv[1]);
	readInput(Xtest,Ytest,argv[2]);

	map<int,vector<float> >class_means;
	map<int,vector<float> >class_std;
	
	computeMean(Xtrain,Ytrain,class_means);
	computeStd(Xtrain,Ytrain,class_means,class_std);

	int correct = 0;
	int pred = 0;
	for(int i=0;i<Xtest.size();i++)
	{
		pred = predict(Xtest[i],class_means,class_std);
		if(pred==Ytest[i]) correct = correct + 1;
	}
	printf("Accuracy is %.6f\n",(float)correct/(int)Xtest.size());
	return 0;
}
