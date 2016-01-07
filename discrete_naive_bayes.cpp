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

std::vector<string> split(const std::string &s, char delim) {
    std::vector<string> elems;
    std::stringstream ss(s);
    std::string number;
    while(std::getline(ss, number, delim)) {
        elems.push_back(number);
    }
    return elems;
}

void readInput(vector< vector<string> > &X,vector<string> &y,char *fname)
{
	FILE *fp = fopen(fname,"r");
	if(!fp) return;

	char str[4098];
	while(fscanf(fp," %[^\n]",str)!=EOF)
	{
		vector<string> features = split(string(str),',');
		y.push_back(features[features.size()-1]);
		features.pop_back();
		X.push_back(features);
	} 
	fclose(fp);
}

map<string,int> getClassDistribution(vector<string> &input)
{
	map<string,int> result;
	for(int i=0;i<input.size();i++)
	{
		if(result.find(input[i])==result.end()) result[input[i]] = 0;
		result[input[i]] = result[input[i]] + 1;
	}
	return result;
}

map< pair<string,string>,int > getClassConditionProb(vector< vector<string> > &input,vector<string> &y)
{
	//<feature,class> -> count
	map< pair<string,string>,int > result;
	for(int i=0;i<input.size();i++)
	{
		for(int j=0;j<input[i].size();j++)
		{
			pair<string,string> P = make_pair(input[i][j],y[i]);
			if(result.find(P) == result.end()) result[P] = 0;
			result[P] = result[P] + 1;
		}
	}
	return result;
}

string predict(vector<string> &input,map< pair<string,string>,int > class_cond_prob,map<string,int> &class_freq,int sample_size)
{
	float best_score = -1e7;
	string label = "";
	
	for(map<string,int>::iterator it = class_freq.begin() ; it != class_freq.end();  it++ )
	{
		float score = 0.0;
		string class_name = it->first;
		int freq = it->second;
	
		for(int i=0;i<input.size();i++)
		{
			pair<string,string> P = make_pair(input[i],class_name);
			if(class_cond_prob.find(P) == class_cond_prob.end()) score = score + log((float)1.0/(sample_size+freq)); //Laplace estimate
			else score = score + log((float)class_cond_prob[P]/freq);
		}
		score = exp(score)*((float)freq/sample_size);
		if(score > best_score)
		{
			best_score = score;
			label = class_name;	
		}
	}
	return label;
}

int main(int argc,char **argv)
{
	vector< vector<string> > Xtrain,Xtest;
	vector< string > Ytrain,Ytest;
	
	readInput(Xtrain,Ytrain,argv[1]);
	readInput(Xtest,Ytest,argv[2]);
	map<string,int> class_freq = getClassDistribution(Ytrain);
	map< pair<string,string>,int > class_cond_prob = getClassConditionProb(Xtrain,Ytrain);
		
	int correct = 0;
	string pred = "";
	for(int i=0;i<Xtest.size();i++)
	{
		pred = predict(Xtest[i],class_cond_prob,class_freq,Xtrain.size());
		if(pred==Ytest[i]) correct = correct + 1;
	}
	printf("Accuracy is %.6f\n",(float)correct/(int)Xtest.size());
	return 0;
}
