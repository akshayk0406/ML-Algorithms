#include<cstdio>
#include<iostream>
#include<algorithm>
#include<cmath>
#include<vector>
#include<string>
#include<set>
#include<map>


using namespace std;
typedef pair< int, float > fii; //movie-id,user-id,rating

map<int,string> readClassFile(char *fname)
{
	int docid = 0;
	char label[512];

	map<int,string> result;
	FILE *fp = fopen(fname,"r");
	if(!fp) return result;

	while(fscanf(fp,"%d%s",&docid,label)!=EOF) result[docid] = string(label);
	fclose(fp);
	
	return result;
}

vector<int> getDocId(char *fname)
{
	vector<int> result;
	FILE *fp = fopen(fname,"r");
	if(!fp) return result;
	
	int docid = 0;
	while(fscanf(fp,"%d",&docid)!=EOF) result.push_back(docid);
	fclose(fp);
	
	return result;
}

map<int, vector< fii > >readTrainData(char *fname)
{
	map<int, vector< fii > > result;
	FILE *fp = fopen(fname,"r");
	if(!fp) return result;

	int docid = 0;
	int featureid = 0;
	float freq = 0.0;

	while(fscanf(fp,"%d%d%f",&docid,&featureid,&freq)!=EOF)
	{
		if(result.find(docid) == result.end()) { vector< fii > document; result[docid] = document; }
		result[docid].push_back(make_pair(featureid,freq));
	}
	return result;
}

map<int, vector< fii > >getData(vector<int> &s,map<int, vector< fii > > &input)
{
	map<int, vector< fii > > result;
	for(vector<int>::iterator it = s.begin() ; it!=s.end() ; it++ ) result[*it] = input[*it];
	return result;
}

map<string,float> getClassFrequency(map<int,string> &class_map,map<int, vector< fii > > &input)
{
	string class_name = "";
	map<string,float> result;
	for(map<int,vector< fii > >::iterator it = input.begin() ; it!= input.end() ; it++ )
	{
		class_name = class_map[it->first];
		if(result.find(class_name)==result.end()) result[class_name] = 0.0;
		for(int i=0;i<it->second.size();i++) result[class_name] = result[class_name] + it->second[i].second;
	}
	return result;
}

map<int,float> getDocumentFrequency(map<int, vector< fii > > &input)
{
	float csum = 0.0;
	map<int,float> result;
	for(map<int, vector< fii > >::iterator it = input.begin(); it!=input.end();it++)
	{
		csum = 0.0;
		for(int i=0;i<it->second.size();i++) csum = csum + it->second[i].second;
		result[it->first] = csum;
	}
	return result;
}

set<int> getFeatures(map<int, vector< fii > > &input)
{
	set<int> result;
	for(map<int, vector< fii > >::iterator it = input.begin(); it!=input.end();it++)
	{
		for(int i=0;i<it->second.size();i++) result.insert(it->second[i].first);
	}
	return result;
}	

map<pair<string,int>,float> getFeaturesByClass(map<int,string> &class_map,map<int, vector< fii > > &input)
{
	string class_name 	= "";
	int featureid		= 0;

	map<pair<string,int>,float> result;
	for(map<int, vector< fii > >::iterator it = input.begin(); it!=input.end();it++)
	{
		class_name 		= class_map[it->first];
		for(int i=0;i<it->second.size();i++)
		{
			pair<string,int> P = make_pair(class_name,it->second[i].first);
			if(result.find(P)==result.end()) result[P] = 0.0;
			result[P] = result[P] + it->second[i].second;
		}
	}
	return result;
}

map<string,float> getClassPrior(map<int,string> &class_map,map<int, vector< fii > > &input)
{
	int total = (int)input.size();
	string class_name 	= "";
	map<string,float> result;
	for(map<int, vector< fii > >::iterator it = input.begin(); it!=input.end();it++)
	{
		class_name 		= class_map[it->first];
		if(result.find(class_name)==result.end()) result[class_name] = 0.0;
		result[class_name] = result[class_name] + 1.0;
	}
	for(map<string,float>::iterator it = result.begin(); it!=result.end(); it++) it->second = it->second/total;
	
	return result;
}

map<int,float> getFeatureFrequency(map<int, vector< fii > > &input)
{
	map<int,float> result;
	for(map<int, vector< fii > >::iterator it = input.begin(); it!=input.end();it++)
	{
		for(int i=0;i<it->second.size();i++) 
		{
			if(result.find(it->second[i].first)==result.end()) result[it->second[i].first] = 0;
			result[it->second[i].first] = result[it->second[i].first] + it->second[i].second;
		}
	}
	return result;
}

int prediction(map<int, vector< fii > > &input,map<pair<string,int>,float> &feature_class_freq,
				map<string,float> &class_freq,map<int,float> &document_freq,
				map<int,string> &class_map,map<string,float> &priors,
				map<int,float> &feature_frequency,int vocab_size)
{
	int correct = 0;
	float num = 0.0;
	float den = 0.0;
	float score = 0.0;

	float total_words = 0.0;
	for(map<string,float>::iterator it1 = class_freq.begin(); it1!= class_freq.end(); it1++) total_words = total_words + it1->second;

	for(map<int, vector< fii > >::iterator it = input.begin(); it!=input.end();it++)
	{
		float best_score = -1e7;
		string label = "";
		for(map<string,float>::iterator it1 = class_freq.begin(); it1!= class_freq.end(); it1++)
		{	
			score = log(priors[it1->first]);
			for(int i=0;i<it->second.size();i++)
			{
				num = feature_frequency[it->second[i].first] - feature_class_freq[make_pair(it1->first,it->second[i].first)]+1.0;
				den = total_words - it1->second + vocab_size;
				score = score - it->second[i].second * log((float)num/den);
			}
			if(score > best_score)
			{
				best_score = score;
				label = it1->first;
			}
		}
		if(label == class_map[it->first]) correct = correct + 1;
	}
	return correct;
}

int main(int argc,char **argv)
{
	map<int,vector< fii > > data 					= readTrainData(argv[1]);
	map<int,string> class_map 						= readClassFile(argv[2]);
	vector<int> train_docid							= getDocId(argv[3]);
	vector<int> test_docid							= getDocId(argv[4]);
	map<int, vector< fii > > train_data 			= getData(train_docid,data);
	map<int, vector< fii > > test_data 				= getData(test_docid,data);
	map<string,float> class_freq					= getClassFrequency(class_map,train_data);
	map<int,float> document_freq					= getDocumentFrequency(train_data);
	set<int> feature_set							= getFeatures(train_data);
	map<string,float> priors						= getClassPrior(class_map,train_data);
	map<pair<string,int>,float> feature_class_freq	= getFeaturesByClass(class_map,train_data);
	map<int,float> feature_frequency				= getFeatureFrequency(train_data);
	int correct										= prediction(test_data,feature_class_freq,class_freq,document_freq,
																class_map,priors,feature_frequency,(int)feature_set.size());
	printf("Accuracy %.6f\n",(float)correct/(int)test_docid.size());
	return 0;
}
