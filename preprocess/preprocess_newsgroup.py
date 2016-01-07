import sys

input_file 	= sys.argv[1]
class_file 	= sys.argv[2]
train_size	= float(sys.argv[3])
train_out	= sys.argv[4]
test_out	= sys.argv[5]

doc_class_map 	= {}
class_freq 		= {}
cur_freq		= {}

train			= []
test			= []

with open(class_file,'r') as f:
	for line in f:
		tokens = line.split(" ")
		docid = int(tokens[0])
		class_name = tokens[1]
		doc_class_map[docid] = class_name
	
		if class_name not in class_freq:
			class_freq[class_name] = 0
			cur_freq[class_name] = 0	

		class_freq[class_name] = class_freq[class_name] + 1

pre_doc = -1
with open(input_file,'r') as f:
	for line in f:
		tokens = line.split(" ")
		
		docid = int(tokens[0])
		feature = int(tokens[1])
		freq = int(tokens[2])
		doc_class = doc_class_map[docid]

		if pre_doc!= docid:
			cur_freq[doc_class] = cur_freq[doc_class] + 1
		
		cur_count = cur_freq[doc_class]
	
		if cur_count > class_freq[doc_class] * train_size:
			test.append(line)
		else:	
			train.append(line)
		
		pre_doc = docid

with open(train_out,'w') as f:
	for line in train:
		f.write(line)

with open(test_out,'w') as f:
	for line in test:
		f.write(line)

