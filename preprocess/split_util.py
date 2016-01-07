
import sys
file_name = sys.argv[1]
train_size = float(sys.argv[2])
output_train = sys.argv[3]
output_test = sys.argv[4]

data = []
class_freq = {}
cur_freq = {}
with open(file_name,'r') as f:
	for line in f:
		tokens = line.split(",")
		class_name = tokens[0];
		
		if class_name not in class_freq:
			class_freq[class_name] = 0
			cur_freq[class_name] = 0			

		class_freq[class_name] = class_freq[class_name] + 1
		data.append(line)

train = []
test = []

for line in data:
	tokens = line.split(",")
	class_name = tokens[0]
	
	if class_freq[class_name] * train_size > cur_freq[class_name]:
		train.append(line)
	else:
		test.append(line)
	
	cur_freq[class_name] = cur_freq[class_name] + 1

with open(output_train,'w') as f:
	for line in train:
		f.write(line)

with open(output_test,'w') as f:
	for line in test:
		f.write(line)

