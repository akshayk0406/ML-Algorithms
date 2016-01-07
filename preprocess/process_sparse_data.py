import sys

input_file	= sys.argv[1]
output_file	= sys.argv[2]

result		= []
with open(input_file,'r') as f:
	for line in f:
		tokens	= line.split(" ")
		result.append((int(tokens[0]),int(tokens[1])-1,float(tokens[2])))

with open(output_file,'w') as f:
	for tup in result:
		f.write(str(tup[0]) + " " + str(tup[1]) + " " + str(tup[2]) + "\n")


