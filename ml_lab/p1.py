import csv
filename ='Dataset1.csv'
lines = csv.reader(open(filename))
data = list()
for row in lines:
	if row[-1] == "Y":
		data.append(row[:-1])

hyp = data[0][:]
for i in data:
	k=0
	for j in i:
		if hyp[k] != j:
			hyp[k] = '?'
		k=k+1
print ("Final hypothesis : ", hyp)