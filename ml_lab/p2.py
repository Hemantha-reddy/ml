from csv import reader
from pprint import pprint


def classify(hypo, row):
	for a, b in zip(hypo, row):
		if not(a == '?' or a == b):
			return 'No'
	return 'Yes'


with open("Dataset2.csv") as f:
    dataset = [row for row in reader(f)]

#print(dataset)

positive_dataset = [row[:-1] for row in dataset if row[-1] == "P"]
negative_dataset = [row[:-1] for row in dataset if row[-1] == "N"]


hypo_len = len(positive_dataset[0])
s = positive_dataset[0][:]
g = [['?'] * hypo_len]




for row in positive_dataset[1:]:
    s = ["?" if tup[0] != tup[1] else tup[0]
                           for tup in zip(s, row)]
print(s)
for row in negative_dataset:
	new = []
	
	for hypo in g:
		
		if classify(hypo, row) == 'Yes':
			candidates = [hypo[:] for _ in range(hypo_len)]
			#print(candidates)

			for i in range(hypo_len):
				if candidates[i][i] == '?':
					candidates[i][i] = s[i]
			new += candidates
			print(new)
	g += new
	g = [x for x in g if classify(x, row) == 'No']


print("G IS ",g)