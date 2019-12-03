from bayespy.nodes import Dirichlet, Categorical, MultiMixture
from csv import reader
from pprint import pprint
import numpy as np


a = {'SuperSeniorCitizen':0, 'SeniorCitizen':1, 'MiddleAged':2, 'Youth':3,'Teen':4}
b = {'Male':0, 'Female':1}
c = {'Yes':0, 'No':1}
d = {'High':0, 'Medium':1, 'Low':2}
e = {'Athlete':0, 'Active':1, 'Moderate':2, 'Sedetary':3}
f = {'High':0, 'BorderLine':1, 'Normal':2}
g = {'Yes':0, 'No':1}

dataset = list(reader(open('Dataset7.csv')))

dataset = [ [ a[x[0]],b[x[1]],c[x[2]],d[x[3]],e[x[4]],f[x[5]],g[x[6]] ] for x in dataset]
dataset=np.array(dataset)
attr = [5,2,2,3,4,3]
n = len(dataset)
arr = []
for i in range(6):
    dirichlet = Dirichlet(np.ones(attr[i]))
    arr.append(Categorical(dirichlet, plates=(n,)))
    arr[i].observe(dataset[:, i])

target = Dirichlet(np.ones(2), plates=(5,2,2,3,4,3))
model = MultiMixture(arr, Categorical, target)
model.observe(dataset[:, -1])
target.update()

tup = [int(input()) for i in range(6)]
result = MultiMixture(tup, Categorical, target).get_moments()[0][0]
print(result)