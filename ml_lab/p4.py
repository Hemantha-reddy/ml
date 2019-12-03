import numpy as np

def derivative(x):
    return x*(1-x)

def sigmoid(x):
    return(1/(1+np.exp(-x)))

x=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
y=y/100

epoch=6500
lr=0.1
h_layer=3
i_layer=2
o_layer=1

wh=np.random.uniform(size=(i_layer,h_layer))
bh=np.random.uniform(size=(o_layer,h_layer))
wout=np.random.uniform(size=(h_layer,o_layer))
bout=np.random.uniform(size=(o_layer,o_layer))

for i in range(epoch):

    hp=np.dot(x,wh)
    hp1=hp+bh
    hidden=sigmoid(hp1)

    out=np.dot(hidden,wout)
    out1=out+bout
    output=sigmoid(out1)

    Eo=y-output
    out_grad=derivative(output)
    d_out=Eo*out_grad

    Eh=d_out.dot(wout.T)
    hidden_grad=derivative(hidden)
    d_hidden=Eh*hidden_grad

    wout+=hidden.T.dot(d_out)
    wh+=x.T.dot(d_hidden)
print(y)
print(output)