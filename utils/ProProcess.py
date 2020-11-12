# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
def get_data(Filetrain,Filetest):
    train,test = pd.read_csv(Filetrain),pd.read_csv(Filetest)
    return train,test

def generat_data(data:pd.DataFrame,batchsize:int,train=True):
    t1, t2 = [], []
    if train:
        while True:
            data = shuffle(data)
            for k,i in data.iterrows():

                if len(t1)==batchsize:
                    t1_,t2_ = np.array(t1),np.array(t2)
                    t1.clear()
                    t2.clear()
                    yield [t1_],[t2_]
                else:
                    t1.append(i.values[1:241])
                    t2.append(i.values[241])
    else:
        for k, i in data.iterrows():
                t1_= np.array([i.values[1:241]])
                yield [t1_]


if __name__=="__main__":
    pass
