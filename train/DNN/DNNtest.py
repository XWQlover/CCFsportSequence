from model.DNN.DNNModel import get_model
from utils.ProProcess import get_data,generat_data
import numpy as np
import pandas as pd

model = get_model()
model.load_weights("C:\\Users\\Administrator\\Desktop\\dd\\weights\\DNNweights\\model_09-0.957143.w")

train ,test = get_data(Filetrain="C:\\Users\\Administrator\\Desktop\\dd\\data\\train.csv",Filetest="C:\\Users\\Administrator\\Desktop\\dd\\data\\test.csv")

result=[]
for i in generat_data(test,batchsize=1,train=False):
    result.append(np.argmax(model.predict(i)))


test["CLASS"]= pd.DataFrame(result)

test[["ID","CLASS"]].to_csv("final_submit.csv",index=False)