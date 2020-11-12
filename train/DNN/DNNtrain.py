# -*- coding: utf-8 -*-
from model.DNN.DNNModel import get_model
from utils.ProProcess import get_data,generat_data
from tensorflow.keras.callbacks import ModelCheckpoint

BATCHSIZE=1


checkpoint = ModelCheckpoint(monitor="val_acc",
                    filepath="C:\\Users\\Administrator\\Desktop\\dd\\weights\\DNNweights\\model_{epoch:02d}-{val_acc:02f}.w",
                    save_best_only=True,
               save_weights_only=False,)

train ,test = get_data(Filetrain="C:\\Users\\Administrator\\Desktop\\dd\\data\\train.csv",Filetest="C:\\Users\\Administrator\\Desktop\\dd\\data\\test.csv")

#test = train.sample(frac=0.2)
#train = train[~train.index.isin(test)]



model = get_model()

model.fit_generator(
                    generat_data(train,BATCHSIZE),
                    steps_per_epoch=len(train)//BATCHSIZE,
                    epochs=9,
                    verbose=1,
                    validation_data=generat_data(train,BATCHSIZE),
                    validation_steps=len(train)//BATCHSIZE,
                    callbacks=[checkpoint]
                        )