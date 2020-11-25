from datasetTrain import X_traindata #ladowanie obrazkow do testowania
from labelsService import labelsTrain #ladowanie labelsów do testowania
import tensorflow as tf
from labelsService import itemsTrain
from numpy import argmax
from pandas import DataFrame
import pandas as pd

print("Recreate the exact same model")
model = tf.keras.models.load_model('model.h5')

print('Data loading...')
itemsTrain = itemsTrain[7001:7690]
X_testdata = X_traindata[7001:7690].astype('float32') / 255
print("Test data - OK")
X_traindata = X_traindata[0:7000].astype('float32') / 255
print("Train data - OK")

labelsTest = labelsTrain[7000:7689]
labelsTrain = labelsTrain[0:7000]

X_train = X_traindata
X_test = X_testdata
y_train = labelsTrain
y_test = labelsTest
y_pred = model.predict_classes(X_test)

# zapisanie rezultatów
prediction_output_file='prediction_result_1.csv'
with open(prediction_output_file,"w") as file:
    file.write("ID,Original,Prediction\n") 
    i=0
    for i in range( (len(X_test)-1)) :
        #print(i, y_pred[i])
        file.write(str(itemsTrain[i])+".png"+","+str(argmax(y_test[i]))+","+str(y_pred[i]))
        file.write('\n')
        i=i+1
        
print (len(y_pred))
output = pd.read_csv(prediction_output_file)
print(output.head(20))