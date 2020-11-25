from datasetTrain import X_traindata #ladowanie obrazkow do testowania
from labelsService import labelsTrain #ladowanie labelsów do testowania
import tensorflow as tf
from labelsService import itemsTrain
from numpy import argmax

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(bcolors.WARNING +"Recreate the exact same model"+ bcolors.ENDC)
model = tf.keras.models.load_model('model.h5')

itemsTrain = itemsTrain[7001:7690]
X_testdata = X_traindata[7001:7690].astype('float32') / 255
X_traindata = X_traindata[0:7000].astype('float32') / 255

labelsTest = labelsTrain[7000:7689]
labelsTrain = labelsTrain[0:7000]

print('Data loading...')

X_train = X_traindata
X_test = X_testdata
y_train = labelsTrain
y_test = labelsTest
print(bcolors.OKGREEN +'Data load!'+ bcolors.ENDC)

print("Prediction with [test] data")
y_pred = model.predict_classes(X_test)
missed=[]
matched=[]
for i in range(len(y_pred)):
    y_val_label_int = argmax(y_test[i])
    if (y_pred[i]!=y_val_label_int):
        missed.append( (y_pred[i], "-", [y_pred[i]], " - ", [y_val_label_int] ))
    else:
        matched.append((y_pred[i], "-", [y_pred[i]], " - ", [y_val_label_int]))

print ("  |__",bcolors.OKGREEN +"match"+"    :"+ bcolors.ENDC, len(matched))
print ("  |__",bcolors.FAIL +"miss"+"     :"+ bcolors.ENDC, len(missed))
print ("  |__accuracy :", round((len(matched)-len(missed))/len(matched)*100,2), "%")
print ("")
#print ("Value missed : \n",missed)

# pokazanie kilku wyników

print (bcolors.HEADER +"---samples---"+ bcolors.ENDC)
for i in range(8):
    print (i,"nazwa pliku:", str(itemsTrain[i])+".png")
    print (i,"predict =", [y_pred[i]])
    print (i,"original=", [argmax(y_test[i])])
    print("-----------")
    print ("")