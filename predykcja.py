from datasetTrain import X_traindata #ladowanie obrazkow do testowania
from labelsService import labelsTrain #ladowanie labelsów do testowania
import tensorflow as tf
from labelsService import itemsTrain
print("Recreate the exact same model")
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
#Potrzebujemy tablicy z NAZWĄ OBRAZKA, HASBIRD, PREDYKCJĄ

predictions = model.predict(X_test)
type(predictions)
#print(predictions[0])
#im=X_test[1].reshape(1,184, 372, 4)
#print(model.predict(im))
#print(itemsTrain[1])

#print("sprawdzenie obrazka nr 26")
#print(itemsTrain[1])
#print(y_test[1])
#to sie zgadza

for i in range(10):
    im=X_test[i].reshape(1,184, 372, 4)
    print(model.predict(im))
    print(itemsTrain[i])
    print("sprawdzenie obrazka nr:", i)
    print(itemsTrain[i])
    print(y_test[i])
