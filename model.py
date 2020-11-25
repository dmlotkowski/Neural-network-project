from datasetTrain import X_traindata #ladowanie obrazkow do trenowania
from labelsService import labelsTrain #ladowanie labelsów do trenowania
from datasetTest import X_testdata #ladowanie obrazkow do testowania
#from labelsServiceTest import labelsTest #ladowanie labelsów od testowania
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

X_testdata = X_traindata[7001:7690].astype('float32') / 255
X_traindata = X_traindata[0:7000].astype('float32') / 255

labelsTest = labelsTrain[7000:7689]
labelsTrain = labelsTrain[0:7000]


print('Data loading...')

X_train = X_traindata
X_test = X_testdata
y_train = labelsTrain
y_test = labelsTest

y_train_one_hot = keras.utils.to_categorical(y_train, 2)
y_test_one_hot = keras.utils.to_categorical(y_test, 2)

print('Dataset ready!')
print('Next block...')
print('=======================')
print("Stary wymiar taindata",X_train.shape)
print("Stary wymiar testdata",X_test.shape)
print('=======================')
print('Reshape stared...')
X_train = X_train.reshape(7000, 184, 372, 4)
X_test = X_test.reshape(689, 184, 372, 4)
print('Reshape sucess!')


model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(184, 372, 4)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.summary()



model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("*** Model compiled ***")


hist = model.fit(X_train, y_train_one_hot, 
           batch_size=64, epochs=3, 
           validation_split=0.2)

#y_train = np.asarray(y_train)
#X_train = np.asfarray(X_train)


model.save("modelepoch3.h5")
print("Model saved to file model.h5")

# Recreate the exact same model, including its weights and the optimizer

print("Recreate the exact same model")
model = tf.keras.models.load_model('modelepoch10.h5')


test_loss, test_acc = model.evaluate(X_test, y_test_one_hot, verbose=0)
print('\nTest accuracy:', test_acc)

predictions = model.predict(X_test)
type(predictions)
print(predictions[0])
im=X_train[16].reshape(1,184, 372, 4)
print(model.predict(im))

print("sprawdzenie obrazka nr 16")
print(X_train[16])
print(y_train[16])
#to sie zgadza