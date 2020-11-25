#load train data
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pickle

testdataload = glob.glob('./imageptaki/test/*.png')

#sortowanie listy odpowiednio do labelsów
testdatalist = []
for i in range(len(testdataload)):
    testdatalist.append(testdataload[i][18:-4])

testdatalist = sorted(testdatalist)

for i in range(len(testdatalist)):
    testdatalist[i] = testdatalist[i]+'.png'
#koniec sortowania

#Zapisanie danych do pliku by nie ładować ich za każdym uruchomieniem programu
#X_testdata = np.array([np.array(Image.open('./imageptaki/test/'+fname)) for fname in testdatalist])
#X_testdata.dump('./imageptaki/X_testdata/X_testdata.npy')

#załadowanie danych
X_testdata = np.load('./imageptaki/X_testdata/X_testdata.npy', allow_pickle=True)
#print(X_testdata[:5])