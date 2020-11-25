#load train data
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pickle



traindataload = glob.glob('./imageptaki/train/*.png')

#sortowanie listy odpowiednio do labelsów
traindatalist = []
for i in range(len(traindataload)):
    traindatalist.append(traindataload[i][19:-4])

traindatalist = sorted(traindatalist, key=int)

for i in range(len(traindatalist)):
    traindatalist[i] = traindatalist[i]+'.png'
#koniec sortowania


Zapisanie danych do pliku by nie ładować ich za każdym uruchomieniem programu
#X_traindata = np.array([np.array(Image.open('./imageptaki/train/'+fname)) for fname in traindatalist])
#X_traindata.dump('./imageptaki/X_traindata/X_traindata.npy')

#załadowanie danych
X_traindata = np.load('./imageptaki/X_traindata/X_traindata.npy', allow_pickle=True)
#print(X_traindata.shape)
