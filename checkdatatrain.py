#Sprawdzam czy dane do trenowania są w tej samej kolejności co labelsy

from datasetTrain import X_traindata, traindatalist
from labelsService import itemsTrain, labelsTrain
import numpy as np

print(traindatalist[0:16])
print(itemsTrain[0:16])
print(labelsTrain[0:16])
print(np.unique(labelsTrain))
#jest ok