#Sprawdzam czy dane do testowania są w tej samej kolejności co labelsy

from datasetTest import X_testdata, testdatalist
from labelsServiceTest import itemsTest, labelsTest
import numpy as np

print(testdatalist[0:10])
print(itemsTest[0:5])
print(labelsTest[0:5])
print(np.unique(labelsTest))

#jest ok