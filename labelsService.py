# LABELSY Z CSV POSORTOWANE
# inny sposób wydobycia labelsów na danych do Trenowania sieci (zamiast stosowania maski tj. masklabels.py)

from pandas import DataFrame
import pandas as pd

data = pd.read_csv("./ptaki/train/ff1010bird_metadata_2018.csv") 
df = DataFrame(data, columns = ['itemid','hasbird'])
df.sort_values(by=['itemid'], inplace=True)
hasbirdtrain = df.loc[:,"hasbird"]
itemidtrain = df.loc[:, "itemid"]

itemsTrain = list(itemidtrain)
labelsTrain = list(hasbirdtrain)