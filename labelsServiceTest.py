#Maska do pliku Crowdsourced dataset, UK ("warblrb10k") - 8000 obrazków
from pandas import DataFrame
import pandas as pd

data = pd.read_csv("./ptaki/test/warblrb10k_public_metadata_2018.csv") 
df = DataFrame(data, columns = ['itemid','hasbird'])
df.sort_values(by=['itemid'], inplace=True)
hasbirdtest = df.loc[:,"hasbird"]
itemidtest = df.loc[:, "itemid"]

itemsTest = list(itemidtest)
labelsTest = list(hasbirdtest)