import pickle
import pandas as pd


# load res.pickle
with open('res.pickle', 'rb') as handle:
    res = pickle.load(handle)




# dict to pandas
df = pd.DataFrame(res)
print(df)

