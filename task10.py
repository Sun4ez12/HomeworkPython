import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder

lst = ['robot'] * 10
lst += ['human'] * 10

random.shuffle(lst)

data = pd.DataFrame({'whoAmI': lst})

encoder = OneHotEncoder(sparse_output=False)

one_hot = encoder.fit_transform(data[['whoAmI']])

one_hot_df = pd.DataFrame(one_hot, columns=encoder.categories_[0])

data_encoded = pd.concat([data, one_hot_df], axis=1)

print(data_encoded.head())