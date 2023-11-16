import numpy as np
import pandas as pd



df = pd.read_csv('mnist_train.csv') 


df['prime'] = df['label'].apply(lambda x: 1 if x > 1 and all(x % i != 0 for i in range(2, int(x**0.5) + 1)) else 0)


print(df)


df.to_csv('dataset.csv', index=False)