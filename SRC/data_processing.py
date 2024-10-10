import pandas as pd
import numpy as np

df = pd.read_csv("./Data/train.csv")
df = df.drop_duplicates()
df = df.dropna(how='any',axis=0)

