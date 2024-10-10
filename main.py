import pandas as pd
import numpy as np
import os
from SRC.model import RandomForest,xgboostmodel1
from SRC.data_processing import df

if __name__ == "__main__":
    RandomForest(df)
    xgboostmodel1(df)
