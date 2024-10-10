import xgboost as xgb 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv("./Data/train.csv")

def xgboostmodel1 (df):
    """
    This function runs an xg boost model to predict the survival outcome based on input variables
    """

    # label encoding for different data types required by xgboost:
    le = LabelEncoder()
    df = df.copy()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    # categorical type data
    df['Cabin'] = df['Cabin'].astype('category')
    # do not provide any useful information in this case, so we drop these columns:
    df = df.drop('Name', axis=1)
    df = df.drop('PassengerId', axis=1)
    df = df.drop('Ticket', axis=1)


    # split the data into our outcome variable (dependent variable), and our input variables (independent variables)
    print(df.head())
    # survival data
    survival_outcome = df['Survived']
    # all other variables except for our outcome (in a later model)
    input_variables = df.drop('Survived', axis=1)

    # splitting the data into train and test (our test.csv does not inlcude survived column, do we have anything to compare to? in the meantime, splitting train data)
    # random_state for reproducible results
    input_train, input_test, survival_train, survival_test = train_test_split(input_variables, survival_outcome, test_size = 0.2, random_state=50)

    # in order to use categorical input variables, enable_categorical=True, random_state for reproducible results
    xgbmodel1 = XGBClassifier(enable_categorical= True, random_state=50)
    # fit the model
    xgbmodel1.fit(input_train, survival_train)

    # collect model predictions so we can compare to the actual data to test for accuracy
    survival_predictions = xgbmodel1.predict(input_test)

    # return these so we can test for accuracy in model_eval.py functions
    return survival_test, survival_predictions


if __name__ == "__main__":
    xgboostmodel1(train_df)