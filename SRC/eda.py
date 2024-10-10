import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
# from data_processing import 

titanic_df = pd.read_csv("./Data/train.csv")
print(titanic_df.head())

def eda_sex_survived (df):
    """
    This function takes in the titanic dataframe and visualizes survival based on sex
    """
    sns.countplot(x='Sex', data=titanic_df, hue='Survived')
    plt.title('Count of Titanic Survivors by Sex')
    plt.show() 
    sns.histplot(data=titanic_df, x='Sex', hue='Survived', multiple='stack')
    plt.title('Count of Titanic Survivors by Sex')
    plt.show()

def eda_pclass_survived (df):
    """
    This function takes in the titanic dataframe and visualizes survival based on passenger class
    """
    sns.countplot(x='Pclass', data=titanic_df, hue='Survived')
    plt.title('Count of Titanic Survivors by Passenger class')
    plt.show()


if __name__ == "__main__":
    eda_sex_survived(titanic_df)
    eda_pclass_survived(titanic_df)