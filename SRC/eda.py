import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from SRC.data_processing import df


def eda_sex_survived (df):
    """
    This function takes in the titanic dataframe and visualizes survival based on sex
    """
    sns.countplot(x='Sex', data=df, hue='Survived')
    plt.title('Count of Titanic Survivors by Sex')
    plt.show() 
    sns.histplot(data=df, x='Sex', hue='Survived', multiple='stack')
    plt.title('Count of Titanic Survivors by Sex')
    plt.show()

def eda_pclass_survived (df):
    """
    This function takes in the titanic dataframe and visualizes survival based on passenger class
    """
    sns.countplot(x='Pclass', data=df, hue='Survived')
    plt.title('Count of Titanic Survivors by Passenger class')
    plt.show()


if __name__ == "__main__":
    titanic_df = df
    eda_sex_survived(titanic_df)
    eda_pclass_survived(titanic_df)