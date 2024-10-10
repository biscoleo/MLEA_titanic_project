from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from model_eval import Random_forest_eval
from data_processing import df
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def RandomForest():
    x = df[['Pclass','Sex','Age','SibSp','Parch','Fare']]
    y = df['Survived']

    x.loc[:, 'Sex'] = x['Sex'].map({'Female':0,'Male':1})

    # Handle missing values in the 'Age' column using .loc
    x.loc[:, 'Age'].fillna(x['Age'].median(), inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

    rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=7)

    rf_classifier.fit(x_train, y_train)

    y_pred = rf_classifier.predict(x_test)

    importances = list(rf_classifier.feature_importances_)
    # print(importances)
    # print(y_pred)

    Random_forest_eval(y_test,y_pred)

if __name__ == "__main__":
    RandomForest()