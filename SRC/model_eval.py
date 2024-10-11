from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

def Random_forest_eval(test_values,predicted_values):
    # Evaluate the model
    accuracy = accuracy_score(test_values, predicted_values)
    classification_rep = classification_report(test_values, predicted_values)

    # Print the results
    print(f"Accuracy for Random Forest: {accuracy*100}\n")
    # print("\nClassification Report:\n", classification_rep)


def XG_boost_eval(test_values,predicted_values):
    accuracy = accuracy_score(test_values,predicted_values)
    print('Accuracy of the XGBoost model is:', accuracy*100, "\n")


def logistic_regression_eval(test_values, predicted_values):
    accuracy = accuracy_score(test_values, predicted_values)
    print(f"Accuracy for Logistic Regression model: {accuracy*100}\n")

    # roc_auc = roc_auc_score(test_values, predicted_values)
    # print(f"AUC-ROC Score for Logistic Regression model: {roc_auc}")

    # print("Confusion Matrix:\n")
    # print(confusion_matrix(test_values, predicted_values))

    # print("Classification Report:\n")
    # print(classification_report(test_values, predicted_values))