from sklearn.metrics import accuracy_score, classification_report

def Random_forest_eval(test_values,predicted_values):
    # Evaluate the model
    accuracy = accuracy_score(test_values, predicted_values)
    classification_rep = classification_report(test_values, predicted_values)

    # Print the results
    print(f"Accuracy for Random Forest: {accuracy*100}")
    # print("\nClassification Report:\n", classification_rep)


def XG_boost_eval(test_values,predicted_values):
    accuracy= accuracy_score(test_values,predicted_values)
    print('Accuracy of the XGBoost model is:', accuracy*100)