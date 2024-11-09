import pandas as pd
import numpy as np

class NaiveBayesianClassifier():
    def __init__(self):
        self.ClassesValues = {}
        self.features_names = None
        self.unique_classes = None

    def fit(self,X,Y):
        self.features_names = X.columns
        self.unique_classes = Y.value_counts()
        # iterate for each feature
        for feature_name in self.features_names:
            feature_unique_values = X[feature_name].unique()
            feature_result = {}
            # Get the probability of the value given each class
            for value in feature_unique_values:
                classes_counts = Y[X[feature_name] == value].value_counts()
                classes_result = [0 if CN not in classes_counts.index else classes_counts.loc[CN]/(Y[Y==CN].value_counts().iloc[0]) for CN in self.unique_classes.index]
                feature_result[value] = classes_result
            self.ClassesValues[feature_name] = feature_result

    def predict_one(self,x):
        classes_pred = self.unique_classes.values/np.sum(self.unique_classes)
        for value_index in range(len(x)):
            classes_pred = classes_pred * self.ClassesValues[self.features_names[value_index]][x[value_index]]
        # Find the class with the max probability
        max_index = np.argmax(classes_pred)
        return self.unique_classes.index[max_index]

    def predict(self,X):
        predictions = []
        for x in X: # Make prediction for each one alone
            result = self.predict_one(x)
            predictions.append(result)
        return predictions






# Load dataset
dataset = pd.read_csv("PlayTennis.csv",header=0)

# Print dataset shape
print(dataset.shape)
# Show first five samples
print(dataset.head())
# Print some info about the dataset
print(dataset.info())
# Show some statistical info
print(dataset.describe())

# Split the data into input and label data
X = dataset.drop(columns=["Play Tennis"])
Y = dataset["Play Tennis"]


# Create and train the model
BSClassifier = NaiveBayesianClassifier()
BSClassifier.fit(X,Y)
# Make predictions
X_test = np.array([['Sunny','Cool','High','Strong'],
                    ['Overcast','Hot','High','Weak'],
                    ['Rain','Hot','High','Weak'],
                    ['Sunny','Mild','Normal','Strong']])

predicted_values = BSClassifier.predict(X_test)





