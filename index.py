import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv(r'C:\Users\bfens\OneDrive\Documents\Supervised_Machine_Learning_and_Model_Evaluation_Assignment\exampledataset1.csv')

# Split the dataset into features (X) and target (y)
X = data.drop("Target", axis=1)
y = data["Target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy score
print(f"Model Accuracy: {accuracy:.2f}")