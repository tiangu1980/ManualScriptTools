import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load the CSV file with the training data
data = pd.read_csv('input_train.csv')

# Extract the first three columns
X = data.iloc[:, :3]

# Extract the target column
y = data.iloc[:, 3:]

# Convert strings to numbers
encoder = LabelEncoder()
X = X.apply(encoder.fit_transform)

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Load the CSV file with the samples to predict
new_samples = pd.read_csv('input_work.csv')

# Extract the first three columns
X_new = new_samples.iloc[:, :3]

# Convert strings to numbers
X_new = X_new.apply(encoder.transform)

# Predict the class of the new samples
predicted_classes = model.predict(X_new)

# Print the predicted classes for each sample
for i in range(len(predicted_classes)):
    print(f"Sample {i+1}: {X_new.iloc[i, :]} -> Predicted class: {predicted_classes[i]}")
