# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load the credit card transaction dataset
# Ensure that the dataset contains relevant features such as transaction amount, location, and timestamp.
credit_card_data = pd.read_csv('data/credit_card_transactions.csv')

# Feature engineering and data preprocessing
# Feature engineering might involve creating new features or transforming existing ones to capture fraud patterns.
# StandardScaler is used to normalize features, an important step for Logistic Regression.
features = credit_card_data[['transaction_amount', 'location', 'timestamp']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Define the target variable (fraud label) and split the dataset into training and testing sets
# It's crucial to maintain the temporal order of transactions to simulate real-world scenarios.
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, credit_card_data['fraud_label'], test_size=0.2, shuffle=False
)

# Initialize and train the Logistic Regression model
# Logistic Regression is chosen for its suitability in binary classification tasks.
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model's performance
# Precision, recall, and F1 score are crucial metrics for assessing the model's ability to detect fraud while minimizing false positives.
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

# Display the results
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
