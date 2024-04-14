import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('datasets/rgb_colorname.csv')
model_path = 'models/color_model.pkl'

# Separate features (RGB values) and target (major color)
X = data[['red', 'green', 'blue']]
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# # Choose a classifier (e.g., Random Forest)
# clf = RandomForestClassifier(n_estimators=100)

# # Train the classifier
# clf.fit(X_train.values, y_train.values)

# # save the model
# with open(model_path, 'wb') as f:
#     pickle.dump(clf, f)


clf = pickle.load(open('models/color_model.pkl', 'rb'))
X_test = [[59, 90, 148], [221, 218, 219], [33, 46, 82]]
# X_test = [[164, 100, 250], [94, 191, 84], [198, 66, 66]]

# Make predictions
y_pred = clf.predict(X_test)
print(y_pred)
# # Evaluate the classifier
# print(classification_report(y_test, y_pred))
