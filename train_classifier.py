import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the processed data
data_dict = pickle.load(open('./asl_data.pickle', 'rb'))

# Define expected length based on the number of landmarks
# Assuming 21 landmarks and 2 coordinates (x, y) for each
expected_length = 42

# Filter out inconsistent data samples
consistent_data = []
consistent_labels = []

for i, sample in enumerate(data_dict['data']):
    if len(sample) == expected_length:  # Check if the sample has the expected length
        consistent_data.append(sample)
        consistent_labels.append(data_dict['labels'][i])

# Convert data and labels to numpy arrays
data = np.asarray(consistent_data)
labels = np.asarray(consistent_labels)

# Encode the labels (ASL letters) to numeric values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)

# Train the model on the training data
model.fit(x_train, y_train)

# Predict the labels on the test set
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_test, y_predict)

# Print the accuracy
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model to a file using pickle
with open('asl_model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)
