import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model and label encoder
with open('asl_model.p', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
label_encoder = model_data['label_encoder']

# Load the processed data again (if needed)
data_dict = pickle.load(open('./asl_data.pickle', 'rb'))

# Define expected length based on the number of landmarks
expected_length = 42

# Filter out inconsistent data samples (similar to training code)
consistent_data = []
consistent_labels = []

for i, sample in enumerate(data_dict['data']):
    if len(sample) == expected_length:
        consistent_data.append(sample)
        consistent_labels.append(data_dict['labels'][i])

# Convert data and labels to numpy arrays
data = np.asarray(consistent_data)
labels = np.asarray(consistent_labels)

# Encode the labels (ASL letters) to numeric values
labels_encoded = label_encoder.transform(labels)

# Split the data into training and testing sets (if not already split)
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

# Predict the labels on the test set
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_predict, target_names=label_encoder.classes_, labels=np.unique(y_predict)))

# Confusion Matrix
cm = confusion_matrix(y_test, y_predict)

# Plotting the confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
