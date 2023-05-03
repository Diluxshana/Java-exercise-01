import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Q1_Assignment_01_2019E033
"""
low argument sets the lower limit for the random integers
high argument sets the upper limit for the random integers
size set the length of matrix
"""
Random_matrix = np.random.randint(low=1, high=11, size=(2, 10))
print("\n Random Matrix between 1 and 10")
print(Random_matrix)

# Q2_Assignment_01_2019E033

# Consider the first column as the test data and the second column as the predicted data
test_data = Random_matrix[0]
predicted_data = Random_matrix[1]

# Calculate the root mean squared error between test and predicted data
root_mean_squared_error = mean_squared_error(test_data, predicted_data, squared=False)
print("\nRoot Mean Squared Error:", root_mean_squared_error)

# Q3_Assignment_01_2019E033

# Create a binary matrix of 2x10
binary_matrix = np.random.randint(0, 2, size=(2, 10))
print("\n Binary Matrix")
print(binary_matrix)

# Split the binary matrix into test and predicted data
test_data = binary_matrix[0]
predicted_data = binary_matrix[1]

# Find the accuracy, precision, recall, and F1-score of the model
accuracy = accuracy_score(test_data, predicted_data)
precision = precision_score(test_data, predicted_data)
recall = recall_score(test_data, predicted_data)
f1 = f1_score(test_data, predicted_data)

print("* Accuracy is ", accuracy)
print("* Precision is ", precision)
print("* Recall is ", recall)
print("* F1-score is ", f1)

# Print the classification report
print("\n")
print(classification_report(test_data, predicted_data))

# Plot the confusion matrix between test and predicted data
conf_matrix = confusion_matrix(test_data, predicted_data)
plt.matshow(conf_matrix, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

# Draw the ROC curve for this model
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_data, predicted_data)
roc_auc = roc_auc_score(test_data, predicted_data)
plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Q4_Assignment_01_2019E033

# Divide the matrix into train and test sets
train_data, test_data = train_test_split(Random_matrix.T, test_size=0.2, random_state=42)

# print the shapes of train and test data
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)