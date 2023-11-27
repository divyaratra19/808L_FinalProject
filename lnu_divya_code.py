import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Loading the dataset
df = pd.read_csv('StressLevelDataset.csv')
y = df['stress_level']
x = df[['anxiety_level', 'self_esteem', 'mental_health_history',
       'depression', 'headache', 'blood_pressure', 'sleep_quality',
       'breathing_problem', 'noise_level', 'living_conditions', 'safety',
       'basic_needs', 'academic_performance', 'study_load',
       'teacher_student_relationship', 'future_career_concerns',
       'social_support', 'peer_pressure', 'extracurricular_activities',
       'bullying']]

# Splitting the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

# Creating and training the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)

# Creating and training the Decision Tree model
dt_model = DecisionTreeClassifier()

#finding best value for k
print("KNN")
k_values = []
accuracy_scores = []
k_range = range(1, 11)

for k in k_range:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(x_train, y_train)
    y_pred = knn_classifier.predict(x_test)
    accuracy = knn_classifier.score(x_test,y_test)
    k_values.append(k)
    accuracy_scores.append(accuracy)

plt.plot(k_values, accuracy_scores, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs. k Value for k-NN')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.grid()
plt.show()

# Creating and training the KNN model
knn_model = KNeighborsClassifier(n_neighbors=2)

# Creating and training the NB model
nb_model = GaussianNB()

# Creating and training the LDA model
lda_model = LinearDiscriminantAnalysis()

# Fit the models
lr_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)
knn_model.fit(x_train, y_train)
nb_model.fit(x_train, y_train)
lda_model.fit(x_train, y_train)

# n-fold cross validation
models = [('Logistic Regression', lr_model),
          ('Decision Tree', dt_model),
          ('KNN', knn_model),
          ('Naive Bayes', nb_model),
          ('LDA', lda_model)]

# Dictionary to hold the cross-validation results for different values of n
n_fold_results = {}

# Values of n to try
n_values = [3, 5, 10, 15, 20]

# Initialize arrays for each model
logreg_scores = []
dt_scores = []
knn_scores = []
nb_scores = []
lda_scores = []

# Perform cross-validation for different values of n
for n in n_values:
    model_scores = {}
    for name, model in models:
        scores = cross_val_score(model, x_train, y_train, cv=n)
        model_scores[name] = scores.mean()
        # Append the mean score to the respective model's array
        if name == 'Logistic Regression':
            logreg_scores.append(scores.mean())
        elif name == 'Decision Tree':
            dt_scores.append(scores.mean())
        elif name == 'KNN':
            knn_scores.append(scores.mean())
        elif name == 'Naive Bayes':
            nb_scores.append(scores.mean())
        elif name == 'LDA':
            lda_scores.append(scores.mean())
    n_fold_results[n] = model_scores

# Names of the models for printing
model_names = ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'LDA']

# Iterate over each value of n and print the scores from each array
for i, n in enumerate(n_values):
    print(f'\nn = {n}')
    print(f'{model_names[0]}: {logreg_scores[i]}')
    print(f'{model_names[1]}: {dt_scores[i]}')
    print(f'{model_names[2]}: {knn_scores[i]}')
    print(f'{model_names[3]}: {nb_scores[i]}')
    print(f'{model_names[4]}: {lda_scores[i]}')
# Plot the accuracies for different values of n
plt.figure(figsize=(10, 6))
plt.plot(n_values, logreg_scores, marker='o', label='Logistic Regression')
plt.plot(n_values, dt_scores, marker='o', label='Decision Tree')
plt.plot(n_values, knn_scores, marker='o', label='KNN')
plt.plot(n_values, nb_scores, marker='o', label='Naive Bayes')
plt.plot(n_values, lda_scores, marker='o', label='LDA')
plt.xlabel('Number of Folds (n)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
