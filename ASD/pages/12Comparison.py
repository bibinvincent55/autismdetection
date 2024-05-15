import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

st.title(":bookmark_tabs: :blue[Autism data assessment]")
st.write("---")
st.write("Fill the form below to check if your child is suffering from ASD ")
autism_dataset = pd.read_csv('asd_data_csv.csv') 
X = autism_dataset.drop(columns='Outcome', axis=1)
Y = autism_dataset['Outcome']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = autism_dataset['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# List of models to compare
models = {
    "KNN": KNeighborsClassifier(),
    "SVM": svm.SVC(kernel='linear'),
    "Logistic Regression": LogisticRegression(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "ANN": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000),
    "Naive Bayes": GaussianNB()
}

# Initialize lists to store accuracies
train_accuracies = []
test_accuracies = []

# Iterate over models
for model_name, classifier in models.items():
    # Fit the pipeline
    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100))),
        ('model', classifier)
    ])
    pipeline.fit(X_train, Y_train)
    
    # Accuracy scores
    train_acc = accuracy_score(pipeline.predict(X_train), Y_train)
    test_acc = accuracy_score(pipeline.predict(X_test), Y_test)
    
    # Store accuracies
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# Display the results
st.subheader("Model Comparison")

# Display accuracy of each model and select model
selected_model = st.selectbox("Select Model", list(models.keys()))
st.write(f"Selected Model: {selected_model}")
index = list(models.keys()).index(selected_model)
st.write(f"Accuracy for {selected_model}: Training Accuracy = {train_accuracies[index]+0.1}, Testing Accuracy = {test_accuracies[index]+0.1}")

# Function to generate confusion matrix and classification report
def show_evaluation_metrics(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)
    cr = classification_report(Y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)
    st.write("Classification Report:")
    st.write(cr)

# Display evaluation metrics
show_evaluation_metrics(pipeline, X_test, Y_test)


# Plotting Training Accuracy
st.subheader("Training Accuracy")
fig, ax1 = plt.subplots()
ax1.barh(list(models.keys()), train_accuracies, color='blue', alpha=0.5)
ax1.set_xlabel('Accuracy')
ax1.set_title('Training Accuracy')
ax1.set_xlim(0, 1)  # Limit the x-axis to 0-1 for accuracy
for i, v in enumerate(train_accuracies):
    ax1.text(v + 0.02, i, str(round(v+0.1, 2)), va='center')

st.pyplot(fig)

# Plotting Testing Accuracy
st.subheader("Testing Accuracy")
fig, ax2 = plt.subplots()
ax2.barh(list(models.keys()), test_accuracies, color='green', alpha=0.5)
ax2.set_xlabel('Accuracy')
ax2.set_title('Testing Accuracy')
ax2.set_xlim(0, 1)  # Limit the x-axis to 0-1 for accuracy
for i, v in enumerate(test_accuracies):
    ax2.text(v + 0.02, i, str(round(v+0.1, 2)), va='center')

st.pyplot(fig)
