import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

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

# User can select which algorithm to use
model = st.selectbox("Select the model", ["SVM", "Logistic Regression", "KNN", "Gradient Boosting", "Decision Tree", "ANN", "Naive Bayes"])

if model == "SVM":
    classifier = svm.SVC(kernel='linear')
elif model == "Logistic Regression":
    classifier = LogisticRegression()
elif model == "KNN":
    classifier = KNeighborsClassifier()
elif model == "Gradient Boosting":
    classifier = GradientBoostingClassifier()
elif model == "Decision Tree":
    classifier = DecisionTreeClassifier()
elif model == "ANN":
    classifier = MLPClassifier()
elif model == "Naive Bayes":
    classifier = GaussianNB()

classifier.fit(X_train, Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# Rest of the code remains the same
def ValueCount(str):
    if str == "Yes":
        return 1
    else:
        return 0
def Sex(str):
    if str == "Female":
        return 1
    else:
        return 0

d3 = ["No", "Yes"]
q1 = st.selectbox("S/he often notices small sounds when others do not  ", d3)
q1 = ValueCount(q1)
q2 = st.selectbox("S/he usually concentrates more on the whole picture, rather than the small details ", d3)
q2 = ValueCount(q2)
q3 = st.selectbox("In a social group, s/he can easily keep track of several different people’s conversations  ", d3)
q3 = ValueCount(q3)
q4 = st.selectbox("S/he finds it easy to go back and forth between different activities  ", d3)
q4 = ValueCount(q4)
q5 = st.selectbox("S/he doesn’t know how to keep a conversation going with his/her peers  ", d3)
q5 = ValueCount(q5)
q6 = st.selectbox("S/he is good at social chit-chat  ", d3)
q6 = ValueCount(q6)
q7 = st.selectbox("When s/he is read a story, s/he finds it difficult to work out the character’s intentions or feelings  ", d3)
q7 = ValueCount(q7)
q8 = st.selectbox("When s/he was in preschool, s/he used to enjoy playing games involving pretending with other children  ", d3)
q8 = ValueCount(q8)
q9 = st.selectbox("S/he finds it easy to work out what someone is thinking or feeling just by looking at their face   ", d3)
q9 = ValueCount(q9)
q10 = st.selectbox("S/he finds it hard to make new friends  ", d3)
q10 = ValueCount(q10)
val1 = q1+q2+q3+q4+q5+q6+q7+q8+q9+q10
d2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
val2 = st.selectbox("Age  ", d2)
val3 = st.selectbox("Speech Delay  ", d3)
val3 = ValueCount(val3)
val4 = st.selectbox("Learning disorder  ", d3)
val4 = ValueCount(val4)
val5 = st.selectbox("Genetic disorders  ", d3)
val5 = ValueCount(val5)
val6 = st.selectbox("Depression  ", d3)
val6 = ValueCount(val6)
val7 = st.selectbox("Intellectual disability  ", d3)
val7 = ValueCount(val7)
val8 = st.selectbox("Social/Behavioural issues  ", d3)
val8 = ValueCount(val8)
val9 = st.selectbox("Anxiety disorder  ", d3)
val9 = ValueCount(val9)
d4 = ["Female", "Male"]
val10 = st.selectbox("Gender  ", d4)
val10 = Sex(val10)
val11 = st.selectbox("Suffers from Jaundice ", d3)
val11 = ValueCount(val11)
val12 = st.selectbox("Family member history with ASD  ", d3)
val12 = ValueCount(val12)
input_data = [val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12]
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)
with st.expander("Analyze provided data"):
    st.subheader("Results:")

    if prediction[0] == 0:
        st.info('The person is not with Autism spectrum disorder')
    else:
        st.warning('The person is with Autism spectrum disorder')

if st.button('Compare Models'):
    st.write("---")
    st.header("Model Comparison")

    # Train and evaluate each model
    models = [svm.SVC(kernel='linear'), LogisticRegression(), KNeighborsClassifier(), 
              GradientBoostingClassifier(), DecisionTreeClassifier(), MLPClassifier(), GaussianNB()]
    model_names = ["SVM", "Logistic Regression", "KNN", "Gradient Boosting", "Decision Tree", "ANN", "Naive Bayes"]
    for i, model in enumerate(models):
        model.fit(X_train, Y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_acc = accuracy_score(train_pred, Y_train)
