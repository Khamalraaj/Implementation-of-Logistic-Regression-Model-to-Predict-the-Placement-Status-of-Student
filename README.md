# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, drop unnecessary columns, and encode categorical variables
2. Define the features (X) and target variable (y).
3. Split the data into training and testing sets.
4. Train the logistic regression model, make predictions, and evaluate using accuracy and other

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Khamalraaj S
RegisterNumber: 212224230122
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
```

## Output:
# HEAD
 <img width="1279" height="330" alt="HEAD" src="https://github.com/user-attachments/assets/aa486efe-7c05-41ca-a8fd-35b0f154aac1" />

# COPY 
<img width="1141" height="345" alt="COPY" src="https://github.com/user-attachments/assets/4a80550b-fce1-49bb-ac76-563465587eb7" />

# FIT TRANSFORM
<img width="1122" height="707" alt="FIT_TRANSFORM" src="https://github.com/user-attachments/assets/2cd85055-8aec-408f-aa95-673f740fa08a" />


# LOGISTIC REGRESSION
<img width="1231" height="309" alt="Logistic_regression" src="https://github.com/user-attachments/assets/9b32602e-00ef-4aaa-9eb1-548703b7e989" />


# ACCURACY SCORE
<img width="1225" height="169" alt="Accuracy_score" src="https://github.com/user-attachments/assets/a1152fe2-e3c3-49e8-b587-a621bb3b3ce8" />


# CONFUSION MATRIX
<img width="1229" height="203" alt="confession_matrix" src="https://github.com/user-attachments/assets/3f22c7cd-9cd1-4375-bf94-4ae5eea76bac" />


# CLASSIFICATION REPORT & PREDICTION

<img width="1217" height="389" alt="CLASSIFICATION REPORT   PREDICTION_crop" src="https://github.com/user-attachments/assets/b1d9fa43-acb2-44f1-b574-7093f4137184" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
