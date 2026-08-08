# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Steps to do:

Step 1: Import Required Libraries

Step 2: Load the Dataset

Step 3: Copy Data & Drop Unwanted Columns

Step 4: Check Data Quality

Step 5: Encode Categorical Variables

Step 6: Define Features (X) and Target (y)

Step 7: Split into Training and Testing Sets

Step 8: Build and Train Logistic Regression Model

Step 9: Make Predictions

Step 10: Evaluate the Model

Step 11: Predict for a New Student




## Program:
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv("Placement_Data.csv")

# View first 5 rows
print("First 5 rows of the dataset:")
print(data.head())
data1 = data.copy()

# Dropping 'sl_no' (serial number) and 'salary' (not needed for predicting placement)
data1 = data1.drop(["sl_no", "salary"], axis=1)

print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())
print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())

print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())

cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]

le = LabelEncoder()

for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])

print("\nData after Label Encoding:")
print(data1.head())
X = data1.iloc[:, :-1]
# y = 'status' column
y = data1["status"]

print("\nFeatures (X) sample:")
print(X.head())

print("\nTarget (y) sample:")
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
lr = LogisticRegression(solver="liblinear")

# Train the model
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("\nPredicted values (y_pred):")
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Classification Report: precision, recall, F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]

new_prediction = lr.predict(new_student)

print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: KARTHIK J
RegisterNumber:  212225040176
*/
```

## Output:

<img width="727" height="317" alt="Screenshot 2026-08-08 102401" src="https://github.com/user-attachments/assets/fa1f3b1c-19bf-45eb-bee3-edc0134655a2" />
<img width="735" height="317" alt="Screenshot 2026-08-08 102416" src="https://github.com/user-attachments/assets/b9beb735-1a7a-422e-8878-c994de511442" />
<img width="497" height="332" alt="Screenshot 2026-08-08 102424" src="https://github.com/user-attachments/assets/d13df7a7-7803-4fc5-aa69-be1901388186" />
<img width="775" height="387" alt="Screenshot 2026-08-08 102435" src="https://github.com/user-attachments/assets/79c119d1-ceaa-4162-aebc-d1e9eb897a38" />
<img width="770" height="318" alt="Screenshot 2026-08-08 102444" src="https://github.com/user-attachments/assets/783cb7b1-9a98-43f0-8986-bb27607afb3f" />
<img width="811" height="382" alt="Screenshot 2026-08-08 102454" src="https://github.com/user-attachments/assets/ec96fb4c-8d19-461c-938d-8cbc8864185b" />
<img width="1230" height="412" alt="Screenshot 2026-08-08 102513" src="https://github.com/user-attachments/assets/942d4ca9-2c7f-4ffc-8441-77b6e72a7e6c" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
