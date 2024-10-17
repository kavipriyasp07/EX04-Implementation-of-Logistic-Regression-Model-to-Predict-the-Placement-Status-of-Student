# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student
## DATE:

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and perform any necessary preprocessing, such as handling missing values
   and encoding categorical variables.
2. Initialize the logistic regression model and train it using the training data.
3. Use the trained model to predict the placement status for the test set.
4. Evaluate the model using accuracy and confusion matrix.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:kavipriya S.P 
RegisterNumber:2305002011  
*/
import pandas as pd
data = pd.read_csv("/content/ex45Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1['gender']=le.fit_transform(data1['gender'])
data1['ssc_b']=le.fit_transform(data1['ssc_b'])
data1['hsc_b']=le.fit_transform(data1['hsc_b'])
data1['hsc_s']=le.fit_transform(data1['hsc_s'])
data1['degree_t']=le.fit_transform(data1['degree_t'])
data1['workex']=le.fit_transform(data1['workex'])
data1['specialisation']=le.fit_transform(data1['specialisation'])
data1['status']=le.fit_transform(data1['status'])
data1
x=data1.iloc[:,:-1]
x
y=data1.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred,x_test
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
classification=classification_report(y_test,y_pred)
print("Accuracy score:\n",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification report:\n",classification)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
```

## Output:
![Screenshot 2024-10-17 103432](https://github.com/user-attachments/assets/b9de3eae-c47c-43c7-a484-89de08ae8869)
![Screenshot 2024-10-17 103440](https://github.com/user-attachments/assets/0117f445-b8d9-4195-9baf-63dd76a5d129)
![Screenshot 2024-10-17 103452](https://github.com/user-attachments/assets/210ff18c-40b2-43a5-8b66-ec1d4916b4ab)
![Screenshot 2024-10-17 103505](https://github.com/user-attachments/assets/a26549fd-9889-45e9-8858-da4a5344f188)
![Screenshot 2024-10-17 103518](https://github.com/user-attachments/assets/303e36bf-294f-4e84-b2e2-60752bef48f7)
![Screenshot 2024-10-17 103525](https://github.com/user-attachments/assets/b18a96e5-9627-46fa-9a7c-de344aaa57ee)
![Screenshot 2024-10-17 103535](https://github.com/user-attachments/assets/72672f6c-4843-4ac5-b613-4066c1f6b15f)
![Screenshot 2024-10-17 103643](https://github.com/user-attachments/assets/0271028a-be54-4957-b59e-ceff48591ccf)
![Screenshot 2024-10-17 103655](https://github.com/user-attachments/assets/a336ed2f-385c-4810-a2b0-982dcc238a71)

![Screenshot 2024-10-17 103708](https://github.com/user-attachments/assets/2b56c114-dfaf-43e8-8d37-fcf465c3dba5)
![Screenshot 2024-10-17 103733](https://github.com/user-attachments/assets/d7f2234d-30cf-4959-b130-5af569f6faf8)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
