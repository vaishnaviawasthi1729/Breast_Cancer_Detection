# Breast_Cancer_Detection
Breast_Cancer_Detection using Linear Regression and Neural Networks Regression


This project aims to detect breast cancer using two different machine learning approaches: Logistic Regression and Neural Networks. The dataset used is pre-processed and fed into these models to predict the presence of cancer. This README file details the setup, usage, and results for both models.


**Source**:<br> 
The dataset used in this project is the Breast Cancer Dataset from Kaggle.


**Features**:<br> The dataset contains 30 features (all numerical values), which are used to predict the diagnosis of breast cancer.
Target Variable: The target variable indicates whether the breast cancer is benign (0) or malignant (1).


**Installation**
To run this project, ensure you have Python installed. You can install the required dependencies by running:
pip install numpy pandas scikit-learn keras matplotlib


#**Logistic Regression Model**<br>
<br>
**Data Preprocessing**<br>
The dataset is loaded using pandas, and then split into features (X) and the target variable (y). The data is further split into training and test sets using an 80-20 split.
<br>
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
<br>


**Model Training**<br>
<br>
The LogisticRegression model from sklearn is used for binary classification. The model is trained on the pre-processed and scaled training data.
<br>
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


**Model Evaluation**<br>
The model's performance is evaluated using a confusion matrix, accuracy score, and cross-validation.
<br>
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
<br>


**Neural Network Model**<br>
<br>
**Data Preprocessing**<br>
<br>
Similar to the Logistic Regression model, the data is loaded and split into training and test sets. The features are scaled using StandardScaler.
<br>
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = read_csv("breast_cancer.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

scaler = StandardScaler()
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)<br>
<br>


**Model Architecture**<br>
<br>
A neural network is built using the Keras Sequential API. The model consists of three dense layers.
<br>

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))<br>
<br>


**Model Training**<br>
<br>
The model is trained for 100 epochs with a validation split of 20%.
<br>
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, verbose=0)

<br>
**Model Evaluation**<br>
<br>
The model's performance is evaluated using accuracy, confusion matrix, and custom rounding for predictions.
<br>
from sklearn.metrics import confusion_matrix, accuracy_score
rounded_predictions = np.array([custom_round(pred) for pred in predictions])
conf_matrix = confusion_matrix(y_test, rounded_predictions)
accuracy = accuracy_score(y_test, rounded_predictions)*100

<br>
**Results**<br>
<br>
**Logistic Regression**: The model performed well on the test set, providing good accuracy and precision.
**Neural Network: **The neural network model achieved comparable results, showing the power of deep learning even in relatively small datasets.
<br>


**Contributing**<br>
<br>
Contributions to this project are welcome. Feel free to submit a pull request or raise an issue.


