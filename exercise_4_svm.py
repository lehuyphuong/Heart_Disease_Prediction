from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

df = pd.read_csv('cleveland.csv', header=None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

classifier = SVC(
    kernel='rbf',
    random_state=42,
    C=0.6,
    gamma='scale',
)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

accuracy_for_train = np.round(
    (cm_train[0][0] + cm_train[1][1])/len(y_train), 2)
print(f"accuracy of train would be {accuracy_for_train}")

accuracy_for_test = np.round((cm_test[0][0] + cm_test[1][1])/len(y_test), 2)
print(f"accuracy of test would be {accuracy_for_test}")

# Question 2: B 0.66, 0.67
