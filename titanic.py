# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


train_df = pd.read_csv('train_2.csv')
test_df = pd.read_csv('test_2.csv')

temp_df = pd.DataFrame()
temp_df['PassengerId'] = test_df['PassengerId']

train_df = train_df.drop(['Ticket', 'Cabin','Name','PassengerId','Embarked'], axis=1).fillna(-99999)
test_df = test_df.drop(['Ticket', 'Cabin','Name','PassengerId','Embarked'], axis=1).fillna(-99999)

train_df['Sex'] = train_df['Sex'].replace('male',1)
train_df['Sex'] = train_df['Sex'].replace('female',0)
test_df['Sex'] = test_df['Sex'].replace('male',1)
test_df['Sex'] = test_df['Sex'].replace('female',0)

x_train = np.array(train_df.drop(['Survived'], 1))
y_train = np.array(train_df['Survived'])

x_test = np.array(test_df)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
accuracy = round(logreg.score(x_train, y_train) * 100, 2)

submission_df = pd.DataFrame(np.reshape(y_pred, (len(y_pred,))),columns=['Survived'])
submission_df['PassengerId'] = temp_df['PassengerId']

submission_df.to_csv('titanic.csv', sep='\t', encoding='utf-8')
print submission_df