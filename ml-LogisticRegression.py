from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


feature_cols = ['Pclass', 'Parch']

file = "C:\\Users\\thecasual\\Downloads\\data.csv"
train = pd.read_csv(file)
X = train.loc[:, feature_cols]
y = train.Survived

logreg = LogisticRegression()
logreg.fit(X, y)
file2 = "C:\\Users\\thecasual\\Downloads\\data_train.csv"
test = pd.read_csv(file2)
X_new = test.loc[:, feature_cols]
new_pred_class = logreg.predict(X_new)

kaggle_data = pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':new_pred_class}).set_index('PassengerId')
correct = 0
for i in range(kaggle_data.size):
    if kaggle_data.iloc[i]['Survived'] == train.iloc[i]['Survived']:
      print("success")
      correct +=1
    else:
        print("wrong")
print("Score : {}%".format(correct / kaggle_data.size ))