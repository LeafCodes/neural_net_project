import sklearn as sk
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
import numpy as np

os.chdir('/Users/ASUS/Downloads/data')
heart = pd.read_csv('data.csv', sep=',', header=0)
heart.head()

y = heart.iloc[:,9]
X = heart.iloc[:,:9]


LR = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='ovr').fit(X, y)
Pred = LR.predict(X.iloc[460:,:])
Pred_1 = round(LR.score(X,y), 4)
print(Pred_1)