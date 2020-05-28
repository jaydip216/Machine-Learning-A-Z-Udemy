import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('Data.csv')
X=df.iloc[:,:-1]
y=df.iloc[:,3]



from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2)

