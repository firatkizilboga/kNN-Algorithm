import pandas as pd 
import numpy as np
from kNNA import kNN
from sklearn.model_selection import train_test_split


data=pd.read_csv("Iris.csv")
data.pop("Id")
y=data.pop("Species")

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test= np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)
algorithm=kNN()
algorithm.fit(X_train,y_train)
predictions = algorithm.predict(5,X_test)


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))
