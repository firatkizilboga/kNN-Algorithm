import numpy as np
import math
def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

class kNN():
    def __init__(self):
        pass
    def fit(self, x, y):
        self.x=x
        self.y=y
    def predict(self, k, x):
        predictions=[]
        for i in x:
            distances=[]
            #applies pythagorous theorem and adds the distances in a list
            sub=abs(self.x-i)
            for s in sub:
                pythagorous=0
                for num in s:
                    pythagorous+=num**2
                pythagorous=math.sqrt(pythagorous)
                distances.append(pythagorous)

            #finds the closest k data points to the current data point
            outcomes=[]
            for times in range(k):
                idx=distances.index(min(distances))
                distances.pop(idx)
                outcomes.append(self.y[idx])
            
            #unique labels in self.y
            u=unique(self.y)

            #finds the most frequent label in the closest k data points and adds it to a list
            counter=[]
            for un in u:
                counter.append(outcomes.count(un))
            idx=counter.index(max(counter))
            predictions.append(u[idx])
        return predictions