import numpy as np
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def itcRegression(x_train,y_train):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(x_train,y_train)
    return reg


data = pd.read_csv(r'C:\Users\ROHAN\Desktop\ML PROJECT\ITC.csv')
#print(data)
data.head()

#collecting x and y 
x = data['Volume'].values
y =data['Close'].values

#mean x and y 
mean_x =np.mean(x)
mean_y =np.mean(y)

'''print(mean_x)
   print(mean_y) ''' 


#reshape of data using numpy conversion to 2D data
x = numpy.reshape(numpy.array(x),(len(x),1))
y = numpy.reshape(numpy.array(y),(len(y),1))

#importing train and test 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y)


reg1 = itcRegression(x_train,y_train)
print("coefficient",reg1.coef_)
print("intercept",reg1.intercept_)

#printing the test and train data
print("training data",reg1.score(x_train,y_train))
print("test data",reg1.score(y_test,y_test))

'''VISUALIZATION'''
plt.figure(figsize=(5,5))
sns.regplot(x=x_train,y=y_train,scatter=True,color='b',marker=".")
plt.xlabel('Volume')
plt.ylabel("CLose")
plt.title("regression PLOt")


plt.figure(figsize=(5,5))
plt.scatter(x_train,y_train,color ="b",label="train data")
plt.scatter(x_test,y_test,color ="r",label="test data")
plt.plot(x_test,reg1.predict(x_test))
plt.xlabel=('Volume')
plt.ylabel=('Close')
plt.legend(loc=2)
plt.show()
