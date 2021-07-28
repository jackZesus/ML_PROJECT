import numpy
import random
import seaborn as sns
import matplotlib.pyplot as plt


def studentReg(ages_train,net_worth_train):
    from sklearn.linear_model import LinearRegression 
    reg = LinearRegression().fit(ages_train , net_worth_train)
    return reg

numpy.random.seed(42)
ages =[]

for ii in range(250):
    ages.append(random.randint(18,75))
    
net_worth = [ii *6.25+numpy.random.normal(scale=40) for ii in ages]

ages = numpy.reshape(numpy.array(ages),(len(ages),1))
net_worth = numpy.reshape(numpy.array(net_worth),(len(net_worth),1))


from sklearn.model_selection import train_test_split

ages_train ,ages_test ,net_worth_train,net_worth_test = train_test_split(ages,net_worth)


reg1 = studentReg(ages_train,net_worth_train)
print("coefficient",reg1.coef_)
print("intercept",reg1.intercept_)

print("training data",reg1.score(ages_train,net_worth_train))
print("test data",reg1.score(ages_test,net_worth_test))

'''VISUAL'''

        
plt.figure(figsize=(5,5))
sns.regplot(x=ages_train,y=net_worth_train, scatter = True,color ="b",marker=".")
plt.xlabel("ages train")
plt.ylabel("networth")
plt.title("regression plot")


plt.figure(figsize=(5,5))
plt.scatter(ages_train,net_worth_train,color ="b",label="train data")
plt.scatter(ages_test,net_worth_test,color ="r",label="test data")
plt.plot(ages_test,reg1.predict(ages_test))
plt.xlabel=('ages')
plt.ylabel=('net_worth')
plt.legend(loc=2)
plt.show()
