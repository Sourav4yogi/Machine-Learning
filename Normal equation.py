import numpy as np
import matplotlib.pyplot as plt 
from numpy import linalg as la
from sklearn.linear_model import LinearRegression


X=2* np.random.rand(100,1)
y=4+3*X+np.random.rand(100,1)

Xb=np.c_[np.ones((100,1)),X]

theta=la.inv(Xb.T.dot(Xb)).dot(Xb.T).dot(y) #Normal Equation

print('Theta : \n',theta)


print('\nLets predict with this theta and input \n[[0]\n[2]]')


Xnew=np.array([[0],[2]])

Xnewb=np.c_[np.ones((2,1)),Xnew]

prediction=Xnewb.dot(theta)
print('\nPrediction : \n',prediction)




plt.figure(figsize=(12,8))
plt.plot(Xnew,prediction,'r-')
plt.plot(X,y,'b.')
plt.axis([0,2,0,15])
plt.show()



print(' \n Lets do the same with sklearn.linear_model.LinearRegression() \n')

lr=LinearRegression()
lr.fit(X,y)
print('\ntheta : \n',lr.intercept_,lr.coef_)



print('\nPrediction \n',lr.predict(Xnew))




















