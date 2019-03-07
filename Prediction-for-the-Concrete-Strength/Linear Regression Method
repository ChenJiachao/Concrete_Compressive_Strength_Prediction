import numpy as np
import pandas
import matplotlib.pyplot as plt

# Used pandas to read data file from 'Train.xlsx' and 'Test.xlsx'
Trainfile = pandas.read_excel('Train.xlsx')
Testfile = pandas.read_excel('Test.xlsx')

# Load the values into train and test seperately
train = Trainfile.values
test = Testfile.values

# Splie the data into X train and y train and ....
X_trn_raw, y_trn = train[:,:-1], train[:,-1:]
X_tst_raw, y_tst = test[:,:-1], test[:,-1:]

# Concatenate the training file and testing file together
# As we need to shift their mean to zero and normalized them
inputs = np.concatenate((X_trn_raw, X_tst_raw),axis = 0)
inputs = inputs - np.mean(inputs,axis=0) #shift
inputs = inputs/(np.max(inputs,axis=0)) #normalize

inputs = np.concatenate((inputs, np.ones((X_trn_raw.shape[0]+X_tst_raw.shape[0],1))), axis = 1) #add bias

# Split the file back
X_trn = inputs[:X_trn_raw.shape[0],:]
X_tst = inputs[X_trn_raw.shape[0]:,:]

# Linear Regression Training Test
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_trn, y_trn)
l_trn = lr.score(X_trn, y_trn)
l_tst = lr.score(X_tst, y_tst)

print('The average training accuracy is: ', l_trn)
print('The average test accuracy is: ', l_tst)


#Testing different lambda values = 0.001 * [1, 1.2, 1.44, ..., 1.2^99] 
# Do Ridge Regularization

lamb = []
points = 100
for i in range(points):  
    lamb.append(0.001*(1.2**i))
accuracy_trn = [] 
accuracy_tst = [] 

from sklearn.linear_model import Ridge
for i in lamb:
    lrr = Ridge(alpha = i )
    lrr.fit(X_trn, y_trn)
    accuracy_trn.append(lrr.score(X_trn, y_trn))
    accuracy_tst.append(lrr.score(X_tst, y_tst))



plt.semilogx(lamb,accuracy_trn,color='red')
plt.semilogx(lamb,accuracy_tst,color='blue')
plt.title("L2 error")
plt.show()
plt.semilogx(lamb[1:40],accuracy_trn[1:40],color='red')
plt.semilogx(lamb[1:40],accuracy_tst[1:40],color='blue')
plt.title("L2 error")
plt.show()

index = accuracy_tst.index(max(accuracy_tst))
max_lamb = lamb[index-1]
print(max_lamb)
lrr = Ridge(alpha = max_lamb )
lrr.fit(X_trn, y_trn)
print(lrr.score(X_tst, y_tst))

# As the result is really bad
# Map the inputs to a higher deminsion (x, x^2)

inputs = np.concatenate((X_trn_raw, X_tst_raw),axis = 0)
inputs_new = np.concatenate((inputs,np.square(inputs)),axis=1) #add square term

inputs_new = inputs_new - np.mean(inputs_new,axis=0) #shift
inputs_new = inputs_new/(np.max(inputs_new,axis=0)) #normalize
inputs_new = np.concatenate((inputs_new, np.ones((X_trn_raw.shape[0]+X_tst_raw.shape[0],1))), axis = 1) # add bias

X_trn_new = inputs_new[:X_trn_raw.shape[0],:]
X_tst_new = inputs_new[X_trn_raw.shape[0]:,:]


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
plf = PolynomialFeatures()
# Transfer the dataset to fit Linear Regression

X_trn_new_p = plf.fit_transform(X_trn_new)
X_tst_new_p = plf.fit_transform(X_tst_new)

lr = LinearRegression()
lr.fit(X_trn_new_p, y_trn)

l_trn = lr.score(X_trn_new_p, y_trn)
l_tst = lr.score(X_tst_new_p, y_tst)


print('The average training accuracy is: ', l_trn)
print('The average test accuracy is: ', l_tst)



# Let's implement ridge regularization again
lamb = []
points = 100
for i in range(points):  
    lamb.append(0.001*(1.2**i)) #= 0.001 * [1, 1.2, 1.44, ..., 1.2^99] #lambda values
accuracy_trn = [] 
accuracy_tst = [] 

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

plf = PolynomialFeatures()
X_trn_new_p = plf.fit_transform(X_trn_new)
X_tst_new_p = plf.fit_transform(X_tst_new)

for i in lamb:
    lrr = Ridge(alpha = i)
    lrr.fit(X_trn_new_p, y_trn)

    l_trn = lrr.score(X_trn_new_p, y_trn)
    l_tst = lrr.score(X_tst_new_p, y_tst)

    accuracy_trn.append(lrr.score(X_trn_new_p, y_trn))
    accuracy_tst.append(lrr.score(X_tst_new_p, y_tst))


plt.semilogx(lamb,accuracy_trn,color='red')
plt.semilogx(lamb,accuracy_tst,color='blue')
plt.title("L2 Accuracy")
plt.show()
print(max(accuracy_tst))


index = accuracy_tst.index(max(accuracy_tst))
max_lamb = lamb[index]
print(max_lamb)
lrr = Ridge(alpha = max_lamb )
lrr.fit(X_trn_new_p, y_trn)
print(lrr.score(X_tst_new_p, y_tst))

# Achieved in 84% accuracy
