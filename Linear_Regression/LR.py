

#PLaying with ordinary least squares and nromal equations

import numpy as np



#2D datapoints
X = np.array([  [1,3],
				[3,4],
				[2,2],
				])

y = np.array([  [0],
				[1],
				[2],
				])

print ('X', X.shape)
print (X)

print ('y', y.shape)
print (y)
print ()


# X^T * y 
X_T = np.transpose(X)
Xy = np.dot(X_T, y)
print ('Xy', Xy.shape)
print (Xy)


# X^T*X
XX = np.dot(X_T, X)
print ('XX', XX.shape)
print (XX)

#XX^-1
XX_inv = np.linalg.inv(XX)
print ('XX inv', XX_inv.shape)
print (XX_inv)

# W
W = np.dot(XX_inv, Xy)
print ('W', W.shape)
print (W)
print ()


# Feature means
x_mean = np.mean(X, 0, keepdims=True)
print ('Feature Means', x_mean.shape)
print (x_mean.T)


# Feature Standard Deviations
x_centered = X - x_mean
print ('Centered X', x_centered.shape)
print (x_centered)


# Centered x * y 
Xy_cent = np.dot(x_centered.T, y)
print ('Centered Xy', Xy_cent.shape)
print (Xy_cent)


# Centered XX
XX_cent = np.dot(x_centered.T, x_centered)
print ('Centered XX like a Pseudo Covariance', XX_cent.shape)
print (XX_cent )


XX_cent_div = XX_cent / 3
print ('Divided by number of datapoints, so its like avg, I think this is covariance', XX_cent_div.shape)
print (XX_cent_div )



y_centered = y - np.mean(y)
print ('Centered y', y_centered.shape)
print (y_centered)
print ()


thing = np.dot(x_centered.T, y_centered)
print ('Centered X * centered y / n_data', thing.shape)
print ('Cov(X,y) I think')
print (thing / 3)
print ('so its the dot prodcut similarity of centered dims vs centered y')
print ('the dims/features are still independent at this point')
print ('I think I should have been centering based on dims, not over data')



print ()
x_centered_2 = X.T - np.mean(X.T, 0, keepdims=True)
print ('Centered X 2', x_centered_2.T.shape)
print (x_centered_2.T)

# Centered x * y 
Xy_cent_2 = np.dot(x_centered_2, y_centered)
print ('Centered Xy 2', Xy_cent_2.shape)
print (Xy_cent_2)


dim0 = np.reshape(X.T[0], [3,1])
print (dim0, dim0.shape)
dim1 = np.reshape(X.T[0], [3,1])
corr = np.corrcoef(dim0, y)
print ('corr', corr.shape)
print (corr)




