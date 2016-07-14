

from sklearn.linear_model import LinearRegression

x_data = [[1],[2],[3],[4],[6],[13],[20],[35],[50]]
y_data = [[1],[4],[9],[16],[36],[169],[400],[1225],[2500]]


model = LinearRegression()
model.fit(x_data,y_data)
print model.predict(1)
print model.predict(5)
print model.predict(50)
print model.predict(60)

#print model.coef_
#print model.intercept_


#so no it cant , its just a straight line. 
#what about svm?

print

from sklearn.svm import SVR

y_data = [1,4,9,16,36,169,400,1225,2500]

model = SVR()
model.fit(x_data,y_data)
print model.predict(1)
print model.predict(5)
print model.predict(50)
print model.predict(60)

#no its worse?

print

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(x_data,y_data)
print model.predict(1)
print model.predict(5)
print model.predict(50)
print model.predict(60)


#just picks the example that is closest