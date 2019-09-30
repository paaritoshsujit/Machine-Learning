from sklearn.linear_model import LinearRegression

f = [[0], [2], [5], [8], [10]]
l = [2000, 5000, 18000, 50000, 55000]

clr = LinearRegression()
trained = clr.fit(f,l)
res = trained.predict([[15], [2]])
print(res)
