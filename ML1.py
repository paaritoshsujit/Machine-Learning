from sklearn.tree import DecisionTreeClassifier

f = [[120, 0], [150, 0], [80, 1], [100, 1]]         # features
l = ['apple', 'apple', 'orange', 'orange']         # labels

clf = DecisionTreeClassifier()
trained = clf.fit(f, l)
res = trained.predict([[120,  0], [100, 1], [180, 0], [120, 1],[130,0],[140,1]])
print(res)
