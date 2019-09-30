from sklearn.naive_bayes import GaussianNB

f = [[120, 0], [150, 0], [80, 1], [100, 1]]         # features
l = ['apple', 'apple', 'orange', 'orange']         # labels

clf = GaussianNB()
trained = clf.fit(f, l)
res = trained.predict([[120,  0], [100, 1], [180, 0], [120, 1]])
print(res)
