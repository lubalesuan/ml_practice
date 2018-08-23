#attributes: Sepal Length, Sepal Width, Petal Length and Petal Width.

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def logisticRegression(dataset):
	logRes = LogisticRegression() #init
	logRes.fit(dataset.data, dataset.target) #train
	print "Predicted class: {}".format(logRes.predict(dataset.data[-1:]))
	print "Actual class: {} \n".format(dataset.target[-1:])
	return logRes

def crossValidation (clf, dataset):
	scores = cross_val_score(clf, dataset.data, dataset.target, cv = 10)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def main():
	digits = datasets.load_digits()
	iris = datasets.load_iris()
	logResDigits = logisticRegression(digits)
	logResIris = logisticRegression(iris)
	crossValidation(logResDigits, digits)
	crossValidation(logResIris, iris)

if __name__ == "__main__":
	main()