import numpy as np
import sys

"""This script implements a two-class perceptron model.
"""

class perceptron(object):
	
	def __init__(self, max_iter):
		self.max_iter = max_iter

	def fit(self, X, y):
		"""Train perceptron model on data (X,y).

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE
                import random

                self.W = np.zeros(X.shape[1],)
                print self.W.shape
                for _ in range(self.max_iter): 
                    misclassifieds = []
                    preds = self.predict(X)
                    for k in range(len(X)):
                        row = X[k]
                        prediction = preds[k]
                        error = y[k] - prediction
                        if error != 0:
                            misclassifieds.append([k,y[k]])

                    #randomly choose a misclassified example
                    idx = random.randint(0,len(misclassifieds)-1)
                    y_temp, x_temp = misclassifieds[idx][1], X[misclassifieds[idx][0]]
                    for i in range(len(self.W)):
                        self.W[i] = self.W[i] + (y_temp * x_temp[i])
		### END YOUR CODE
		
		return self


	def get_params(self):
		"""Get parameters for this perceptron model.

		Returns:
			W: An array of shape [n_features,].
		"""
		if self.W is None:
			print("Run fit first!")
			sys.exit(-1)
		return self.W

	def predict(self, X):
		"""Predict class labels for samples in X.

		Args:
			X: An array of shape [n_samples, n_features].

		Returns:
			preds: An array of shape [n_samples,]. Only contains 1 or -1.
		"""
		### YOUR CODE HERE
                def predict_helper(x):
                    pred = 0
                    for i in range(len(x)):
                        pred = pred + x[i]*self.W[i]
                    if pred >=0: return 1
                    return -1
                preds = []
                for row in X:
                    preds.append(predict_helper(row))
                return preds


		### END YOUR CODE

	def score(self, X, y):
		"""Returns the mean accuracy on the given test data and labels.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			score: An float. Mean accuracy of self.predict(X) wrt. y.
		"""
		### YOUR CODE HERE
                preds = self.predict(X)
                score = 0
                for a,b in zip(preds,y):
                    if int(a)==int(b):
                        score+=1
                return score/float(len(y))

		### END YOUR CODE


