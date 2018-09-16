import numpy as np
import sys

"""This script implements a two-class perceptron model.
"""

class perceptron(object):
	
	def __init__(self, max_iter):
		self.max_iter = max_iter
                # added parameters
                self.W = None   
                self.learning_rate = 1

	def fit(self, X, y):
		"""Train perceptron model on data (X,y).

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE
                #self.W = [0 for _ in range(len(X[0])+1)]
                self.W = np.zeros(X.shape[0],)
                for _ in range(self.max_iter): 
                    for k in range(len(X)):
                        row = X[k]
                        prediction = self.predict_helper(row)
                        error = y[k] - prediction  # Cross check this
                        for i in range(1,len(self.W)):
                            self.W[i] = self.W[i] + (self.learning_rate * error * row[i-1])
                        self.W[0] = self.W[0] + (self.learning_rate * error)




		### END YOUR CODE
		
		return self

        def predict_helper(self, x):
            pred, bias = 0, self.W[0]
            for i in range(len(x)):
                pred = pred + x[i]*self.W[i+1]
            pred = pred + bias
            if pred >=0: return 1
            return -1

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
                preds = []
                for row in X:
                    preds.append(self.predict_helper(row))
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
                #print "preds :", preds
                #print "Y :", y
                #print "X :",X
                score = 0
                for a,b in zip(preds,y):
                    if int(a)==int(b):
                        score+=1
                return score/float(len(y))

		### END YOUR CODE


