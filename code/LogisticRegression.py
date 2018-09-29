import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
	def __init__(self, learning_rate, max_iter):
		self.learning_rate = learning_rate
		self.max_iter = max_iter

	def fit_GD(self, X, y):
		"""Train perceptron model on data (X,y) with GD.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE
                self.fit_BGD(X,y,X.shape[0])
		### END YOUR CODE

		return self

	def fit_BGD(self, X, y, batch_size):
		"""Train perceptron model on data (X,y) with BGD.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.
			batch_size: An integer.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE
                #self.W = [0 for _ in range(len(X[0])+1)]
                self.W = np.zeros(X.shape[1],)
                no_rows = X.shape[0]
                i = 0
                while i < no_rows:
                    sum_ = np.zeros(X.shape[1],)
                    for _ in range(batch_size):
                        if i>=no_rows: break
                        sum_ += self._gradient(X[i],y[i])
                        i+=1
                    grad = sum_/batch_size
                    self.W = self.W - self.learning_rate*grad

		### END YOUR CODE

		return self

	def fit_SGD(self, X, y):
		"""Train perceptron model on data (X,y) with SGD.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE
                self.fit_BGD(X,y,1)
		### END YOUR CODE
		
		return self

	def _gradient(self, _x, _y):
		"""Compute the gradient of cross-entropy with respect to self.W
		for one training sample (_x, _y). This function is used in SGD.

		Args:
			_x: An array of shape [n_features,].
			_y: An integer. 1 or -1.

		Returns:
			_g: An array of shape [n_features,]. The gradient of
				cross-entropy with respect to self.W.
		"""
		### YOUR CODE HERE
                num = -1 * _y * _x
                z = _y * _x.dot(self.W)
                den = 1 + np.exp(z)
                _g = num/den
                return _g

		### END YOUR CODE

	def get_params(self):
		"""Get parameters for this perceptron model.

		Returns:
			W: An array of shape [n_features,].
		"""
		if self.W is None:
			print("Run fit first!")
			sys.exit(-1)
		return self.W

	def predict_proba(self, X):
		"""Predict class probabilities for samples in X.

		Args:
			X: An array of shape [n_samples, n_features].

		Returns:
			preds_proba: An array of shape [n_samples, 2].
				Only contains floats between [0,1].
		"""
		### YOUR CODE HERE
                import math
                def predict_helper(x):
                    pred = 0
                    for i in range(len(x)):
                        pred = pred + x[i]*self.W[i]
                    return pred

                preds_proba = []
                for row in X:
                    k = predict_helper(row)
                    prob = 1 / (1 + math.exp(-k))
                    preds_proba.append([prob,1-prob])
                return preds_proba
		### END YOUR CODE
        


	def predict(self, X):
		"""Predict class labels for samples in X.

		Args:
			X: An array of shape [n_samples, n_features].

		Returns:
			preds: An array of shape [n_samples,]. Only contains 1 or -1.
		"""
		### YOUR CODE HERE
                preds = []
                predict_proba = self.predict_proba(X)
                for row in predict_proba:
                    if row[0] >= 0.5:
                        preds.append(1)
                    else:
                        preds.append(-1)
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


