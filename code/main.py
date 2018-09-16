import os
import matplotlib.pyplot as plt
from Perceptron import perceptron
from LogisticRegression import logistic_regression
from DataReader import *

data_dir = "../data/"
train_filename = "train.txt"
test_filename = "test.txt"

def visualize_features(X, y):
	'''This function is used to plot a 2-D scatter plot of training features. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 1 or -1.
	
	Returns:
		No return. Save the plot to 'train_features.*' and include it
		in submission.
	'''
	### YOUR CODE HERE
        plt.scatter(X[:,0], X[:,1], c=y)
        plt.show()
        plt.xlabel("Symmetry")
        plt.ylabel("Intensity")

	### END YOUR CODE

def visualize_result(X, y, W):
	'''This function is used to plot the linear model after training. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 1 or -1.
		W: An array of shape [n_samples,].
	
	Returns:
		No return. Save the plot to 'train_result.*' and include it
		in submission.
	'''
	### YOUR CODE HERE

	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
	raw_data = load_data(os.path.join(data_dir, train_filename))
	raw_train, raw_valid = train_valid_split(raw_data, 1000)
	train_X, train_y = prepare_data(raw_train)
	valid_X, valid_y = prepare_data(raw_valid)
	# Visualize training data.
	#visualize_features(train_X[:, 1:3], train_y)

	# ------------Perceptron------------
	perceptron_models = []
	for max_iter in [10, 20, 50, 100, 200]:
		# Initialize the model.
		perceptron_classifier = perceptron(max_iter=max_iter)
		perceptron_models.append(perceptron_classifier)

		# Train the model.
		perceptron_classifier.fit(train_X, train_y)
		
		print('Max interation:', max_iter)
		print('Weights after training:',perceptron_classifier.get_params())
		print('Training accuracy:', perceptron_classifier.score(train_X, train_y))
		print('Validation accuracy:', perceptron_classifier.score(valid_X, valid_y))
		print()
	# Visualize the the 'best' one of the five models above after training.
	# visualize_result(train_X[:, 1:3], train_y, best_perceptron.get_params())
	### YOUR CODE HERE

	### END YOUR CODE
	
	# Use the 'best' model above to do testing.
	### YOUR CODE HERE

	### END YOUR CODE


	# ------------Logistic Regression------------

	# Check GD, SGD, BGD
	logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

	logisticR_classifier.fit_GD(train_X, train_y)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))

	logisticR_classifier.fit_BGD(train_X, train_y, 1000)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))

	logisticR_classifier.fit_SGD(train_X, train_y)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))

	logisticR_classifier.fit_BGD(train_X, train_y, 1)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))

	logisticR_classifier.fit_BGD(train_X, train_y, 10)
	print(logisticR_classifier.get_params())
	print(logisticR_classifier.score(train_X, train_y))
	print()

	# Explore different hyper-parameters.
	### YOUR CODE HERE

	### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())
	### YOUR CODE HERE

	### END YOUR CODE

	# Use the 'best' model above to do testing.
	### YOUR CODE HERE

	### END YOUR CODE

if __name__ == '__main__':
	main()
