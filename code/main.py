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
        '''
        plt.scatter(X[:,0], X[:,1], c=y)
        plt.xlabel("Symmetry")
        plt.ylabel("Intensity")
        plt.show()
        '''

        x=X[:,0]
        classes = y
        y=X[:,1]
        unique = list(set(classes))
        colors = ['b', 'y']
        for i, u in enumerate(unique):
            xi = [x[j] for j  in range(len(x)) if classes[j] == u]
            yi = [y[j] for j  in range(len(x)) if classes[j] == u]
            plt.scatter(xi, yi, c=colors[i], label=str(u))
        plt.legend()

        plt.show()





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
        x=X[:,0]
        classes = y
        y=X[:,1]
        unique = list(set(classes))
        #colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
        colors = ['b', 'y']
        for i, u in enumerate(unique):
            xi = [x[j] for j  in range(len(x)) if classes[j] == u]
            yi = [y[j] for j  in range(len(x)) if classes[j] == u]
            plt.scatter(xi, yi, c=colors[i], label=str(u))
        plt.legend()

        x1_min, x1_max = X[:,0].min(), X[:,0].max()
        x2_min, x2_max = X[:,1].min(), X[:,1].max()
	plt.plot([x1_max,x1_min], [-(W[0]/W[2]), ((-W[0] + W[1])/W[2])], color='red', linewidth=2)
        plt.xlabel("Symmetry")
        plt.ylabel("Intensity")
        plt.show()

	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
	raw_data = load_data(os.path.join(data_dir, train_filename))
	raw_train, raw_valid = train_valid_split(raw_data, 1000)
	train_X, train_y = prepare_data(raw_train)
	valid_X, valid_y = prepare_data(raw_valid)
	# Visualize training data.
	visualize_features(train_X[:, 1:3], train_y)

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

        best_model, max_accuracy = None, 0
        for model in perceptron_models:
            score_ = model.score(valid_X,valid_y)
            if(score_ > max_accuracy):
                max_accuracy = score_
                best_model = model
	visualize_result(train_X[:, 1:3], train_y, best_model.get_params())

	### END YOUR CODE
	
	# Use the 'best' model above to do testing.
	### YOUR CODE HERE

	raw_test = load_data(os.path.join(data_dir, test_filename))
        raw_test_x, raw_test_y = prepare_data(raw_test)
        print('Test accuracy (PLA):', best_model.score(raw_test_x, raw_test_y))


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

        learning_rates = [0.8,0.5,0.1, 0.01, 0.001]
        max_iters = [500,1000,2000,5000]
        batches = [1,10,20,50,100,500,1000]
        models, best_model, max_accuracy, best_batch_size = [], None,0,1
        for lr in learning_rates:
            for mi in max_iters:
                for bat_ in batches:
                    model_ = logistic_regression(learning_rate = lr, max_iter=mi)
                    model_.fit_BGD(train_X, train_y,bat_)
                    score_ = model_.score(valid_X, valid_y)
                    print "lr = ",lr , ", epochs = ",mi, ", batch_size = ",bat_,", score: ",score_
                    if(score_ > max_accuracy):
                        max_accuracy = score_
                        best_model = model_
                        best_batch_size = bat_




	### END YOUR CODE

	# Visualize the your 'best' model after training.
	# visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())
	### YOUR CODE HERE
	visualize_result(train_X[:, 1:3], train_y, best_model.get_params())

	### END YOUR CODE

	# Use the 'best' model above to do testing.
	### YOUR CODE HERE
        print("Best model parameters : Learning rate: ",best_model.learning_rate, ", Epochs: ",best_model.max_iter,", batch size: ",best_batch_size)
        print('Test accuracy (Logistic Regression):', best_model.score(raw_test_x, raw_test_y))

	### END YOUR CODE

if __name__ == '__main__':
	main()
