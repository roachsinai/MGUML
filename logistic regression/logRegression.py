##############################
# From : blog.csdn.net/zouxy09
# ############################

from numpy import *
import matplotlib.pyplot as plt
import time

# calculate the sigmoid function
def sigmoid(inX):
	return 1.0 / (1 + exp(-inX))

# train a logistic regression model using some optional optimize algorithm
# input: train_x is a mat datatype, each row stands for one sample
#        train_y is a mat datatype, each row is the corresponding label
#        opts is optimize option include step and maximum number of iterations
def trainLogRegress(train_x, train_y, opts):
	# calculate training time
	startTime = time.time()

	numSamples, numFeatures = shape(train_x)
	alpha = opts['alpha']; maxIter = opts['maxIter']
	weights = ones((numFeatures, 1))
	# print type(weights)												

	# optimize through gradient descent algorithm
	for k in range(maxIter):
		if opts['optimizeType'] == 'gradDescent': # gradient descent algorithm
			output = sigmoid(train_x * weights)
			error = train_y - output
			weights += alpha * train_x.transpose() * error
		elif opts['optimizeType'] == 'stocGradDescent': # stochastic gradient descent
			for i in range(numSamples):
				output = sigmoid(train_x[i, :] * weights)
				error = train_y[i, 0] - output
				weights += alpha * train_x[i, :] * error
		elif opts['optimizeType'] == 'smoothStocGradDescent': # smooth stochastic gradient descent
			# randomly select samples to optimize for reducing cycle fluctuations
			dataIndex = range(numSamples)
			for i in range(numSamples):
				alpha = 4.0 / (1.0 + k + i) + 0.01
				randIndex = int(random.uniform(0, len(dataIndex)))
				output = sigmoid(train_x[randIndex, :] * weights)
				error = train_y[randIndex, 0] - output
				weights += alpha * train_x[randIndex, :].transpose() * error
				del(dataIndex[randIndex])
		else:
			raise NameErroe('Not support optimize method type!')

	print 'Congratulations, training complate! Took %fs!' % (time.time() - startTime)
	return weights

# test your trained Logistic Regression model given test set
def testLogRegress(weights, test_x, test_y):
	numSamples, numFeatures = shape(test_x)
	matchCout = 0
	for i in xrange(numSamples):
		predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
		if predict == bool(test_y[i, 0]):
				matchCout += 1
	#accuracy = float(matchCout) / numSamples
	return float(matchCout) / numSamples

# show your traind logistic regression model only available with 2-D data
def showLogRegress(weights, train_x, train_y):
	# notice: train_x and train_y is mat datatype
	numSamples, numFeatures = shape(train_x)
	if numFeatures != 3:
		print 'Sorry! I can not darw because the dimension of your data is not 2!'
		return 1

	# draw all samples
	for i in xrange(numSamples):
		if int(train_y[i, 0]) == 0:
			plt.plot(train_x[i, 1], train_x[i, 2], 'or')
		elif int(train_y[i, 0]) == 1:
			plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

	# print numSamples
	# draw the classigy line
	min_x = min(train_x[:, 1])[0, 0]
	max_x = max(train_x[:, 1])[0, 0]
	# weights = weights# .getA() # convert mat to array
	y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
	y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
	plt.plot([min_x, max_x], [y_min_x, y_max_x])
	plt.xlabel('X1'); plt.ylabel('X2')
	plt.show()