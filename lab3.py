#!/usr/bin/python
# coding: utf-8
# flake8: noqa

# # Lab 3: Bayes Classifier and Boosting

# ## Jupyter notebooks
# 
# In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.
# 
# If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.
# 
# And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.

# ## Import the libraries
# 
# In Jupyter, select the cell below and press `ctrl + enter` to import the needed libraries.
# Check out `labfuns.py` if you are interested in the details.

import numpy as np
import scipy
from scipy import misc
from importlib import reload
from labfuns import *
import random
import math
import time


# ## Bayes classifier functions to implement
# 
# The lab descriptions state what each function should do.


# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))
    prior_w = np.zeros((Nclasses,1))

    for i in range(Nclasses):
        length = len(np.where(labels == i)[0])
        prior[i] = length/Npts
        pos = np.where(labels == i)[0]
        prior_w[i] = W[pos].sum(axis=0,dtype='float')/W.sum(axis=0,dtype='float')

    return prior_w

# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    #mu = np.zeros((Nclasses,Ndims))
    mu_w = np.zeros((Nclasses,Ndims))
    #sigma = np.zeros((Nclasses,Ndims,Ndims))
    sigma_w = np.zeros((Nclasses,Ndims,Ndims))

    for i in range(Nclasses):
        # For each class, sum all the points from that class and divide them by the number of points of that class
        pos = np.where(labels == i)[0]
        length = len(np.where(labels == i)[0])

        #mu[i] = X[pos].sum(axis=0,dtype='float')/length
        mu_w[i] = (W[pos]*X[pos]).sum(axis=0,dtype='float')/W[pos].sum(axis=0,dtype='float')
        
        #diag = (X[pos]-mu[i])**2
        #diag = diag.sum(axis=0,dtype='float')/length
        #np.fill_diagonal(sigma[i],diag)  
        diag_w = W[pos]*((X[pos]-mu_w[i])**2)
        diag_w = diag_w.sum(axis=0,dtype='float')/W[pos].sum(axis=0,dtype='float')
        np.fill_diagonal(sigma_w[i],diag_w)              

    return mu_w, sigma_w

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    # USING LOOPS:
    #start_time = time.time()
    for i in range(Npts):
        for j in range(Nclasses):
            A = (X[i]-mu[j])
            #B = np.reciprocal(sigma[j], where= (sigma[j]!=0))  #this makes 0 entries be 1.123e-303, maybe too much storage
            #B=np.linalg.inv(sigma[j])
            # B is the inverse of sigma
            B = np.zeros((Ndims,Ndims))
            for n in range(sigma[j].shape[0]):
                for m in range(sigma[j].shape[1]):
                    if n==m and sigma[j][n][m]!=0:
                        B[n][m]= 1/sigma[j][n][m]

            C = (X[i]-mu[j]).T
            logProb[j,i] = -0.5*math.log(np.linalg.det(sigma[j]))  + math.log(prior[j]) - 0.5*A.dot(B).dot(C)

    # WITHOUT LOOPS
    # start_time = time.time()
    # for j in range(Nclasses):
    #     Amat = X - mu[j]
    #     Aflat = np.reshape(Amat, (Npts*Ndims,1))
    #     B = [np.reciprocal(sigma[j], where=(sigma[j]!=0))]*Npts
    #     #B = np.kron(np.eye(Npts,dtype=float),B) #THIS IS SUPER SLOW
    #     Bdiag = scipy.linalg.block_diag(*B)
    #     C = [np.reshape(x, (1,Ndims))  for x in Amat.tolist()]
    #     D = scipy.linalg.block_diag(*C)
    #     logProb[j] = -0.5*math.log(np.linalg.det(sigma[j]))*np.ones((1,Npts)) + math.log(prior[j])*np.ones((1,Npts)) - 0.5*np.matmul(D,Bdiag.dot(Aflat)).T

    h = np.argmax(logProb,axis=0)
    #print(h)
    #print("--- %s seconds ---" % (time.time() - start_time))

    return h


# The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:


# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# ## Test the Maximum Likelihood estimates
# 
# Call `genBlobs` and `plotGaussian` to verify your estimates.
X, labels = genBlobs(centers=5)
Npts,Ndims = np.shape(X)
W = np.ones((Npts,1))/float(Npts)
mu, sigma = mlParams(X,labels, W)

#plotGaussian(X,labels,mu,sigma)

prior = computePrior(labels,W)
#classifyBayes(X, prior, mu, sigma)

# Call the `testClassifier` and `plotBoundary` functions for this part.

# testClassifier(BayesClassifier(), dataset='iris', split=0.7)
# plotBoundary(BayesClassifier(), dataset='iris',split=0.7)

#testClassifier(BayesClassifier(), dataset='vowel', split=0.7, ntrials = 100)
#plotBoundary(BayesClassifier(), dataset='vowel',split=0.7)







# ## Boosting functions to implement
# 
# The lab descriptions state what each function should do.


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # Compute the error
        error = 0#np.ones((Npts,1))
        #print(vote.shape)
        for p in range(Npts): #I can do labels==vote and convert from true/flase to 1/0
            if (labels[p]==vote[p]):
                delta = 1
            else:
                delta = 0
            error += wCur[p]*(1-delta)
        if error==0:
            error = 1.0e-100  
        alpha = 0.5*(math.log(1-error)-math.log(error))
        alphas.append(alpha) # you will need to append the new alpha

        
    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:

        return classifiers[0].classify(X)
    else:
    
        votes = np.zeros((Npts,Nclasses))

        for i in range(Ncomps):
            pred = classifiers[i].classify(X)

            for j in range(Npts):
                votes[j,pred[j]] +=1

        return np.argmax(votes,axis=1)


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
# 
# Call the `testClassifier` and `plotBoundary` functions for this part.


testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)



#testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)



#plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)


# Now repeat the steps with a decision tree classifier.


#testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)



#testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)



#plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)



#plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)


# ## Bonus: Visualize faces classified using boosted decision trees
# 
# Note that this part of the assignment is completely voluntary! First, let's check how a boosted decision tree classifier performs on the olivetti data. Note that we need to reduce the dimension a bit using PCA, as the original dimension of the image vectors is `64 x 64 = 4096` elements.


#testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare this with using pure decision trees or a boosted bayes classifier. Not too bad, now let's try and classify a face as belonging to one of 40 persons!


#X,y,pcadim = fetchDataset('olivetti') # fetch the olivetti data
#xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) # split into training and testing
#pca = decomposition.PCA(n_components=20) # use PCA to reduce the dimension to 20
#pca.fit(xTr) # use training data to fit the transform
#xTrpca = pca.transform(xTr) # apply on training data
#xTepca = pca.transform(xTe) # apply on test data
# use our pre-defined decision tree classifier together with the implemented
# boosting to classify data points in the training data
#classifier = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr)
#yPr = classifier.classify(xTepca)
# choose a test point to visualize
#testind = random.randint(0, xTe.shape[0]-1)
# visualize the test point together with the training points used to train
# the class that the test point was classified to belong to
#visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])

