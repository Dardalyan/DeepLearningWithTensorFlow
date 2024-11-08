# Few types of classification problems
# 1) Binary Classification
# 2) Multiclass Classification
# 3) Mutilabel Classification


# https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.75134&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false


"""
CREATE MODEL:
    Add layers for your problem. Arrange the number of units and if you need set the activation function.

    ! searched for "sigmoid vs tanh vs relu vs softmax"
    ** activation functions : https://medium.com/@cmukesh8688/activation-functions-sigmoid-tanh-relu-leaky-relu-softmax-50d3778dcea5

COMPILE MODEL:
    *loss:  how wrong your model's predictions are compared to the truth labels (you want to minimise this).
    *optimizer:  how your model should update its internal patterns to better its predictions.
    *metrics:  human interpretable values for how well your model is doing.

FIT MODEL:
    *epochs: how many times the model will go through all of the training samples.
"""


import keras
import numpy as np
from matplotlib import pyplot as plt


def plot_desicion_boundary(model:keras.models.Model,x,y):
    """
    Plots the axis boundaries of the plot and create a meshgrid.
    1.https://cs231n.github.io/neural-networks-case-study/
    2.https://github.com/madewithml/basics/blob/master/notebooks/09_Multilayer_Perceptrons/09_TF_Multilayer_Perceptrons.ipynb

    """

    # Define the axis boundaries of the plot and create a meshgrid
    x_min,x_max = x[:,0].min()-0.1,x[:,0].max()+0.1
    y_min,y_max = x[:,1].min()-0.1,x[:,1].max()+0.1

    xx,yy = np.meshgrid(np.linspace(x_min,x_max,100),np.linspace(y_min,y_max,100))

    # Create x value (we're going to make predictions on these)
    x_in = np.c_[xx.ravel(),yy.ravel()] # Stack 2D arrays together

    # make predictions
    y_pred = model.predict(x_in)

    #Check for multi-class
    if len(y_pred[0]) > 1:
        print("doing multicalss classification")
        # We have to reshape our prediction to get them ready for plotting
        y_pred = np.argmax(y_pred,axis=1).reshape(xx.shape)
    else:
        print("doing binary classification")
        y_pred = np.round(y_pred).reshape(xx.shape)

    # Plot the desicion boundary
    plt.contourf(xx,yy,y_pred,cmap=plt.cm.RdYlBu,alpha=0.7)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())



"""
    Note : tp => true positive | tn => true negative | fp => false positive  | fn => false negative 

    Classification evaluation methods
    
    METRIC NAME              METRIC FORMULA                     CODE                                     WHEN TO USE
    
    accuracy               (tp+tn)/(tp+tn+fp+fn)        keras.metrics.Accuracy()            Default metric for classification problems.
                                                    sklearn.metrics.accuracy_score()        Note the best for imbalanced classes.
    
    precision                   tp/(tp+fp)              keras.metrics.Precision()                
                                                    sklearn.metrics.precision_score()       Higher precisions leads to less false positives.         
    
    recall                      tp/(tp+fn)              keras.metrics.Recall()              Higher recall leads to less false positives.
                                                    sklearn.metrics.recall_score()          
    
    F1-score     2 * (precision * recall)/(precision + recall)   sklearn.metrics.f1_score()     Combination of precisions and recall, usually a good
                                                                                                overall metric for a classification model.
    
    Confusion Matrix           NA                  custom function  OR                      When comparing predictions to truth labels to see where
                                                    sklearn.metrics.confusion_matrix()      model gets confused. Can be hard to use with large number
                                                                                            of classes.

"""







