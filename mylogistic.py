"""
    Logistic Regression Implementation. Based on https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/
"""
import copy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class MyLogisticRegression():
    """
        @params: epochs: numbers of iterations you want to run with gradient descent
    """
    def __init__(self, epochs = 50):
        self.epochs = epochs
        self.losses = []
        self.train_accuracies = []
        
    """
    Sigmoid Function σ = 1 / (1 + e^aTu + b)
    """
    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)    
    
    #arrayize for each x
    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])
    
    """
        Log-loss function L(a,b)
        @params: y_true : (1 x n) array of label (yi)
                 y_pred : (1 x n) array of predictions (pi)
        @return: (1 x n) array of log-loss values
    """ 
    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)
    
    """
        Compute the gradient of log-loss function with respective to a (dL(a,b) / da)
        @params: x: (m x n) array of independent variable matrix (ui matrix)
                 y_true : (1 x n) array of label (yi)
                 y_pred : (1 x n) array of predictions (pi)   
        @return: (m x 1) array of gradient values
    """
    def compute_gradients_wrt_a(self, x, y_true, y_pred):
        diff = y_pred - y_true
        grad = np.matmul(x.transpose(), diff)
        return np.array([np.mean(values) for values in grad])
    
    """
        Compute the gradient of log-loss function with respective to a (dL(a,b) / db)
        @params: y_true : (1 x n) array of label (yi)
                 y_pred : (1 x n) array of predictions (pi)
        @return: (float) gradient values of b
    """
    def compute_gradients_wrt_b(self, y_true, y_pred):
        # derivative of binary cross entropy
        diff =  y_pred - y_true
        return np.mean(diff)
    
    """
        Stepwise steepest descent. Shift our value to some directions. a_new = a - lr * gradient
        @params: error_a: (m x 1) array of gradient values
                 error_b: (float) gradient values of b
                 learning_rate: learning rate you wish to adjust
        @return: None. We update it to our original weights and bias
    """
    def update_model_parameters(self, error_a, error_b, learning_rate = 0.1):
        self.weights = self.weights - learning_rate * error_a
        self.bias = self.bias - learning_rate * error_b
        
    """
        Helper function to transform dimensions of x and y
    """    
    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x.values

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.values.reshape(y.shape[0], 1)
    
    """
        Train the Logistic Regression Model (i.e. running Steepest Descent Algorithm)
        @params: x: (m x n) array of independent variables (ui)
                 y: (m x 1) array of true predictions (label)
    """
    def train(self, x, y, learning_rate = 0.1):
        x = self._transform_x(x)
        y = self._transform_y(y)
        
        #Initialize a to zeros matrix, b to zero, which is our start case
        self.weights = np.zeros(x.shape[1])
        self.bias = 0
        
        #Our algorithm starts here
        for _ in tqdm(range(self.epochs),desc = "Training model"):
            # Compute pi
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            # Compute loss function. Store this in self.losses to see how optimized the algorithm works. 
            loss = self.compute_loss(y, pred)
            self.losses.append(loss)
            # Compute the gradient w.r.t. a and b
            error_a = self.compute_gradients_wrt_a(x, y, pred)
            error_b = self.compute_gradients_wrt_b(y, pred)
            # Perform 1 step of steepest descent
            self.update_model_parameters(error_a, error_b, learning_rate)
            #  Getting accuracy score of the model
            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            accuracy = accuracy_score(y, pred_to_class)
            self.train_accuracies.append(accuracy)
            # Stop training if accuracy score is larger than 95%
            if accuracy > 0.95:
                break
            
    """
        Predicting function
        @params: x: (m x n) array of independent variables (ui)
        @return: (1 x n) array of predictions
    """        
    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]
