""" 
Linear Regression Formula
Y = a + bX

Y = dependent variable
X = independent variable
a = intercept
b = slope

"""

import numpy as np

class LinearRegression:
    def __init__(self):
        self.x = None
        self.y = None
        
    def fit(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
    
    def __mean(self, axis_var = "x"):
        
        if axis_var == "x":
            loop_var = self.x
        elif axis_var == "y":
            loop_var = self.y
        else:
            raise ValueError("just X dan Y")
                
        temp = 0
        for i in loop_var:
            temp += i
            
        mean = temp / len(loop_var)

        return mean
    
    def __find_slope_b(self):
        
        
        """ 
        
        
        b = summation (x[i] - x_mean) * (y[i] - y_mean) / summation (x[i] - x_mean)**2

        """
        
        if len(self.x) != len(self.y):
            raise ValueError("Lenght X and Y must be equal")

        x_mean = self.__mean(axis_var= "x")
        y_mean = self.__mean(axis_var= "y")
        
        up = 0
        down = 0
        for i in range(len(self.x)):
            up += (self.x[i] - x_mean) * (self.y[i] - y_mean)     
            down += (self.x[i] - x_mean)**2

        b = up / down
        
        return b
    
    def __find_intercept_a(self):
        
        """ 
        
        a = y_mean - b*x_mean
        
        """
        a = self.__mean(axis_var= "y") - self.__find_slope_b() * self.__mean(axis_var= "x")
        return a
    
    def predict(self, pred_var):
        
        """ 
        
        Y = a + b (X)
        
        X = prediction variable
        
        """
        
        slope = self.__find_slope_b()
        intercept = self.__find_intercept_a()
        
        pred_var = np.array(pred_var)

        Y_pred = []
        for i in pred_var:
            Y = intercept + slope * i
            Y_pred.append(Y)
        
        return Y_pred
        
    