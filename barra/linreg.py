""" 
Linear Regression Formula
Y = a + bX

Y = dependent variable
X = independent variable
a = intercept
b = slope

"""
class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __mean(self, axis_var = "x"):
        
        if axis_var == "x":
            loop_var = self.x
        elif axis_var == "y":
            loop_var = self.y
        else:
            raise ValueError("Axis just X dan Y")
                
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

        x_mean = LinearRegression.__mean(self, axis_var= "x")
        y_mean = LinearRegression.__mean(self, axis_var= "y")
        
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
        a = LinearRegression.__mean(self, axis_var= "y") - LinearRegression.__find_slope_b(self) * LinearRegression.__mean(self, axis_var= "x")
        return a
    
    def predict(self, pred_var):
        
        """ 
        
        Y = a + b (X)
        
        X = prediction variable
        
        """
        
        slope = LinearRegression.__find_slope_b(self)
        intercept = LinearRegression.__find_intercept_a(self)

            
        Y = intercept + slope * pred_var
        
        return Y
        
    