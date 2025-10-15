from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from scipy import stats
import pandas as pd
from scipy.optimize import minimize
import numpy as np

def mean_square_error(X, y_pred, y_true):
    return (sum((y_true-y_pred)**2))/(X.shape[0]-X.shape[1])

class CustomLinearRegression(LinearRegression):
    def __init__(self, loss_function=mean_square_error, 
                 X=None, Y=None):
        self.loss_function=loss_function
        self.X=X
        self.Y=Y

        # Call init from base class
        super().__init__()

    def fit(self):
        super().fit(self.X, self.Y)
        return np.append(self.intercept_, self.coef_[1:])
    
    def score(self):
        return super().score(self.X, self.Y)
        
    def predict(self, X):
        predictions=super().predict(X)
        return predictions
    
    def model_error(self):
        error=self.loss_function(
            self.X, self.predict(self.X), self.Y
        )
        return error

    def summary(self):
        params=self.fit()
        v_b=self.model_error()*(np.linalg.inv(np.dot(self.X.T, self.X)).diagonal())
        s_b=np.sqrt(v_b)
        t_b=params/s_b

        p_val=[2*(1-stats.t.cdf(np.abs(i), (self.X.shape[0]-self.X.shape[1]-1))) for i in t_b]

        p_val=np.round(p_val, 3)
        params=np.round(params, 3)
        s_b=np.round(s_b, 3)
        t_b=np.round(t_b, 3)

        my_array=np.array([params.tolist(), s_b.tolist(), t_b.tolist(), p_val.tolist()]).T
        data=pd.DataFrame(my_array, columns=["coef", "std err", "t", "P>|t|"])
    
        return data

class CustomCrossValidation:
    def __init__(self, loss_function=mean_square_error, 
                 X=None, Y=None, ModelClass=CustomLinearRegression):
        self.loss_function=loss_function
        self.X=X
        self.Y=Y
        self.ModelClass = ModelClass
    
    def cross_validate(self):
        X=self.X
        Y=self.Y

        clf=self.ModelClass(
            loss_function=self.loss_function, 
            X=X, 
            Y=Y
        )
        clf.fit()
        data=clf.summary()

        index_lst = []
        for index, row in data.iterrows():
            if float(row["P>|t|"]) > 0.05:
                index_lst.append(index)

        for idx in sorted(index_lst, reverse=True):
            X=np.delete(X,(idx),axis=1)

        self.X_star_index=index_lst
        self.X_star=X