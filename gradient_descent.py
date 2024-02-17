import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

x = 15 + 5 * np.random.randn(100, 1)
y = 22 + x + 2 * np.random.randn(100,1)


class GradientDescent:
    
    def __init__(self, 
                 lr:float=0.001, 
                 verbose:bool=True, 
                 plot_errors=False, 
                 plot_weights=False) -> None:
        
        self.lr:float = lr
        self.verbose:bool = verbose
        self.theta: np.ndarray = None
        self.shape: tuple = None
        self.plot_errors = plot_errors
        self.plot_weights = plot_weights
        
        self.metrics = {
            'mean_absolute_error': mean_absolute_error,
            'mean_squared_error': mean_squared_error,
            'mae': mean_absolute_error,
            'mse': mean_squared_error
        }
    
    def train(self,
              x: np.ndarray, 
              y: np.ndarray,
              epochs:int=100,
              batch_size:int=32,
              include_bias:bool=True,
              metric:str='mean_absolute_error'):
        
        if len(x.shape) != 2:
            raise ValueError(f'Expected shape (any,any) received {x.shape}')
        
        if batch_size > x.shape[0]:
            raise ValueError('Batch size must be smaller than the number of instances')
        
        if metric not in self.metrics.keys():
            raise ValueError('You must use a valid metric')
        
        self.shape = x.shape
        
        self.theta = np.random.randn(x.shape[1] + 1,1)
        xb = np.c_[np.ones((x.shape[0],1)), x] if include_bias else x
        
        errors = []
        thetas = []
        for i in range(epochs):
            for j in range(int(epochs/xb.shape[0])):
                indexes = np.random.randint(0, x.shape[0], size=batch_size)
                samples = xb[indexes]
                y_ = y[indexes]
                
                gradients = (2/xb.shape[0]) * samples.T @ (samples @ self.theta - y_)
                self.theta -= self.lr * gradients
            
            thetas.append([self.theta[0,0], self.theta[1,0]])
            ynew_ = xb @ self.theta
            error = self.metrics.get(metric)(y, ynew_)
            errors.append(error)
            
            if self.verbose:
                print(f'{metric}: {error:.3f}')
                
        if self.plot_errors:
            plt.plot([x for x in range(epochs)], errors)
            plt.show()
            plt.close()
            
        if self.plot_weights:
            thetas_ = np.array(thetas)
            plt.plot(thetas_[:,0], thetas_[:,1])
            plt.show()
            plt.close()

    
    def train_on_batch(self,
              x: np.ndarray, 
              y: np.ndarray,
              include_bias:bool=True,
              metric:str='mean_absolute_error') -> float:
        
        if len(x.shape) != 2:
            raise ValueError(f'Expected shape (any,any) received {x.shape}')
        
        if metric not in self.metrics.keys():
            raise ValueError('You must use a valid metric')
        
        self.shape = x.shape
        
        if self.theta is None:
            self.theta = np.random.randn(x.shape[1] + 1,1)
        
        xb = np.c_[np.ones((x.shape[0],1)), x] if include_bias else x
        
        gradients = (2/xb.shape[0]) * xb.T @ (xb @ self.theta - y)
        self.theta -= self.lr * gradients
           
           
        ynew_ = xb @ self.theta
        error = self.metrics.get(metric)(y, ynew_)
        if self.verbose:
            print(f'{metric}: {error:.3f}')
        return error


    def predict(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) != len(self.shape) or x.shape[1] != self.shape[1]:
            raise ValueError(f'x shape must be of size (any, {self.shape[1]})')
        return np.c_[np.ones((self.shape[0],1)), x] @ self.theta

        

gd = GradientDescent(lr = 0.002, verbose=True, plot_weights=True, plot_errors=True)
gd.train(x, y, epochs=10000, batch_size=100)
ynew = gd.predict(x)

plt.scatter(x, y)
plt.plot(x, ynew)

