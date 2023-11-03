import pandas as pd
import math
from enum import Enum, auto

df = pd.DataFrame({
        'x': [1, 1],
        'w': [0, 0]
    })

# ENUM ActivationFunction
class ActivationFunction(Enum):
    LOGISTIC = auto()
    BINARY = auto()
# END ENUM ActivationFunction

# CLASS Perceptron
class Perceptron:
    
    def __init__(self, x, w, true: int, iters: int = 1, actv_func: ActivationFunction = ActivationFunction.LOGISTIC):
        if(len(x) != len(w)):
            raise ValueError('x and w must be of the same length')
        self.x = x
        self.w = w
        self.true = true
        self.iters = iters
        self.result = 0
        self.activated = False
        self.actv_func = actv_func
        
    def sum(self) -> float:
        return (self.x * self.w).sum()
    
    def logistic(self):
        return 1/(math.exp(-self.sum() + 1))
    
    def binary(self):
        return 1 if self.sum() >= 1 else 0
    
    def step(self) -> int:
        result = self.logistic() if self.actv_func == ActivationFunction.LOGISTIC else self.binary()
        self.result = result
        return result
    
    def updateWeight(self):
        self.w += 0.0501 * self.x * (self.true - self.result)
        
    def learn(self):
        for i in range(self.iters):
            self.updateWeight()
            self.step()
            if(self.result >= 0.5):
                self.activated = True
# END CLASS Perceptron
             
pc = Perceptron(df['x'], df['w'], 1, iters=5, actv_func=ActivationFunction.LOGISTIC)

pc.learn()
print(pc.w)
print(pc.x)
print(pc.result)
print(pc.sum())
print('activated?', 'YES' if pc.activated else 'NO')
