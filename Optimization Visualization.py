import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pylab as pl
import math

def SGD(x, ddx, alpha, opt_data):
    '''
    Num, (Num -> Num), Num, dict(None) -> Num

    Returns a one-step stochastic gradient descent value.
    '''
    return x - alpha * ddx(x)

def Adam(x, ddx, alpha, opt_data):
    '''
    Num, (Num -> Num), Num, dict(mv1, mv2, beta1, beta2, epsilon, t) -> Num

    Returns a one-step Adam optimizer (with momentum) value.
    '''
    opt_data['mv1'] = opt_data['beta1'] * opt_data['mv1'] + (1 - opt_data['beta1']) * ddx(x)
    opt_data['mv2'] = opt_data['beta2'] * opt_data['mv2'] + (1 - opt_data['beta2']) * (ddx(x))**2
    mv1 = opt_data['mv1']/(1 - opt_data['beta1'] ** opt_data['t'])
    mv2 = opt_data['mv2']/(1 - opt_data['beta2'] ** opt_data['t'])
    opt_data['t'] += 1
    
    return x - (alpha * mv1)/(math.sqrt(mv2) + opt_data['epsilon'])

class V:
    
    def __init__(self, fnc, ddx_fnc, alpha, x0,
                 optimizer=SGD, opt_data = {}, start=-10, end=10,
                 tolerance=None, g_truth=None, interval=100):
        '''
        (Num -> Num), (Num -> Num), Num, Num, OPTIMIZER,
            dict(...), Num, Num, Num/None, Num/None, Int -> V

        Create a V (Visualization) object that visualizes each step of
        any optimization algorithm.

        Parameters:
        * fnc - function to be optimized
        * ddx_fnc - derivative of fnc
        * alpha - learning rate
        * x0 - starting x-value
        * optimizer - optimizer function
        * opt_data - data dictionary to be fed to optimizer function
        * start - left boundary of visualization
        * end - right boundary of visualization
        * tolerance - minimum error before stopping visualization
        * g_truth - actual x-value of optimum
        * interval - period of each iteration

        Requires:
        * tolerance and g_truth paramater must be both None or
            both specified
        * optimizer function must have 4 parameters:
            * x
            * derivative of function
            * learning rate
            * custom data to be fed in dictionary format
        
        '''
        self.fnc = fnc
        self.ddx_fnc = ddx_fnc
        self.alpha = alpha
        self.optimizer = optimizer
        self.opt_data = opt_data
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.ani = animation.FuncAnimation(self.fig, self.cycle, interval=interval)

        self.lin = pl.linspace(start,end,1000)
        self.tolerance = tolerance
        self.g_truth = g_truth

        self.x = x0
        self.index = 0

    def tangent(self, x):
        grad = self.ddx_fnc(x)
        intercept = self.fnc(x) - grad * x
        return lambda z: grad * z + intercept
    
    def read_cur(self):
        x = self.x
        print(x)
        self.ax.plot(x, self.fnc(x), 'ro')

    def show_graph(self):
        y = self.fnc(self.lin)
        self.ax.plot(self.lin, y)

    def show_tangent(self):
        y = self.tangent(self.x)
        x1 = self.lin[0]
        x2 = self.lin[-1]
        x = [x1, x2]
        y = [y(i) for i in x]
        self.ax.plot(x, y)

    def update(self):
        self.x = self.optimizer(self.x, self.ddx_fnc, self.alpha, self.opt_data)
        self.index += 1

    def cycle(self, i):
        self.ax.clear()
        x = self.read_cur()
        self.show_graph()
        self.show_tangent()
        if self.tolerance == None:
            pass
        elif abs(self.g_truth - self.x) < self.tolerance:
            self.ani._stop()
        self.update()
    
    def visualize(self):
        '''
        Initiate visualization of algorithm.
        '''
        self.index = 0
        plt.show()
        return self.index

# Example of Visualization of Adam Optimizer
v = V(lambda x: x ** 3 - 13 * x ** 2 + 6 * x,
      lambda x: 3 * x ** 2 - 26 * x + 6, 0.8, 1,
      optimizer=Adam, opt_data={'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8,
                                'mv1':0, 'mv2':0, 't':1},
      tolerance=0.01, g_truth=8.429401909148168, start=-5, end=15)
index = v.visualize()
print(index)

