"""
Defintions for problem 0
"""

import numpy as np
import scipy.integrate
from scipy.integrate import DenseOutput
from scipy.interpolate import interp1d


class ForwardEuler(scipy.integrate.OdeSolver):
    """
    Forward Euler for solving ODE
    """
    def __init__(self, fun, t0, y0, t_bound, vectorized, support_complex = False, h = None):
        if h == None:
            h = (t_bound - t0) / 100
        super(ForwardEuler, self).__init__(fun, t0, y0, t_bound, vectorized)
        self.h = h
        self.ys = []
        self.ts = []
    
    def _step_impl(self):
        self.ts.append(self.t)
        self.ys.append(self.y)
        y = self.ys[-1]
        t = self.ts[-1]
        self.y = y + self.h * self.fun(t,y)
        self.t = t + self.h

        if(np.abs(self.t_bound - t) < 1e-08):
            self.t = self.t_bound
        
        return True,None

    def _dense_output_impl(self):
        return ForwardEulerOutput(self.ts, self.ys)
    
    



class ForwardEulerOutput(DenseOutput):
    """
    Interpolate ForwardEuler output

    """
    def __init__(self, ts, ys):

        """
        store ts and ys computed in forward Euler method

        These will be used for evaluation
        """
        super(ForwardEulerOutput, self).__init__(np.min(ts), np.max(ts))
        self.interp = interp1d(ts, ys, kind='linear', copy=True)


    def _call_impl(self, t):
        """
        Evaluate on a range of values
        """
        return self.interp(t)
