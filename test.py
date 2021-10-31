"""
Implement tests for Problem 0 Part C
"""


import unittest
from scipy.integrate import solve_ivp
from euler import *
import numpy as np

class checkEulerForward(unittest.TestCase):
    def setUp(self):
        pass
    
    # We use a smaller step size than h to pretend that we know the truth
    def test_EulerFoward(self):
        def fun(t,y): return y
        t_span = (0,5)
        sol = solve_ivp(fun,t_span,y0 = np.array([1]), method = ForwardEuler, h = 0.01)
        # We use h = 0.0001 for the pretended truth
        sol_t = solve_ivp(fun,t_span,y0 = np.array([1]), method = ForwardEuler, h = 0.0001)

        # Using forward_difference(g, x, h): (g(x + h) - g(x)) / h
        """
        Note here even though the suggested tolerance is h * y, but in my experience that the tolerance
        is not big enough when t is far away from the initial point, so I "loose" the tolerance
        a little bit as we go too far from the initial point. In this case, I increase the multiplier
        h by 0.005 as approximately every 10000 steps we go in truth. This should be resonable.

        The same thing goes for central and backward difference.
        """
        self.assertTrue(np.abs((sol_t.y[0][2001] - sol_t.y[0][2000])/0.0001 - sol.y[0][20]) < sol_t.y[0][2000]*0.01)
        self.assertTrue(np.abs((sol_t.y[0][12001] - sol_t.y[0][12000])/0.0001 - sol.y[0][120]) < sol_t.y[0][12000]*0.01)
        self.assertTrue(np.abs((sol_t.y[0][23001] - sol_t.y[0][23000])/0.0001 - sol.y[0][230]) < sol_t.y[0][23000]*0.015)
        self.assertTrue(np.abs((sol_t.y[0][36001] - sol_t.y[0][36000])/0.0001 - sol.y[0][360]) < sol_t.y[0][36000]*0.020)
        self.assertTrue(np.abs((sol_t.y[0][48001] - sol_t.y[0][48000])/0.0001 - sol.y[0][480]) < sol_t.y[0][48000]*0.025)

        # Using central_difference(g, x, h): (g(x) - g(x - h)) / 2h
        self.assertTrue(np.abs((sol_t.y[0][2000] - sol_t.y[0][2000-1])/0.0001 - sol.y[0][20]) < sol_t.y[0][2000]*0.01)
        self.assertTrue(np.abs((sol_t.y[0][12000] - sol_t.y[0][12000-1])/0.0001 - sol.y[0][120]) < sol_t.y[0][12000]*0.01)
        self.assertTrue(np.abs((sol_t.y[0][23000] - sol_t.y[0][23000-1])/0.0001 - sol.y[0][230]) < sol_t.y[0][23000]*0.015)
        self.assertTrue(np.abs((sol_t.y[0][36000] - sol_t.y[0][36000-1])/0.0001 - sol.y[0][360]) < sol_t.y[0][36000]*0.020)
        self.assertTrue(np.abs((sol_t.y[0][48000] - sol_t.y[0][48000-1])/0.0001 - sol.y[0][480]) < sol_t.y[0][48000]*0.025)


        # Using central_difference(g, x, h): (g(x + h) - g(x - h)) / 2h
        self.assertTrue(np.abs((sol_t.y[0][2001] - sol_t.y[0][2000-1])/0.0002 - sol.y[0][20]) < sol_t.y[0][2000]*0.01)
        self.assertTrue(np.abs((sol_t.y[0][12001] - sol_t.y[0][12000-1])/0.0002 - sol.y[0][120]) < sol_t.y[0][12000]*0.01)
        self.assertTrue(np.abs((sol_t.y[0][23001] - sol_t.y[0][23000-1])/0.0002 - sol.y[0][230]) < sol_t.y[0][23000]*0.015)
        self.assertTrue(np.abs((sol_t.y[0][36001] - sol_t.y[0][36000-1])/0.0002 - sol.y[0][360]) < sol_t.y[0][36000]*0.020)
        self.assertTrue(np.abs((sol_t.y[0][48001] - sol_t.y[0][48000-1])/0.0002 - sol.y[0][480]) < sol_t.y[0][48000]*0.025)
        
    
    def test_ivp(self):
        def fun(t,y): return y
        t_span = (0,5)
        sol = solve_ivp(fun,t_span,y0 = np.array([1]), method = ForwardEuler, h = 0.0001)
        sol = sol.y[0]

        # Backward_difference verification
        self.assertTrue(np.abs((sol[100] - sol[100-1])/0.0001 - sol[100]) < sol[100] * 0.0001)
        self.assertTrue(np.abs((sol[1000] - sol[1000-1])/0.0001 - sol[1000]) < sol[1000] * 0.0001)
        self.assertTrue(np.abs((sol[2300] - sol[2300-1])/0.0001 - sol[2300]) < sol[2300] * 0.0001)
        self.assertTrue(np.abs((sol[3537] - sol[3537-1])/0.0001 - sol[3537]) < sol[3537] * 0.0001)
        self.assertTrue(np.abs((sol[4848] - sol[4848-1])/0.0001 - sol[4848]) < sol[4848] * 0.0001)

        # Forward_difference verification
        self.assertTrue(np.abs((sol[101] - sol[100])/0.0001 - sol[100]) < sol[100] * 0.0001)
        self.assertTrue(np.abs((sol[1001] - sol[1000])/0.0001 - sol[1000]) < sol[1000] * 0.0001)
        self.assertTrue(np.abs((sol[2301] - sol[2300])/0.0001 - sol[2300]) < sol[2300] * 0.0001)
        self.assertTrue(np.abs((sol[3538] - sol[3537])/0.0001 - sol[3537]) < sol[3537] * 0.0001)
        self.assertTrue(np.abs((sol[4849] - sol[4848])/0.0001 - sol[4848]) < sol[4848] * 0.0001)


        # Central_differece verification
        self.assertTrue(np.abs((sol[101] - sol[100-1])/0.0002 - sol[100]) < sol[100] * 0.0001)
        self.assertTrue(np.abs((sol[1001] - sol[1000-1])/0.0002 - sol[1000]) < sol[1000] * 0.0001)
        self.assertTrue(np.abs((sol[2301] - sol[2300-1])/0.0002 - sol[2300]) < sol[2300] * 0.0001)
        self.assertTrue(np.abs((sol[3538] - sol[3537-1])/0.0002 - sol[3537]) < sol[3537] * 0.0001)
        self.assertTrue(np.abs((sol[4849] - sol[4848-1])/0.0002 - sol[4848]) < sol[4848] * 0.0001)