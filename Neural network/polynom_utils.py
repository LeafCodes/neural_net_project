# polynom_utils.py
import numpy as np

def polynom_curve_fit(x_t, y_t, degree_t):
    coefficients = np.polyfit(x_t, y_t, degree_t)
    poly_function = np.poly1d(coefficients)
    x_fit = np.linspace(min(x_t), max(x_t), 100)
    y_fit = poly_function(x_fit)
    return x_fit, y_fit