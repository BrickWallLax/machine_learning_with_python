from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random as rand


# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([4, 6, 8, 5, 6, 7], dtype=np.float64)


def create_dataset(num_points, variance, step=2, correlation=True):
    val = 1
    ys = []
    xs = []
    for i in range(num_points):
        y = val + rand.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
        xs.append(i)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    numerator = (mean(xs) * mean(ys)) - mean(xs * ys)
    denominator = (mean(xs) * mean(xs)) - mean(xs * xs)
    m = numerator / denominator
    b = mean(ys) - (m * mean(xs))
    return m, b


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_mean)


num_points = 40
xs, ys = create_dataset(num_points, 40, 10, correlation=False)  # New random dataset

m, b = best_fit_slope_and_intercept(xs, ys)  # Sets m and b to best fit line

predict_x = num_points + 1
predict_y = m * predict_x + b

regression_line = [m * x + b for x in xs]

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

style.use('ggplot')  # Styling the plot
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, s=10, color='green')
plt.plot(xs, regression_line)
plt.show()
