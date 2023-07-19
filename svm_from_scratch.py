import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        #  Don't worry about this
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # Training
    def fit(self, data):
        self.data = data
        # { ||w||: [w, b]}
        opt_dict = {}

        # # Transform the data after the equation b/c data comes out as [+, +] and we want to test all vectors instead of
        # # just positive ones
        # transform = [[1, 1],
        #              [-1, -1],
        #              [-1, 1],
        #              [1, -1]]

        rot_matrix = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
                                            [np.sin(theta), np.cos(theta)]])

        theta_step = np.pi / 10
        transforms = [(np.matrix(rot_matrix(theta)) * np.matrix([1, 0]).T).T.tolist()[0]
                      for theta in np.arange(0, np.pi, theta_step)]

        # Add all features to dataset
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        # For each "swing", get the max and min then reset the data and swing back
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # Take smaller and smaller steps until you get a "perfect" support vector
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # Point of expense
                      self.max_feature_value * 0.001]

        # Extremely expensive
        b_range_multiple = 2
        # We don't need to take as small of steps
        # with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            # Start the first swing at max value
            w = np.array([latest_optimum, latest_optimum])
            optimised = False
            while not optimised:
                for b in np.arange(-1 * (self.max_feature_value*b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # Weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w + b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                # If yi*(np.dot(w_t, xi) + b) < 1, don't add it to the opt_dict
                                if not yi*(np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    break
                                # print(xi, ':', yi*(np.dot(w_t, xi) + b))
                            if not found_option:
                                break
                        if found_option:
                            # np.linalg.norm(w_t) is the magnitude of the vector
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimised = True
                    print('Optimized a step')
                else:
                    # w = [5,5], step = 1, w - step = [4, 4]
                    w = w - step

            # Sorted list of magnitudes, sort them from lowest to highest
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            # ||w|| : [w, b]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            # Reset the latest optimum to smaller value
            latest_optimum = opt_choice[0][0]+step*2
            print(self.w / np.linalg.norm(self.w))

    # Checks whether the class is a positive or negative
    def predict(self, features):
        # sign(x.w + b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualise(self):
        [[self.ax.scatter(x[0], x[1], s=100, c=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # Hyperplane = x.w + b
        # v = x.w + b
        # psv = 1

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        data_range = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]

        # x.w + b = 1
        # positive support vector
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # x.w + b = -1
        # negative support vector
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # x.w + b = 0
        # decision boundary
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


data_dict = {-1: np.array([[1, 1],
                           [2, -1],
                           [3, 2]]),
              1: np.array([[1, 3],
                           [2, 7],
                           [3, 6]])}

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0, 10],
              [1, 3],
              [5, 5],
              [5, 6],
              [6, -5],
              [5, 8]]
for p in predict_us:
    svm.predict(p)
svm.visualise()