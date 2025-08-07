import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# for function f:
f_coeffs = np.array([0.80, 0.0, 0.06]) # https://www.desmos.com/calculator/wczixvz2zk
f = lambda sec_loss_param: np.dot(f_coeffs, np.array([sec_loss_param**i for i in range(3, -1, -1)]))

class AM_nozzle:
    """Ainley and Mathieson profile loss coefficients"""

    def __init__(self) -> None:

        # for function Y_nozzle
        def f_nozzle_dec(sc_coeffs, beta_coeffs):
            def f_nozzle(sc, beta):
                return np.dot(sc_coeffs, np.array([sc**i for i in range(3, -1, -1)])) + \
                        np.dot(beta_coeffs, np.array([beta**i for i in range(3, -1, -1)]))
            return f_nozzle

        # get the data points
        sc = np.array([np.linspace(0.30, 0.91),    # z = 80 degrees
                       np.linspace(0.30, 1.00),    # z = 75 degrees
                       np.linspace(0.30, 1.10),    # z = 70 degrees
                       np.linspace(0.30, 1.10),    # z = 65 degrees
                       np.linspace(0.30, 1.10),    # z = 60 degrees
                       np.linspace(0.30, 1.10),    # z = 50 degrees
                       np.linspace(0.30, 1.10)])   # z = 40 degrees

        beta = np.array([[np.deg2rad(80) for i in range(50)],
                    [np.deg2rad(75) for i in range(50)],
                    [np.deg2rad(70) for i in range(50)],
                    [np.deg2rad(65) for i in range(50)],
                    [np.deg2rad(60) for i in range(50)],
                    [np.deg2rad(50) for i in range(50)],
                    [np.deg2rad(40) for i in range(50)]])

        # source: https://www.desmos.com/calculator/fdhb1vvplf
        Yp_funcs = [lambda x: np.dot(np.array([0.00472746457763, 0.134583745183, -0.163697000305, 0.107366420974]),  np.array([x**i for i in range(3, -1, -1)])),  # z = 80 degrees
                    lambda x: np.dot(np.array([-0.0414167573814, 0.223613959998, -0.238872125931, 0.121529411906]),  np.array([x**i for i in range(3, -1, -1)])),  # z = 75 degrees
                    lambda x: np.dot(np.array([-0.0486578892528, 0.257992471753, -0.2955067698091, 0.135810367152]), np.array([x**i for i in range(3, -1, -1)])), # z = 70 degrees
                    lambda x: np.dot(np.array([0.0206060273238, 0.108790013801, -0.218053636229, 0.12410152093]),    np.array([x**i for i in range(3, -1, -1)])),    # z = 65 degrees
                    lambda x: np.dot(np.array([0.00127100059953, 0.14745239856, -0.252092613573, 0.13143542614]),    np.array([x**i for i in range(3, -1, -1)])),    # z = 60 degrees
                    lambda x: np.dot(np.array([-0.0118351782982, 0.141873179622, -0.230092081748, 0.120869144074]),  np.array([x**i for i in range(3, -1, -1)])), # z = 50 degrees
                    lambda x: np.dot(np.array([-0.0626302442791, 0.239070828964, -0.290072414112, 0.131547607304]),   np.array([x**i for i in range(3, -1, -1)]))] # z = 40 degrees
        Yp = np.array([[Yp_funcs[i](sc[i, j]) for j in range(len(sc[i]))] for i in range(len(sc))])

        data = []
        for i in range(len(beta)):
            for j in range(len(beta[0])):
                data.append([sc[i, j], beta[i, j], Yp[i, j]])
        data = np.array(data)
        print(f'shape of the data: {np.shape(data)}')

        self.Yp_funcs = Yp_funcs
        self.sc = sc
        self.beta = beta
        self.Yp = Yp

        # conduct the regression
        X = data[:, :2]   # shape (n, 2), for x and y
        z = data[:, 2]    # shape (n,), for z



        # Generate polynomial features up to degree 3
        poly = PolynomialFeatures(degree=3, include_bias=True)
        X_poly = poly.fit_transform(X)  # shape (n, 10)

        # Fit linear regression model on the polynomial features
        model = LinearRegression()
        model.fit(X_poly, z)

        # Coefficients of the model
        coefficients = model.coef_
        intercept = model.intercept_

        print(f'coefficients: {coefficients}')

        self.model = model
        self.poly = poly

    def nozzle_Y(self, sc, beta):
        # s/c
        # beta in radians
        # returns the value at a test point

        new_point = np.array([[sc, beta]])
        new_points_poly = self.poly.transform(new_point)
        z_new = self.model.predict(new_points_poly)
        return z_new
        
    def compare_2d_y_slices(self):
        # cross section at 65 degrees

        for ang, sc in zip(np.deg2rad(np.array([80, 75, 70, 65, 60, 50, 40], dtype=float)), self.sc):
            Y_p = [self.nozzle_Y(sc[i], ang) for i in range(len(sc))]
            plt.plot(sc, Y_p, color='r') # evaluation of 3d function in slices
            plt.title(f'cross section {np.rad2deg(ang)} degrees')

        angs = np.array([80, 75, 70, 65, 60, 50, 40])
        for i, sc in zip(range(7), self.sc):
            Y_p = [self.Yp_funcs[i](sc[j]) for j in range(len(sc))]
            plt.plot(sc, Y_p, color='g')        # 2d functions evaluated
            plt.plot(sc, self.Yp[i], color='b') # points used to make the 3d interpolation function
            plt.title(f'cross section {angs[i]} degrees')
        plt.show()

    def test_plot_3d_y(self):
        plt.style.use('_mpl-gallery')
        # transform data

        # plot the 3d interpolation
        test_sc = np.linspace(0.30, 1.10)
        test_beta = np.linspace(np.deg2rad(80), np.deg2rad(40))
        test_sc, test_beta = np.meshgrid(test_sc, test_beta)

        test_data = []
        for i in range(len(test_beta)):
            for j in range(len(test_beta[0])):
                test_data.append([test_sc[i, j], test_beta[i, j]])
        test_data = np.array(test_data)
        print(f'shape of the data: {np.shape(test_data)}')

        new_points_poly = self.poly.transform(test_data)
        Y_p = np.array(self.model.predict(new_points_poly)).reshape(np.shape(test_sc))

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(test_sc, test_beta, Y_p)    # type: ignore

        # Plot the 2d polyfits
        ax.plot_surface(self.sc, self.beta, self.Yp, vmin=self.Yp.min() * 2, cmap=cm.Blues) # type: ignore
        ax.set(xticklabels=[],
            yticklabels=[],
            zticklabels=[])

        plt.show()


class AM_impulse:
    """Ainley and Mathieson profile loss coefficients"""

    def __init__(self) -> None:


        # get the data points
        sc = np.array([np.linspace(0.30, 1.00),    # z = 80 degrees
                       np.linspace(0.30, 1.00),    # z = 75 degrees
                       np.linspace(0.30, 1.00),    # z = 70 degrees
                       np.linspace(0.30, 1.00),    # z = 65 degrees
                       np.linspace(0.30, 1.00),    # z = 60 degrees
                       np.linspace(0.36, 1.00),    # z = 50 degrees
                       np.linspace(0.36, 1.00)])   # z = 40 degrees

        beta = np.array([[np.deg2rad(70) for i in range(50)],
                         [np.deg2rad(65) for i in range(50)],
                         [np.deg2rad(60) for i in range(50)],
                         [np.deg2rad(55) for i in range(50)],
                         [np.deg2rad(50) for i in range(50)],
                         [np.deg2rad(40) for i in range(50)]])

        # source: https://www.desmos.com/calculator/mtdyrs5nsb
        Yp_funcs = [lambda x: np.dot(np.array([-0.522617375138, 1.29226068611, -0.91747610856, 0.339347477755]),  np.array([x**i for i in range(3, -1, -1)])),  # z = 80 degrees
                    lambda x: np.dot(np.array([-0.266687874816, 0.817311943903, -0.694002305521, 0.298341935381]),  np.array([x**i for i in range(3, -1, -1)])),  # z = 75 degrees
                    lambda x: np.dot(np.array([-0.306004835077, 0.865094222314, -0.729457288437, 0.296540954294]), np.array([x**i for i in range(3, -1, -1)])), # z = 70 degrees
                    lambda x: np.dot(np.array([-0.091501022008, 0.461507886507, -0.523558441469, 0.259625486233]),    np.array([x**i for i in range(3, -1, -1)])),    # z = 60 degrees
                    lambda x: np.dot(np.array([-0.200367557195, 0.717021413878, -0.732498435497, 0.306537203779]),  np.array([x**i for i in range(3, -1, -1)])), # z = 50 degrees
                    lambda x: np.dot(np.array([-0.213599271696, 0.734269267167, -0.754320373846, 0.310429281863]),   np.array([x**i for i in range(3, -1, -1)]))] # z = 40 degrees
        Yp = np.array([[Yp_funcs[i](sc[i, j]) for j in range(len(sc[i]))] for i in range(len(sc))])

        data = []
        for i in range(len(beta)):
            for j in range(len(beta[0])):
                data.append([sc[i, j], beta[i, j], Yp[i, j]])
        data = np.array(data)
        print(f'shape of the data: {np.shape(data)}')

        self.Yp_funcs = Yp_funcs
        self.sc = sc
        self.beta = beta
        self.Yp = Yp

        # conduct the regression
        X = data[:, :2]   # shape (n, 2), for x and y
        z = data[:, 2]    # shape (n,), for z



        # Generate polynomial features up to degree 3
        poly = PolynomialFeatures(degree=3, include_bias=True)
        X_poly = poly.fit_transform(X)  # shape (n, 10)

        # Fit linear regression model on the polynomial features
        model = LinearRegression()
        model.fit(X_poly, z)

        # Coefficients of the model
        coefficients = model.coef_
        intercept = model.intercept_

        print(f'coefficients: {coefficients}')

        self.model = model
        self.poly = poly

    def nozzle_Y(self, sc, beta):
        # s/c
        # beta in radians
        # returns the value at a test point

        new_point = np.array([[sc, beta]])
        new_points_poly = self.poly.transform(new_point)
        z_new = self.model.predict(new_points_poly)
        return z_new
        
    def compare_2d_y_slices(self):
        # cross section at 65 degrees

        for ang, sc in zip(np.deg2rad(np.array([80, 75, 70, 65, 60, 50, 40], dtype=float)), self.sc):
            Y_p = [self.nozzle_Y(sc[i], ang) for i in range(len(sc))]
            plt.plot(sc, Y_p, color='r') # evaluation of 3d function in slices
            plt.title(f'cross section {np.rad2deg(ang)} degrees')

        angs = np.array([80, 75, 70, 65, 60, 50, 40])
        for i, sc in zip(range(7), self.sc):
            Y_p = [self.Yp_funcs[i](sc[j]) for j in range(len(sc))]
            plt.plot(sc, Y_p, color='g')        # 2d functions evaluated
            plt.plot(sc, self.Yp[i], color='b') # points used to make the 3d interpolation function
            plt.title(f'cross section {angs[i]} degrees')
        plt.show()

    def test_plot_3d_y(self):
        plt.style.use('_mpl-gallery')
        # transform data

        # plot the 3d interpolation
        test_sc = np.linspace(0.30, 1.10)
        test_beta = np.linspace(np.deg2rad(80), np.deg2rad(40))
        test_sc, test_beta = np.meshgrid(test_sc, test_beta)

        test_data = []
        for i in range(len(test_beta)):
            for j in range(len(test_beta[0])):
                test_data.append([test_sc[i, j], test_beta[i, j]])
        test_data = np.array(test_data)
        print(f'shape of the data: {np.shape(test_data)}')

        new_points_poly = self.poly.transform(test_data)
        Y_p = np.array(self.model.predict(new_points_poly)).reshape(np.shape(test_sc))

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(test_sc, test_beta, Y_p)    # type: ignore

        # Plot the 2d polyfits
        ax.plot_surface(self.sc, self.beta, self.Yp, vmin=self.Yp.min() * 2, cmap=cm.Blues) # type: ignore
        ax.set(xticklabels=[],
            yticklabels=[],
            zticklabels=[])

        plt.show()

class AM:
    def __init__(self,
                  sc, # pitch/chord
                    tc, # thickness/chord
                      beta2,
                        beta3):
        # sc = pitch / chord
        # beta = 
        am_nozzle = AM_nozzle()
        am_impulse = AM_impulse()
        # use takachi's interpolation formula

        Y_p_nozzle = am_nozzle.nozzle_Y(sc, beta3)
        Y_p_impulse = am_impulse.nozzle_Y(sc, beta3) # for beta2 = beta3
        self.Y_p = (Y_p_nozzle + (beta2/beta3)**2 * (Y_p_impulse - Y_p_nozzle))*(5*tc)**(beta2/beta3)



if __name__ == '__main__':
    am = AM_nozzle()
    am.nozzle_Y(0.5, np.deg2rad(80))
    am.compare_2d_y_slices()
    # am.compare_2d_y_slices()