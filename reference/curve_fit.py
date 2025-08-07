import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm


# for function f:
f_coeffs = np.array([0.80, 0.0, 0.06]) # https://www.desmos.com/calculator/wczixvz2zk
f = lambda sec_loss_param: np.dot(f_coeffs, np.array([sec_loss_param**i for i in range(3, -1, -1)]))

class AM:
    """Ainley and Mathieson profile loss coefficients"""

    def __init__(self) -> None:

        # for function Y_nozzle
        def f_nozzle_dec(sc_coeffs, beta_coeffs):
            def f_nozzle(sc, beta):
                return np.dot(sc_coeffs, np.array([sc**i for i in range(3, -1, -1)])) + \
                        np.dot(beta_coeffs, np.array([beta**i for i in range(3, -1, -1)]))
            return f_nozzle

        def func(xy, A_x, B_x, C_x, D_x, 
                A_y, B_y, C_y, D_y):
            x, y = xy # xy comes in as a tuple
            return np.dot(np.array([A_x, B_x, C_x, D_x]), np.array([x**3, x**2, x**1, x**0])) + \
                    np.dot(np.array([A_y, B_y, C_y, D_y]), np.array([y**3, y**2, y**1, y**0])) 

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

        # source: https://www.desmos.com/calculator/mtdyrs5nsb
        Yp_funcs = [lambda x: np.dot(np.array([0.00472746457763, 0.134583745183, -0.163697000305, 0.107366420974]),  np.array([x**i for i in range(3, -1, -1)])),  # z = 80 degrees
                lambda x: np.dot(np.array([-0.0414167573814, 0.223613959998, -0.238872125931, 0.121529411906]),  np.array([x**i for i in range(3, -1, -1)])),  # z = 75 degrees
                lambda x: np.dot(np.array([-0.0486578892528, 0.257992471753, -0.2955067698091, 0.135810367152]), np.array([x**i for i in range(3, -1, -1)])), # z = 70 degrees
                lambda x: np.dot(np.array([0.0206060273238, 0.108790013801, -0.218053636229, 0.12410152093]),    np.array([x**i for i in range(3, -1, -1)])),    # z = 65 degrees
                lambda x: np.dot(np.array([0.00127100059953, 0.14745239856, -0.252092613573, 0.13143542614]),    np.array([x**i for i in range(3, -1, -1)])),    # z = 60 degrees
                lambda x: np.dot(np.array([-0.0118351782982, 0.141873179622, -0.230092081748, 0.120869144074]),  np.array([x**i for i in range(3, -1, -1)])), # z = 50 degrees
                lambda x: np.dot(np.array([-0.0626302442791, 0.239070828964, -0.290072414112, 0.131547607304]),   np.array([x**i for i in range(3, -1, -1)]))] # z = 40 degrees
        Yp = np.array([[Yp_funcs[i](sc[i, j]) for j in range(len(sc[i]))] for i in range(len(sc))])
        self.Yp_funcs = Yp_funcs
        self.Yp = Yp

        popt, pcov = curve_fit(func, (sc.ravel(), beta.ravel()), Yp.ravel())
        print(f'sc coefficients: {popt[:4]}\n'
            f'beta coefficients (radians): {popt[4:]}')
        self.f_nozz = f_nozzle_dec(popt[:4], popt[4:])

    def compare_2d_y_slices(self):
        # cross section at 65 degrees

        for ang in np.deg2rad(np.array([80, 75, 70, 65, 60, 50, 40], dtype=float)):
            sc = np.linspace(0.30, 1.10)
            Y_p = [self.f_nozz(sc[i], ang) for i in range(len(sc))]
            plt.plot(sc, Y_p, color='r')
            plt.title(f'cross section {np.rad2deg(ang)} degrees')

        angs = np.array([80, 75, 70, 65, 60, 50, 40])
        for i in range(6):
            sc = np.linspace(0.30, 1.10)
            Y_p = [self.Yp_funcs[i](sc[j]) for j in range(len(sc))]
            plt.plot(sc, Y_p, color='g')
            plt.plot(sc, self.Yp[i], color='b')
            plt.title(f'cross section {angs[i]} degrees')
        plt.show()

    def plot_3d_y(self):
        plt.style.use('_mpl-gallery')
        # transform data

        # plot the 3d interpolation
        test_sc = np.linspace(0.30, 1.10)
        test_beta = np.linspace(np.deg2rad(80), np.deg2rad(40))
        test_sc, test_beta = np.meshgrid(test_sc, test_beta)

        test_Yp = np.array([[self.f_nozz(test_sc[i,j], test_beta[i,j]) for i in range(len(test_sc[0]))] for j in range(len(test_sc[1]))])

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(test_sc, test_beta, test_Yp)    # type: ignore

        # Plot the 2d polyfits
        ax.plot_surface(sc, beta, Yp, vmin=Yp.min() * 2, cmap=cm.Blues) # type: ignore
        ax.set(xticklabels=[],
            yticklabels=[],
            zticklabels=[])

        plt.show()


am = AM()
am.compare_2d_y_slices()