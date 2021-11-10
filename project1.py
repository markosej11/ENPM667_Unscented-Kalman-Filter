import sys
import argparse
import numpy as np
import numpy.linalg.linalg as la
import matplotlib.pyplot as plt
from matplotlib import patches
# from scipy import stats
import matplotlib.transforms as transforms


class EKF:
    """
    Extended Kalman filter, for filtering input data within nonlinear systems.
    """
    def __init__(self, n, x, p, f=None, h=None, q=1.0, r=1.0):
        self.n = n
        self.f = (lambda xs: np.zeros((self.n, 1))) if f is None else f
        self.h = (lambda xs: np.array([[(xs[i, 0] if i == 0 else 0.0)] for i in range(self.n)])) if h is None else h
        if len(x) != self.n:
            raise ValueError("Initial condition \"x\" has length %d, but this filter has %d state variables!"
                             % (len(x), n))
        if len(p) != n or any([(len(p[i]) != n) for i in range(n)]):
            raise ValueError("Initial condition \"P\" does not have proper shape for filter having %d state variables!"
                             % n)
        self.x = np.copy(x) if issubclass(type(x), np.ndarray) else np.array(x)
        self.P = np.copy(p) if issubclass(type(p), np.ndarray) else np.array(p)
        self.Q = np.eye(self.n, dtype=np.float64) * q
        self.R = np.eye(self.n, dtype=np.float64) * r

    def predict(self):
        # A:  Jacobian of f(x) with respect to x
        A = np.zeros((self.n, self.n))
        h = 1.0e-7
        for i in range(self.n):
            for j in range(self.n):
                xc = self.x.astype(np.complex128)
                xc[j] += h*1j
                A[i, j] = np.imag(self.f(xc)[i, 0]) / h

        # Update x and P
        x = self.f(self.x)
        P = A @ self.P @ A.T + self.Q
        self.x = np.copy(x)
        self.P = np.copy(P)
        return x, P

    def measure(self, z):
        # H:  Jacobian of h(x) with respect to x
        H = np.zeros((self.n, self.n))
        h = 1.0e-7
        for i in range(self.n):
            for j in range(self.n):
                xc = np.copy(self.x).astype(np.complex128)
                xc[j] += h*1j
                H[i, j] = np.imag(self.h(xc)[i, 0]) / h

        # K: Kalman Gain
        K = self.P @ H.T @ la.inv(H @ self.P @ H.T + self.R)

        # Update x and P
        x = self.x + K @ (z - self.h(self.x))
        P = (np.eye(self.n) - K @ H) @ self.P
        self.x = np.copy(x)
        self.P = np.copy(P)
        return x, P


class UKF:
    def __init__(self, n, x, p, z, f=None, h=None, q=1.0, r=1.0):
        self.n = n
        m = z.size
        alpha = 0.001
        ki = 0
        beta = 2
        Lambda = (alpha**2) * (self.n + ki) - self.n
        c = self.n + Lambda
        weights_m = np.concatenate((np.array([Lambda/c]), np.full((2*self.n,), 0.5/c)), axis=0)
        weights_c = np.copy(weights_m)
        weights_c[0] = weights_c[0] + (1 - alpha**2 + beta)
        c = np.sqrt(c)

        # Calculate sigma points
        A = (c * la.cholesky(p)).T
        Y = np.tile(x, x.shape[0])
        sigma_points = np.concatenate((x, Y+A, Y-A), axis=1)

        # Kalman filter properties
        self.f = (lambda xs: np.zeros((self.n, 1))) if f is None else f
        self.h = (lambda xs: np.array([[(xs[i, 0] if i == 0 else 0.0)] for i in range(self.n)])) if h is None else h
        if len(x) != self.n:
            raise ValueError("Initial condition \"x\" has length %d, but this filter has %d state variables!"
                             % (len(x), n))
        if len(p) != n or any([(len(p[i]) != n) for i in range(n)]):
            raise ValueError("Initial condition \"P\" does not have proper shape for filter having %d state variables!"
                             % n)
        self.x = np.copy(x) if issubclass(type(x), np.ndarray) else np.array(x)
        self.P = np.copy(p) if issubclass(type(p), np.ndarray) else np.array(p)
        self.Q = np.eye(self.n, dtype=np.float64) * q
        self.R = np.eye(self.n, dtype=np.float64) * r

        # Prediction and measurement (located here instead of within separate functions)
        x1, X1, P1, X2 = self.unscented_transform(self.f, sigma_points, weights_m, weights_c, self.n, self.Q)
        z1, Z1, P2, Z2 = self.unscented_transform(self.h, X1, weights_m, weights_c, m, self.R)
        P12 = X2 @ np.diag(weights_c) @ Z2.T
        K = P12 @ la.inv(P2)
        self.x = np.expand_dims(x1 + K @ (z - z1), axis=1)
        self.P = P1 - K @ P12.T

    def predict(self):
        pass

    def measure(self, z):
        pass

    @staticmethod
    def unscented_transform(f, x, weights_m, weights_c, n, r):
        """
        Transformation of sigma points.

        Parameters
        ----------
        f: callable
            Nonlinear map.
        x: np.ndarray
            Array of sigma points to transform.
        weights_m: np.ndarray
            Weights assigned for mean
        weights_c: np.ndarray
            Weights assigned for covariance
        n: int
            Number of outputs from f
        r: np.array
            Additive covariance matrix

        Returns
        -------
        tuple
            Returns the following values:
            y: transformed mean
            Y: transformed sampling points
            p: transformed covariance matrix
            y1: transformed standard deviations

        """
        L = np.shape(x)[1]
        y = np.zeros((n,))
        Y = np.zeros((n, L))
        for k in range(L):
            Y[:, k] = f(x[:, k])
            y += weights_m[k] * Y[:, k]
        y1 = Y - np.repeat(np.expand_dims(y, axis=1), L, axis=1)
        p = y1 @ np.diag(weights_c) @ y1.T + r
        return y, Y, p, y1


def plot_figure_1():
    """
    References
    ----------
    Plotting covariance ellipses:
        https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    """
    # Create the random distribution around target coordinate (0, 1) in Fig. 1
    n = 2000  # Number of points
    expected_rad = 1.0
    expected_ang = np.pi/2.0
    noise_rad = 0.02
    noise_ang = 15 * np.pi / 180.0
    rad = np.random.randn(n) * noise_rad + expected_rad
    ang = np.random.randn(n) * noise_ang + expected_ang

    # Generate a more accurate ellipse
    x_arc = rad * np.cos(ang)
    y_arc = rad * np.sin(ang)
    # confidence = stats.norm.ppf(0.9, 2)
    cov = np.cov(x_arc, y_arc)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    xc1 = float(np.mean(x_arc))
    yc1 = float(np.mean(y_arc))
    x_ellipse_dia = 2.0 * np.sqrt(1 + pearson)
    y_ellipse_dia = 2.0 * np.sqrt(1 - pearson)
    try:
        eigenvalues, eigenvectors = la.eig(cov)
    except np.linalg.LinAlgError:
        return
    ellipse_angle = -np.arctan2(eigenvectors[0][1], eigenvectors[0][0])
    ellipse_plot1 = patches.Ellipse((0, 0), x_ellipse_dia, y_ellipse_dia, ellipse_angle, facecolor="#FFFFFF",
                                    edgecolor="#000000", fill=False, linestyle=(5, (5, 5)))
    transformation1 = transforms.Affine2D().rotate_deg(ellipse_angle * 180.0 / np.pi)
    transformation1 = transformation1.scale(np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])).translate(xc1, yc1)

    # Generate the linearized covariance ellipse
    x_arc = rad * np.cos(ang)
    y_arc = rad * np.sin(ang)
    # confidence = stats.norm.ppf(0.9, 2)
    xc2 = float(np.mean(rad) * np.cos(np.mean(ang)))
    yc2 = float(np.mean(rad) * np.sin(np.mean(ang)))
    cov = np.cov(x_arc, y_arc * 0.4)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    x_ellipse_dia = 2.0 * np.sqrt(1 + pearson)
    y_ellipse_dia = 2.0 * np.sqrt(1 - pearson)
    try:
        eigenvalues, eigenvectors = la.eig(cov)
    except np.linalg.LinAlgError:
        return
    ellipse_angle = -np.arctan2(eigenvectors[0][1], eigenvectors[0][0])
    ellipse_plot2 = patches.Ellipse((0, 0), x_ellipse_dia, y_ellipse_dia, ellipse_angle, facecolor="#FFFFFF",
                                    edgecolor="#000000", fill=False, linestyle=(5, (1, 5)), linewidth=2.0)
    transformation2 = transforms.Affine2D().rotate_deg(ellipse_angle * 180.0 / np.pi)
    transformation2 = transformation2.scale(np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])).translate(xc2, yc2)

    # Plot figure 1a
    fig_1 = plt.figure(figsize=(5.5, 8.5))
    plt.subplot(2, 1, 1)
    plt.xlim([-0.8, 0.8])
    plt.ylim([0.5, 1.2])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Nonlinear Transformation")
    plt.scatter(x_arc, y_arc, s=14.0, marker="+", color="#00000080", linewidth=1.0, label="Measured value")

    # Plot figure 1b
    ax2 = plt.subplot(2, 1, 2)
    plt.xlim([-0.5, 0.5])
    plt.ylim([0.88, 1.04])
    plt.title("Result from Linearization")
    plt.scatter(xc1, yc1, s=50.0, marker="x", color="#000000", linewidth=1.5, label="True mean")
    plt.scatter(xc2, yc2, s=50.0, marker="o", color="#000000", linewidth=1.5, label="True mean", facecolors="none")
    ellipse_plot1.set_transform(transformation1 + ax2.transData)
    ax2.add_patch(ellipse_plot1)
    ellipse_plot2.set_transform(transformation2 + ax2.transData)
    ax2.add_patch(ellipse_plot2)
    plt.show()
    plt.close(fig_1)


def plot_example():
    def sin_function_ekf(xs):
        """
        Sine function prediction
        """
        return np.array([[xs[0, 0] + xs[1, 0] * dt], [xs[1, 0] - xs[0, 0] * (omega ** 2) * dt]])

    def sin_function_ukf(xs):
        """
        Sine function prediction
        """
        return np.array([xs[0] + xs[1] * dt, xs[1] - xs[0] * (omega ** 2) * dt])

    def sin_measure(xs):
        """
        Sine function measurement
        """
        return np.array([xs[0], xs[1]])

    # Sine wave basic example
    dt = 0.05  # seconds
    t = np.arange(0, 40, dt)
    T = 10  # Period (seconds)
    omega = 2.0 * np.pi / T  # Angular frequency (rad/s)
    A = 20  # Amplitude
    B = 10  # Amplitude
    y = A * np.sin(omega * t) + B * np.cos(omega * t)
    ydot = A * omega * np.cos(omega * t) - B * omega * np.sin(omega * t)
    # y = np.array([(2.0 if i == 1 else 0.0) for i in range(len(t))])
    y_meas = np.zeros_like(y)
    y_approx = np.zeros_like(y)
    noise_p = A * 0.0005
    noise_m = A * 0.05
    Q = np.eye(2) * noise_p
    # noise_p = 0.0
    x = np.array([[y[0]], [A * omega]])
    P = np.eye(2) * 10000.0
    filter_type = "unscented"
    if filter_type == "extended":
        kf = EKF(2, x, np.eye(2) * 20000, f=sin_function_ekf, q=noise_p, r=noise_m)
    else:
        kf = None
    for i in range(len(t)):
        # Sense input data
        y_meas[i] = y[i] + np.random.randn() * noise_m
        y_dot_meas = ydot[i] + np.random.randn() * noise_m * 0.5

        # Run predict and measure steps
        if filter_type == "unscented":
            kf = UKF(2, x, P, np.array([y_meas[i], y_dot_meas]), f=sin_function_ukf, h=sin_measure, q=Q, r=noise_m)
            x = kf.x
            P = kf.P
        else:
            kf.predict()
            x, P = kf.measure(np.array([[y_meas[i]], [0.0]]))
        y_approx[i] = x[0, 0]
    # Plot basic example
    plt.figure()
    plt.plot(t, y, color="#000000", label="True value")
    plt.scatter(t, y_meas, s=14.0, marker="x", color="#FF3030", linewidth=0.75, label="Measured value")
    plt.plot(t, y_approx, color="#0060FF", linewidth=1.0, label="Filtered value")
    plt.title("Unscented Kalman Filter (sinusoidal wave example)")
    plt.xlabel("Time $t$ (seconds)")
    plt.ylabel("$y={0:d}\\sin(\\frac{{2\\pi}}{{{2:d}}} t) + {1:d}\\cos(\\frac{{2\\pi}}{{{2:d}}} t)$".format(A, B, T))
    plt.legend(loc="upper right")
    plt.show()
    plt.close()


def plot_figure_9():
    def beta(xs):
        return beta_o * np.exp(xs[4])

    def R(xs):
        """Distance from Earth"""
        return np.sqrt(xs[0]**2 + xs[1]**2)

    def V(xs):
        """Velocity"""
        return np.sqrt(xs[2]**2 + xs[3]**2)

    def force_d(xs):
        """Drag force"""
        return beta(xs) * np.exp((Ro - R(xs)) / Ho) * V(xs)

    def force_g(xs):
        """Force of gravity"""
        return -Gmo / (R(xs)**3)

    def f_state(xs):
        return np.array([xs[0] + xs[2]*dt,
                         xs[1] + xs[3]*dt,
                         xs[2] + (force_d(xs)*xs[2] + force_g(xs)*xs[0])*dt,
                         xs[3] + (force_d(xs)*xs[3] + force_g(xs)*xs[1])*dt,
                         xs[4] + 0.0])

    def h_meas(xs):
        return np.array([xs[0],
                         xs[1],
                         xs[2],
                         xs[3],
                         xs[4]])

    def get_radar_radius(xs):
        w1 = np.random.randn() * 0.001
        return np.sqrt((xs[0] - x_radar) ** 2 + (xs[1] - y_radar) ** 2) + w1

    def get_radar_angle(xs):
        w2 = np.random.randn() * 0.017
        return np.arctan2(xs[1] - y_radar, xs[0] - x_radar) + w2

    # Physical constants for forces
    beta_o = -0.59783
    Ho = 13.406
    Gmo = 3.986e5
    Ro = 6374
    y0 = np.array([[6500.4, 349.14, -1.8093, -6.7967, 0.6932]]).T
    Eo = 6378  # Radius of Earth

    # Radar location
    radar_angle = np.pi * 2.0018
    x_radar = Eo * np.cos(radar_angle)
    y_radar = Eo * np.sin(radar_angle)

    # Generate true data
    dt = 0.1  # seconds
    t = np.array([0.0])
    N = 0
    y = np.copy(y0)
    x_state = np.copy(y0)[:, 0]
    while np.sqrt(x_state[0]**2 + x_state[1]**2) > Eo:  # N-1
        y_next = f_state(x_state) + np.array([0.0,
                                              0.0,
                                              np.random.randn() * 2.4064e-5,
                                              np.random.randn() * 2.4064e-5,
                                              0.0])
        y = np.append(y, np.expand_dims(y_next, axis=1), axis=1)
        x_state = np.copy(y[:, N])
        t = np.append(t, np.array([t[-1] + dt]), axis=0)
        N += 1
    t_limit = next(i for i in range(N) if t[i] >= 200)

    # Run Kalman filter(s)
    y_meas = np.zeros_like(y)
    y_approx = np.zeros_like(y)
    x = np.copy(y0)
    x[4, 0] = 0.0
    P = np.eye(5) * 1.0e-6
    P[4, 4] = 1.0
    y_cov = np.zeros_like(y)
    for i in range(t_limit):
        y_meas[:, i] = y[:, i] + np.random.randn(5) * 0.003  # Sense input data
        ukf = UKF(5, x, P, y_meas[:, i], f=f_state, h=h_meas, q=2.1064e-5, r=0.001)  # Prediction and measurement
        x = ukf.x
        P = ukf.P
        y_approx[:, i] = x[:, 0]
        y_cov[:, i] = [P[j, j] for j in range(5)]

    # Plot figure 8 (true trajectory)
    plt.figure()
    plt.plot(y[0, :N], y[1, :N], linestyle=(5, (5, 5)),
             color="#000000", linewidth=1.0, label="Trajectory")
    theta = np.linspace(np.pi * 1.995, np.pi * 2.023, num=1000, endpoint=False)
    x_earth = Eo * np.cos(theta)
    y_earth = Eo * np.sin(theta)
    plt.plot(x_earth, y_earth, linestyle="-", color="#000000", linewidth=2.0, label="Surface of Earth")
    plt.scatter(x_radar, y_radar, s=70.0, marker="o",
                color="#000000", linewidth=1.0, label="Radar position", facecolors="none")
    plt.title("Spacecraft trajectory (from Fig. 8)")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend(loc="upper right")
    plt.show()
    plt.close()

    # Plot the Kalman filter position versus the true position
    plt.figure()
    plt.plot(y[0, :N], y[1, :N], linestyle="-",
             color="#000000", linewidth=1.0, label="True Trajectory")
    plt.plot(x_earth, y_earth, linestyle="-", color="#000000", linewidth=2.0, label="Surface of Earth")
    plt.scatter(x_radar, y_radar, s=70.0, marker="o",
                color="#000000", linewidth=1.0, label="Radar position", facecolors="none")
    plt.plot(y_approx[0, :t_limit], y_approx[1, :t_limit], linestyle="-", color="#FF0000", linewidth=1.0,
             label="Estimated trajectory")
    plt.scatter(y_approx[0, t_limit-1], y_approx[1, t_limit-1], s=50.0, marker="x",
                color="#FF0000", linewidth=1.0, label="Approximation end")
    plt.title("Kalman-filtered trajectory estimation")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend(loc="lower right")
    plt.show()
    plt.close()

    # Plot the Kalman filter x-position versus time
    plt.figure()
    plt.plot(t[:t_limit], y[0, :t_limit], linestyle="-", color="#000000", linewidth=1.0,
             label="True $x$-position")
    plt.plot(t[:t_limit], y_approx[0, :t_limit], linestyle="-", color="#FF0000", linewidth=1.0,
             label="Estimated $x$-position")
    plt.title("Kalman-filtered $x$-position estimation")
    plt.xlabel("time $t$")
    plt.ylabel("$x$")
    plt.legend(loc="upper right")
    plt.show()
    plt.close()

    # Plot figure 9 (mean-squared error and covariance)
    state_descriptions = [(0, "Position", "km"),
                          (2, "Velocity", "km/s"),
                          (4, "Aerodynamic Coefficient", None)]
    for state_description in state_descriptions:
        state, title, units = state_description
        plt.figure()
        mean_sqr_err = np.array([np.mean((y_approx[state, :i+1] - y[state, :i+1])**2) for i in range(N)])
        # estimated_cov = np.array([np.cov((y_approx[state, :i+1] - y[state, :i+1])**2) for i in range(N)])
        plt.plot(t[:t_limit], mean_sqr_err[:t_limit], linestyle="--", color="#000000", linewidth=2,
                 label="Mean-squared error")
        plt.plot(t[:t_limit], y_cov[state, :t_limit], linestyle="-.", color="#000000", linewidth=1, label="Covariance")
        # plt.plot(t[:pts], estimated_cov[:pts], linestyle="-.", color="#000000", linewidth=1.0, label="Covariance")
        plt.title("Mean-squared error and variance of $x_{%d}$ from Fig. 9" % (state + 1))
        plt.xlabel("Time $t$ (seconds)")
        plt.ylabel("%s ($x_{%d}$) variance%s" % (title, state+1, "" if units is None else (" $(%s)^{2}$" % units)))
        plt.yscale("log")
        plt.legend(loc="upper right")
        plt.show()
        plt.close()


def main(argv):
    # Parse system arguments
    parser = argparse.ArgumentParser()
    parser.parse_args(argv[1:])

    # Create basic example
    plot_example()

    # Create Figure 1 (a) and (b)
    plot_figure_1()

    # Create Figure 9
    plot_figure_9()


if __name__ == '__main__':
    main(sys.argv)
