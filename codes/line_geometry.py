import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

#constants
r_s = 31e-2
r_o = 23e-2
r_inner = 2e-2
r_outer = 6e-2
alpha_exp = 75.1 * np.pi / 180
theta_exp = 90 * np.pi / 180
phi_exp = 40.6 * np.pi / 180

#80.1 deg, 50.6cm

#calibration
def func1(alpha, theta, phi, radial_dist):
    x_o = r_o * np.sin(phi) * np.cos(theta)
    y_o = r_o * np.sin(phi) * np.sin(theta)
    z_o = r_o * np.cos(phi)
    x_s = r_s * np.sin(alpha)
    y_s = 0
    z_s = r_s * np.cos(alpha)

    def equation1(p):
        alpha_prime, theta_prime, phi_prime, theta_dist, theta_o, theta_s = p
        x_d = radial_dist * np.cos(theta_dist)
        y_d = radial_dist * np.sin(theta_dist)

        o2s = np.sqrt((x_s - x_o)**2 + (y_s - y_o)**2)
        o2d = np.sqrt((x_d - x_o)**2 + (y_d - y_o)**2)
        s2d = np.sqrt((x_d - x_s)**2 + (y_d - y_s)**2)
        return [alpha_prime - np.arctan(s2d / z_s),
                phi_prime - np.arctan(o2d / z_o),
                theta_prime - np.arctan2(y_o - y_d, x_o - x_d) + np.arctan2(y_s - y_d, x_s - x_d),
                theta_dist - theta_s - np.arctan2(y_s - y_d, x_s - x_d),
                theta_o + theta_s - theta_prime,
                np.sin(alpha_prime) * np.sin(theta_s) - np.sin(phi_prime) * np.sin(theta_o)]
    
    res1 = fsolve(equation1, [alpha, theta, phi, 0, theta / 2, theta / 2])

    return res1

def func2(alpha, theta, phi, radial_dist):
    x_o = r_o * np.sin(phi) * np.cos(theta)
    y_o = r_o * np.sin(phi) * np.sin(theta)
    z_o = r_o * np.cos(phi)
    x_s = r_s * np.sin(alpha)
    y_s = 0
    z_s = r_s * np.cos(alpha)

    def equation1(p):
        alpha_prime, theta_prime, phi_prime, theta_dist, theta_o, theta_s = p
        x_d = radial_dist * np.cos(theta_dist)
        y_d = radial_dist * np.sin(theta_dist)

        o2s = np.sqrt((x_s - x_o)**2 + (y_s - y_o)**2)
        o2d = np.sqrt((x_d - x_o)**2 + (y_d - y_o)**2)
        s2d = np.sqrt((x_d - x_s)**2 + (y_d - y_s)**2)
        return [alpha_prime - np.arctan(s2d / z_s),
                phi_prime - np.arctan(o2d / z_o),
                theta_prime - np.arctan2(y_o - y_d, x_o - x_d) + np.arctan2(y_s - y_d, x_s - x_d),
                theta_dist - theta_s - np.arctan2(y_s - y_d, x_s - x_d) - np.pi,
                theta_o + theta_s - theta_prime,
                np.sin(alpha_prime) * np.sin(theta_s) - np.sin(phi_prime) * np.sin(theta_o)]
    
    res2 = fsolve(equation1, [alpha, theta, phi, np.pi, theta / 2, theta / 2])
    return res2

r_list = np.linspace(r_inner, r_outer, 100)
theta_list = np.linspace(0, 2 * np.pi, 100)
R_list, Theta_list = np.meshgrid(r_list, theta_list)
x_list = R_list * np.cos(Theta_list)
y_list = R_list * np.sin(Theta_list)
theta_dist_list_1 = [func1(alpha_exp, theta_exp, phi_exp, r)[3] for r in r_list]
theta_dist_list_2 = [func2(alpha_exp, theta_exp, phi_exp, r)[3] for r in r_list]

plt.figure(figsize=(10, 10))
plt.polar(theta_dist_list_1, r_list, color='blue')
print(np.array(theta_dist_list_1) * 180 / np.pi)
plt.polar(theta_dist_list_2, r_list, color='blue')
plt.polar(theta_list, r_outer * np.ones(len(theta_list)), color='black')
plt.polar(theta_list, r_inner * np.ones(len(theta_list)), color='black')
plt.polar(np.zeros(len(r_list)), r_list, color='orange', label='source')
plt.polar(theta_exp * np.ones(len(r_list)), r_list, color='green', label='observer')
plt.legend()
plt.ylim(0, r_outer)
plt.show()

r = np.linspace(r_inner, r_outer, 100)
theta = np.linspace(0, 2 * np.pi, 100)
X = r * np.cos(theta)
Y = r * np.sin(theta)
x, y = np.meshgrid(X, Y)