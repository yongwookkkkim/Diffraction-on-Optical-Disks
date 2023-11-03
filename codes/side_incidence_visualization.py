import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from conversion import *
import pandas as pd
from PIL import Image

#constants
Lambda = 740e-9
w = 320e-9
h = 120e-9
visible_low = 350e-9
visible_high = 750e-9
r_s = 70.8e-2
r_o = 22.7e-2
r_inner = 2e-2
r_outer = 6e-2
alpha_exp = 83.5 * np.pi / 180
phi_exp = 40.6 * np.pi / 180
thick = 1.2e-3
n_pc = 1.6

#case 1: 22.7cm, 38deg
#case 2: 29.8cm, 29deg
#case 3: 35cm, 23deg

#regress the initial SPD
df = pd.read_excel('D:\\OneDrive\\Desktop\\IYPT2023_backup\\colored line\\experiment analysis\\initial_spd_spectroscopy.xlsx', sheet_name='Sheet1')
df = df.to_numpy()

x_i0 = df[:,0]
y_i0 = df[:,1]

#initial spd function
def i_0(wavelength):
    return np.interp(wavelength * 1e9, x_i0, y_i0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#diffraction envelope
def diffraction_envelope_m(wavelength, alpha_twoprime, phi_twoprime):
    k = n_pc * (np.sin(phi_twoprime) - np.sin(alpha_twoprime)) / wavelength
    opl = 2 * h / np.sin(alpha_twoprime) - 2 * h * np.tan(alpha_twoprime) * np.sin(phi_twoprime)
    sinc_track = np.sinc(k * w / 2)
    sinc_pit = np.sinc(k * (Lambda - w) / 2)
    return (w)**2 * sinc_pit**2 + (Lambda - w)**2 * sinc_track**2 + 2 * (Lambda - w) * (w) * sinc_track * sinc_pit * np.cos(2 * np.pi * k * Lambda / 2 - n_pc * opl)

#create visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.cla()

#draw disk
for radius in np.arange(r_inner, r_outer, 0.0025):
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta) * radius
    y = np.sin(theta) * radius
    z = np.zeros_like(theta)
    ax.plot(x, y, z, color='lightgray', linewidth=4)


x_o = r_o * np.sin(phi_exp)
z_o = r_o * np.cos(phi_exp)
x_s = r_s * np.sin(alpha_exp)
z_s = r_s * np.cos(alpha_exp)

#draw the additional lines (farther)
#radial_dists = np.arange(0, thick * np.sin(alpha_twoprime_front), 3e-4)
alpha_prime = np.arctan((x_s - r_outer) / z_s)
alpha_twoprime = np.arccos(np.cos(alpha_prime) / n_pc)
radial_dists = np.arange(0, thick * np.tan(alpha_twoprime), 1e-4)
for radial_dist in radial_dists:
    phi_prime = np.arctan((x_o + r_outer - radial_dist) / z_o)
    phi_twoprime = np.arcsin(np.sin(phi_prime) / n_pc)
    color = [0, 0, 0]
    for m in range(1, 10):
        max_wv = -n_pc * Lambda * (np.sin(phi_twoprime) - np.sin(alpha_twoprime)) / m
        if max_wv < visible_high and max_wv > visible_low:
            color += wavelength2xyz(max_wv) * i_0(max_wv) * diffraction_envelope_m(max_wv, alpha_twoprime, phi_twoprime)
            print('front: ' + str(round(radial_dist*10000)) + 'e-4m   ' + str(round(max_wv * 1e9)) + 'nm    m=' + str(m) + '    envelope=' + str(diffraction_envelope_m(max_wv, alpha_twoprime, phi_twoprime)))
    if color[0] > 0 or color[1] > 0 or color[2] > 0:
        color /= np.linalg.norm(color)
    color = xyz2rgb(*color)
    thetas = np.linspace(- 0.01, 0.01, 20)
    xs = np.cos(thetas) * (r_outer - radial_dist)
    ys = np.sin(thetas) * (r_outer - radial_dist)
    zs = np.zeros_like(thetas)
    ax.plot(xs, ys, zs, color=color, linewidth=4)


#draw the additional lines (closer)
alpha_prime = np.arctan((x_s + 0.75e-2) / z_s)
alpha_twoprime = np.arccos(np.cos(alpha_prime) / n_pc)
radial_dists = np.arange(0, thick * np.tan(alpha_twoprime)-0.75e-2, 1e-4)
for radial_dist in radial_dists:
    phi_prime = np.arctan((x_o - r_inner - radial_dist) / z_o)
    phi_twoprime = np.arcsin(np.sin(phi_prime) / n_pc)
    color = [0, 0, 0]
    for m in range(1, 10):
        #print('closer; m=' + str(m) + '\t' + str(round(-n_pc * Lambda * (np.sin(phi_twoprime) - np.sin(alpha_twoprime)) / m * 1e9)))
        max_wv = -n_pc * Lambda * (np.sin(phi_twoprime) - np.sin(alpha_twoprime)) / m
        if max_wv < visible_high and max_wv > visible_low:
            color += wavelength2xyz(max_wv) * i_0(max_wv) * diffraction_envelope_m(max_wv, alpha_twoprime, phi_twoprime)
            print('back: ' + str(round(radial_dist*10000)) + 'e-4m   ' + str(round(max_wv * 1e9)) + 'nm m=' + str(m))
    if color[0] > 0 or color[1] > 0 or color[2] > 0:
        color /= np.linalg.norm(color)
    color = xyz2rgb(*color)
    thetas = np.linspace(np.pi - 0.01, np.pi + 0.01, 20)
    xs = np.cos(thetas) * (r_inner + radial_dist)
    ys = np.sin(thetas) * (r_inner + radial_dist)
    zs = np.zeros_like(thetas)
    ax.plot(xs, ys, zs, color=color, linewidth=4)
    
ax.set_xlim([-0.1, 0.1])
ax.set_ylim([-0.1, 0.1])
ax.set_zlim([0, 0.05])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.tick_params(axis='x',colors='white')
ax.tick_params(axis='y',colors='white')
ax.tick_params(axis='z',colors='white')
ax.set_facecolor((0, 0, 0, 1))
ax.xaxis.set_pane_color((0, 0, 0, 1))
ax.yaxis.set_pane_color((0, 0, 0, 1))
ax.zaxis.set_pane_color((0, 0, 0, 1))
ax.view_init(elev=90 - phi_exp * 180 / np.pi, azim = 180)
fig.canvas.draw()
plt.show()