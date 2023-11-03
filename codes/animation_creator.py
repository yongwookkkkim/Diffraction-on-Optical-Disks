import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from conversion import *
import pandas as pd
from PIL import Image

#constants
Lambda = 1500e-9
w = 600e-9
h = 110e-9
visible_low = 400e-9
visible_high = 750e-9
r_s = 70.8e-2
r_o = 35e-2
r_inner = 2e-2
r_outer = 6e-2
alpha_exp = 83.5 * np.pi / 180
phi_exp = 23 * np.pi / 180
thick = 1.2e-3

#23.0cm, 40.6deg -> 17.5, 14.5; 36
#26, 14.5 -> 29deg, 29.8cm
#34.5, 14.5 + tilting -> 21deg, 37.4cm; 19, 35

#case 1: 22.7cm, 38deg
#case 2: 29.8cm, 29deg
#case 3: 35cm, 23deg

def n_pc(wv):
    wv_in = wv*1e9
    return 1.51334706 + 1.01457072e+02/wv_in - 6.91264454e+04/wv_in**2 + 1.98566840e+07/wv_in**3

#regress the initial SPD
df = pd.read_excel('D:\\OneDrive\\Desktop\\IYPT2023_backup\\colored line\\experiment analysis\\initial_spd_spectroscopy.xlsx', sheet_name='Sheet1')
df = df.to_numpy()

x_i0 = df[:,0]
y_i0 = df[:,1]

#initial spd function
def i_0(wavelength):
    return np.interp(wavelength * 1e9, x_i0, y_i0)

# Create a list to store frames
frames = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#calibration of angles
def calibrate_front(alpha, theta, phi, radial_dist):
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

        return [(z_s - thick) * np.tan(alpha_prime) + thick * np.sin(alpha_prime) / np.sqrt(1.64**2 - np.sin(alpha_prime)**2) - s2d,
                (z_o - thick) * np.tan(phi_prime) + thick * np.sin(phi_prime) / np.sqrt(1.64**2 - np.sin(phi_prime)**2) - o2d,
                theta_prime - np.arctan2(y_o - y_d, x_o - x_d) + np.arctan2(y_s - y_d, x_s - x_d),
                theta_dist - theta_s - np.arctan2(y_s - y_d, x_s - x_d),
                theta_o + theta_s - theta_prime,
                np.sin(alpha_prime) * np.sin(theta_s) - np.sin(phi_prime) * np.sin(theta_o)]
    
    res1 = fsolve(equation1, [alpha, theta, phi, 0, theta / 2, theta / 2])
    return res1

def calibrate_back(alpha, theta, phi, radial_dist):
    x_o = r_o * np.sin(phi) * np.cos(theta)
    y_o = r_o * np.sin(phi) * np.sin(theta)
    z_o = r_o * np.cos(phi)
    x_s = r_s * np.sin(alpha)
    y_s = 0
    z_s = r_s * np.cos(alpha)

    def equation2(p):
        alpha_prime, theta_prime, phi_prime, theta_dist, theta_o, theta_s = p
        x_d = radial_dist * np.cos(theta_dist)
        y_d = radial_dist * np.sin(theta_dist)

        o2s = np.sqrt((x_s - x_o)**2 + (y_s - y_o)**2)
        o2d = np.sqrt((x_d - x_o)**2 + (y_d - y_o)**2)
        s2d = np.sqrt((x_d - x_s)**2 + (y_d - y_s)**2)
        return [
            (z_s - thick) * np.tan(alpha_prime) + thick * np.sin(alpha_prime) / np.sqrt(1.64**2 - np.sin(alpha_prime)**2) - s2d,
            (z_o - thick) * np.tan(phi_prime) + thick * np.sin(phi_prime) / np.sqrt(1.64**2 - np.sin(phi_prime)**2) - o2d,
            theta_prime - np.arctan2(y_o - y_d, x_o - x_d) + np.arctan2(y_s - y_d, x_s - x_d),
            theta_dist - np.pi - theta_s - np.arctan2(y_s - y_d, x_s - x_d),
            theta_o + theta_s - theta_prime,
            np.sin(alpha_prime) * np.sin(theta_s) - np.sin(phi_prime) * np.sin(theta_o)]
    
    res2 = fsolve(equation2, [alpha, theta, phi, np.pi, theta / 2, theta / 2])
    return res2

#diffraction envelope
def diffraction_envelope_m(wavelength, alpha_prime, phi_prime, theta_o, theta_s):
    alpha = np.arcsin(np.sin(alpha_prime))
    phi = np.arcsin(np.sin(phi_prime))
    opl = n_pc(wavelength) * h * (1 / np.sqrt(n_pc(wavelength)**2 - np.sin(phi)**2) + 1/ np.sqrt(n_pc(wavelength)**2 - np.sin(alpha)**2)) - h * np.sin(phi_prime) * np.sin(theta_o) * (np.sin(phi) * np.sin(theta_o) / np.sqrt(n_pc(wavelength)**2 - np.sin(phi)**2) + np.sin(alpha) * np.sin(theta_s) / np.sqrt(n_pc(wavelength)**2 - np.sin(alpha)**2)) / n_pc(wavelength) - h * np.sin(phi_prime) * np.cos(theta_o) * (np.sin(phi) * np.cos(theta_o) / np.sqrt(n_pc(wavelength)**2 - np.sin(phi)**2) - np.sin(alpha) * np.cos(theta_s) / np.sqrt(n_pc(wavelength)**2 - np.sin(alpha)**2)) / n_pc(wavelength)
    delta_x_ref = h * np.sin(phi) * np.cos(theta_o) / np.sqrt(n_pc(wavelength)**2 - np.sin(phi)**2)
    delta_x_inc = h * np.sin(alpha) * np.cos(theta_s) / np.sqrt(n_pc(wavelength)**2 - np.sin(alpha)**2)
    temp_min = np.min((delta_x_inc, delta_x_ref, 0))
    temp_max = np.max((delta_x_inc, delta_x_ref, 0))
    if delta_x_ref < 0:
        delta_x = 0#(temp_min + temp_max) / 2
        delta_w = 0#abs(temp_max) + abs(temp_min)
    else:
        delta_x = 0#(temp_min + temp_max) / 2
        delta_w = 0#abs(temp_max) + abs(temp_min)
    sinc_track = np.sinc((Lambda - w) * (np.sin(alpha) * np.cos(theta_s) + np.sin(phi) * np.cos(theta_o)) / wavelength)
    sinc_pit = np.sinc((w - delta_w) * (np.sin(alpha) * np.cos(theta_s) + np.sin(phi) * np.cos(theta_o)) / wavelength)
    return (w - delta_w)**2 * sinc_pit**2 + (Lambda - w)**2 * sinc_track**2 + 2 * (Lambda - w) * (w - delta_w) * sinc_track * sinc_pit * np.cos(np.pi * (-Lambda / 2 + delta_x) * (np.sin(alpha) * np.cos(theta_s) + np.sin(phi) * np.cos(theta_o)) / wavelength + 2 * np.pi * n_pc(wavelength) * opl / wavelength)
    
#create animation
thetas = np.arange(15, 181, 5)
thetas  = [np.pi * theta / 180 for theta in thetas]

frames = []

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for theta_exp in thetas:
    #delete previous frame
    ax.cla()

    for radius in np.arange(r_inner, r_outer, 0.0025):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        z = np.zeros_like(theta)
        ax.plot(x, y, z, color='lightgray', linewidth=4)

    radius = np.arange(r_inner, r_outer, 1e-3)
    for r in radius:
        alpha_prime_front, theta_prime_front, phi_prime_front, theta_dist_front, theta_o_front, theta_s_front = calibrate_front(alpha_exp, theta_exp, phi_exp, r)
        color_front = [0, 0, 0]
        for m in range(1, 10):
            max_wv = Lambda * (np.sin(alpha_prime_front) * np.cos(theta_s_front) + np.sin(phi_prime_front) * np.cos(theta_o_front)) / m
            if max_wv < visible_high and max_wv > visible_low:
                if abs(r - 0.03) < 1e-6:
                    print(f'front: theta={round(theta_exp * 180 / np.pi)}, m={m}, wv={round(max_wv * 1e9)}nm, envelope={diffraction_envelope_m(max_wv, alpha_prime_front, phi_prime_front, theta_o_front, theta_s_front)}, alpha={round(alpha_prime_front * 180 / np.pi, 3)}, theta={round(theta_prime_front * 180 / np.pi, 3)}, phi={round(phi_prime_front * 180 / np.pi, 3)}, theta_s={round(theta_s_front * 180 / np.pi, 3)}, theta_o={round(theta_o_front * 180 / np.pi, 3)}')
                color_front += wavelength2xyz(max_wv) * i_0(max_wv) * diffraction_envelope_m(max_wv, alpha_prime_front, phi_prime_front, theta_o_front, theta_s_front)
        if color_front[0] > 0 or color_front[1] > 0 or color_front[2] > 0:
            color_front /= np.linalg.norm(color_front)
        color_front = xyz2rgb(*color_front)
        theta1 = np.linspace(theta_dist_front - 0.01, theta_dist_front + 0.01, 20)
        x1 = np.cos(theta1) * r
        y1 = np.sin(theta1) * r
        z1 = np.zeros_like(theta1)
        ax.plot(x1, y1, z1, color=color_front, linewidth=4)

        alpha_prime_back, theta_prime_back, phi_prime_back, theta_dist_back, theta_o_back, theta_s_back = calibrate_back(alpha_exp, theta_exp, phi_exp, r)
        color_back = [0, 0, 0]
        for m in range(1, 10):
            max_wv = Lambda * (np.sin(alpha_prime_back) * np.cos(theta_s_back) + np.sin(phi_prime_back) * np.cos(theta_o_back)) / m
            if max_wv < visible_high and max_wv > visible_low:
                if abs(r - 0.055) < 1e-6:
                    print(f'back: theta={round(theta_exp * 180 / np.pi)}, m={m}, wv={round(max_wv * 1e9)}nm, envelope={diffraction_envelope_m(max_wv, alpha_prime_back, phi_prime_back, theta_o_back, theta_s_back)}, alpha={round(alpha_prime_back * 180 / np.pi, 3)}, theta={round(theta_prime_back * 180 / np.pi, 3)}, phi={round(phi_prime_back * 180 / np.pi, 3)}, theta_s={round(theta_s_back * 180 / np.pi, 3)}, theta_o={round(theta_o_back * 180 / np.pi, 3)}, place={round(theta_dist_back * 180 / np.pi, 3)}')
                color_back += wavelength2xyz(max_wv) * i_0(max_wv) * diffraction_envelope_m(max_wv, alpha_prime_back, phi_prime_back, theta_o_back, theta_s_back)
        if color_back[0] > 0 or color_back[1] > 0 or color_back[2] > 0:
            color_back /= np.linalg.norm(color_back)
        color_back = xyz2rgb(*color_back)
        theta2 = np.linspace(theta_dist_back - 0.01, theta_dist_back + 0.01, 20)
        x2 = np.cos(theta2) * r
        y2 = np.sin(theta2) * r
        z2 = np.zeros_like(theta2)
        ax.plot(x2, y2, z2, color=color_back, linewidth=4)
    
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

    ax.view_init(elev=90 - phi_exp * 180 / np.pi, azim = theta_exp * 180 / np.pi)

    #if abs(theta_exp - 13 * np.pi / 180) < 1e-6:
    #    plt.show()

    # Save the current frame
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(Image.fromarray(frame))

# Save the frames as a GIF
frames[0].save('D:\\OneDrive\\Desktop\\2023-2\\CD_paper\\color_vids\\simu_cd3.gif', format='GIF',
               append_images=frames[1:],
               save_all=True, duration=400, loop=0)