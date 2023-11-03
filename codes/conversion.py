import numpy as np
import pandas as pd

df = pd.read_csv('D:\\OneDrive\\Desktop\\IYPT2023_backup\\colored line\\codes\\xyz_color_matching_function.csv')
df = df.to_numpy()
cmf_wavelengths = df[:,0]
x_bar = df[:,1]
y_bar = df[:,2]
z_bar = df[:,3]

def wavelength2xyz(wavelength):
    x = np.interp(wavelength * 1e9, cmf_wavelengths, x_bar)
    y = np.interp(wavelength * 1e9, cmf_wavelengths, y_bar)
    z = np.interp(wavelength * 1e9, cmf_wavelengths, z_bar)
    return np.array([x, y, z])

def rgb2xyz(r, g, b):
    r_lin, g_lin, b_lin = sRGB2RGB_linear(r, g, b)
    return [0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin,
            0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin,
            0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin]

def xyz2xyY(x, y, z):
    return [x / (x + y + z), y / (x + y + z), y]

def sRGB2RGB_linear(r, g, b):
    def f(x):
        if x <= 0.04045:
            return x / 12.92
        else:
            return ((x + 0.055) / 1.055) ** 2.4
    return [f(r), f(g), f(b)]

def RGB_linear2sRGB(r_linear, g_linear, b_linear):
    def f(x):
        if x <= 0.0031308:
            return 12.92 * x
        else:
            return 1.055 * x ** (1/2.4) - 0.055
    return [f(r_linear), f(g_linear), f(b_linear)]

def xyz2rgb(x, y, z):
    r_lin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    r_lin = np.clip(r_lin, 0, 1)
    g_lin = np.clip(g_lin, 0, 1)
    b_lin = np.clip(b_lin, 0, 1)
    return RGB_linear2sRGB(r_lin, g_lin, b_lin)