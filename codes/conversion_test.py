from conversion import *
wv = 600e-9
xyz = wavelength2xyz(wv)
print('xyz: ' + str(xyz))
rgb = xyz2rgb(*xyz)
print('rgb: ' + str(rgb))
#actual RGB = [255, 123, 0]