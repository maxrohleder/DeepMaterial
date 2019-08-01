import pyconrad.autoinit
import time
import numpy as np

file = '/home/mr/Documents/bachelor/data/simulation/0731125340_0/iodine/MAT_620x480x200.raw'

dt = np.dtype('>f4')
f = np.fromfile(file=file, dtype=dt)
f = f.reshape(200, 480, 620)

print(f.shape, f.dtype)
pyconrad.imshow(f, 'poly120')

f_le = f.newbyteorder().byteswap()
pyconrad.imshow(f_le, 'poly120le')


print(f_le.shape, f_le.dtype)

input()

pyconrad.close_all_windows()
pyconrad.terminate_pyconrad()

# print('Enjoy white noise!')
# for i in range(300):
#     noise = np.random.rand(200, 200)
#     pyconrad.imshow(noise, 'White noise', spacing=[200, 2], origin=[0, 2])
#     time.sleep(0.01)
#
# _ = pyconrad.ClassGetter()
#
# array = np.random.rand(4, 2, 3).astype(pyconrad.java_float_dtype)
# grid = _.NumericGrid.from_numpy(array)
#
#
#
# pyconrad.ij()
# pyconrad.close_all_windows()
# #pyconrad.terminate_pyconrad()