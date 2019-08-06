
import numpy as np

elements = np.array(['H', 'O', 'C', 'N', 'Cl', 'Ca', 'I', 'Si', 'B', 'Na', 'Mg', 'Fe'])

SolidWater =  np.array([0.0841, 0.1849, 0.6697, 0.0216, 0.0013, 0.0143, 0.0000, 0.0108, 0.0005, 0.0017, 0.0110, 0.0000])
Iodine5mg =   np.array([0.0837, 0.1839, 0.6668, 0.0214, 0.0013, 0.0142, 0.0050, 0.0107, 0.0005, 0.0017, 0.0110, 0.0000])
Iodine10mg =  np.array([0.0832, 0.1824, 0.6643, 0.0212, 0.0013, 0.0140, 0.0099, 0.0106, 0.0005, 0.0017, 0.0109, 0.0000])
Iodine15mg =  np.array([0.0827, 0.1809, 0.6618, 0.0211, 0.0013, 0.0139, 0.0148, 0.0105, 0.0005, 0.0017, 0.0108, 0.0000])

print("should be 1.0: " + str(SolidWater.sum()))
print("should be 1.0: " + str(Iodine5mg.sum()))
print("should be 1.0: " + str(Iodine10mg.sum()))
print("should be 1.0: " + str(Iodine15mg.sum()))
# ASSUMPTION: iodineXmg is based on solid water with mixed in iodine.
# extrapolation of iodine concentration should verify that
# using euclidian distance (l2) as measure of similarity

# 1. calculate delta vector
deltaIod5mg = Iodine5mg - SolidWater
print(["{0:.4}".format(i) for i in deltaIod5mg])
# 2. calculating errors of extrapolation
extraIod10mg = SolidWater + 2 * deltaIod5mg
extraIod15mg = SolidWater + 3 * deltaIod5mg
e10mg = np.sqrt(np.sum(np.square(np.subtract(Iodine10mg, extraIod10mg)), axis=0))
e15mg = np.sqrt(np.sum(np.square(np.subtract(Iodine15mg, extraIod15mg)), axis=0))
print('Differences in Iod10mg (percent): \n{}\n{}'.format(elements, np.subtract(Iodine10mg, extraIod10mg)))
print('Differences in Iod15mg (percent): \n{}\n{}'.format(elements, np.subtract(Iodine15mg, extraIod15mg)))

print("should be 1.0: " + str(extraIod10mg.sum()))
print("should be 1.0: " + str(extraIod15mg.sum()))

print("distance from extrapolation to real Iod10mg: " + str(e10mg))
print("distance from extrapolation to real Iod15mg:  " + str(e15mg))
# 3. compose calcium of water plus iodine
calcium50mg = 0

