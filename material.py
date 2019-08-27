import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from scipy.optimize import least_squares

# elements = np.array(['H', 'O', 'C', 'N', 'Cl', 'Ca', 'I', 'Si', 'B', 'Na', 'Mg', 'Fe'])
#
# SolidWater = np.array([0.0841, 0.1849, 0.6697, 0.0216, 0.0013, 0.0143, 0.0000, 0.0108, 0.0005, 0.0017, 0.0110, 0.0000])
# Iodine5mg = np.array([0.0837, 0.1839, 0.6668, 0.0214, 0.0013, 0.0142, 0.0050, 0.0107, 0.0005, 0.0017, 0.0110, 0.0000])
# Iodine10mg = np.array([0.0832, 0.1824, 0.6643, 0.0212, 0.0013, 0.0140, 0.0099, 0.0106, 0.0005, 0.0017, 0.0109, 0.0000])
# Iodine15mg = np.array([0.0827, 0.1809, 0.6618, 0.0211, 0.0013, 0.0139, 0.0148, 0.0105, 0.0005, 0.0017, 0.0108, 0.0000])
#
# print("should be 1.0: " + str(SolidWater.sum()))
# print("should be 1.0: " + str(Iodine5mg.sum()))
# print("should be 1.0: " + str(Iodine10mg.sum()))
# print("should be 1.0: " + str(Iodine15mg.sum()))
# # ASSUMPTION: iodineXmg is based on solid water with mixed in iodine.
# # extrapolation of iodine concentration should verify that
# # using euclidian distance (l2) as measure of similarity
#
# # 1. calculate delta vector
# deltaIod5mg = Iodine5mg - SolidWater
# print(["{0:.4}".format(i) for i in deltaIod5mg])
# # 2. calculating errors of extrapolation
# extraIod10mg = SolidWater + 2 * deltaIod5mg
# extraIod15mg = SolidWater + 3 * deltaIod5mg
# e10mg = np.sqrt(np.sum(np.square(np.subtract(Iodine10mg, extraIod10mg)), axis=0))
# e15mg = np.sqrt(np.sum(np.square(np.subtract(Iodine15mg, extraIod15mg)), axis=0))
# print('Differences in Iod10mg (percent): \n{}\n{}'.format(elements, np.subtract(Iodine10mg, extraIod10mg)))
# print('Differences in Iod15mg (percent): \n{}\n{}'.format(elements, np.subtract(Iodine15mg, extraIod15mg)))
#
# print("should be 1.0: " + str(extraIod10mg.sum()))
# print("should be 1.0: " + str(extraIod15mg.sum()))
#
# print("distance from extrapolation to real Iod10mg: " + str(e10mg))
# print("distance from extrapolation to real Iod15mg:  " + str(e15mg))
# # 3. compose calcium of water plus iodine
# calcium50mg = 0
def detectSizesFromFilename(filename):
    pa, filename = os.path.split(filename)
    fle, ext = os.path.splitext(filename)
    parts = fle.split("x")
    X = int(parts[0].split('_')[1])
    Y = int(parts[1])
    P = int(parts[2].split('.')[0])
    return X, Y, P

def readRawImage(path):
    X, Y, P = detectSizesFromFilename(path)
    dt = np.dtype('>f4')
    return np.fromfile(file=path, dtype=dt).newbyteorder().byteswap().reshape(P, Y, X)

class materialDecomposer:
    def __init__(self, folderpath, n=4):
        plt.ioff()
        self.l = readRawImage(glob.glob(os.path.join(folderpath, "POLY80*.raw")).pop())
        self.h = readRawImage(glob.glob(os.path.join(folderpath, "POLY120*.raw")).pop())
        self.waterCoeffs = self.fit(n)

        # fig, (ax0, ax1) = plt.subplots(1, 2)
        # ax0.imshow(self.l[0])
        # ax0.set_title("low energy @ 80 kV")
        # ax0.set_axis_off()
        # ax1.imshow(h[0])
        # ax1.set_title("high energy @ 120 kV")
        # ax1.set_axis_off()
        # plt.show()

    def transform(self, l, h):
        pass

    def compose(self, co):
        pass

    def fit(self, n):
        c = np.ones(n)


def scroll(ax, n):
    volume = ax.volume
    ax.index = (ax.index - n) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def process_scroll(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    scroll(ax, int(event.step))
    print(event.step)
    fig.canvas.draw()


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


class multi_slice_viewer:
    def __init__(self, volume, name="untitled"):
        remove_keymap_conflicts({'up', 'down'})
        fig, ax = plt.subplots()
        ax.volume = volume
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index])
        fig.suptitle(name, fontsize=14)
        fig.canvas.mpl_connect('scroll_event', process_scroll)
        self.fig = fig

    def show(self):
        self.fig.show()


if __name__ == "__main__":
    img = materialDecomposer("/home/mr/Documents/bachelor/data/simulation/0826183857_1")
    # plt.imshow(img[0])
    v = multi_slice_viewer(img, "POLY 120")
    v.show()

    input("to end display press any key")
