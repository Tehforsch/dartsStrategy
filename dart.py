import numpy as np
import bisect
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.signal import convolve

# Radii on the board
#
# (see: https://upload.wikimedia.org/wikipedia/commons/0/00/Dartboard_Abmessungen.svg)
#
# Double bull 0  -  6.35 mm
# Bull        ...-  15.9 mm
# Single 1    ...-  99.0 mm
# Double      ...- 107.0 mm
# Single 2    ...- 162.0 mm
# Triple      ...- 170.0 mm
regions = [0, 6.35, 15.9, 99.0, 107.0, 162.0, 170.0]
totalRadius = max(regions)
regions = [x / totalRadius for x in regions]

RESOLUTION = 1000
SAVE = True


def score(x, y):
    r = np.sqrt(x**2 + y**2)
    if r > 1:
        return 0
    index = bisect.bisect(regions, r) - 1
    if index == 0:
        return 50
    elif index == 1:
        return 25
    elif index == 6:
        return 0
    multiplier = [1, 3, 1, 2][index-2]
    phi = np.arctan2(-y, x)
    if phi < 0:
        phi = phi + 2 * np.pi
    # phi = 0 is to the right, i.e. (1, 0)
    order = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]
    # the 20 begins at phi = -pi/20 (= 2pi / 20 / 2)
    phiIndex = int((phi + np.pi/20) / (2 * np.pi) * 20) % 20
    assert phiIndex >= 0
    return order[phiIndex] * multiplier


def getLinGrid():
    xs = np.linspace(-1, 1, RESOLUTION)
    return np.meshgrid(xs, xs)


def getDartBoard():
    xv, yv = getLinGrid()
    s = np.vectorize(score)
    return s(xv, yv)


def getScoreGrid(xSpread, ySpread, dartBoard):
    gauss = multivariate_normal(mean=(0, 0), cov=np.array([[xSpread ** 2.0, 0],
                                                           [0, ySpread ** 2.0]]))
    xx, yy = getLinGrid()
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    gaussianGrid = gauss.pdf(xxyy).reshape((RESOLUTION, RESOLUTION))
    # gaussianGrid = np.zeros((RESOLUTION, RESOLUTION))
    # gaussianGrid[RESOLUTION/2,RESOLUTION/2] = 1
    return convolve(gaussianGrid, dartBoard)


def getAvPoints(scoreGrid):
    return np.amax(scoreGrid)


def plotHeatmap(data, filename=None):
    plt.imshow(data, cmap='hot', interpolation='nearest')
    if filename is None or not SAVE:
        plt.show()
    else:
        plt.savefig(filename, dpi=300)


def movie():
    for x in range(1, 300):
        spread = 0.00001*x
        gauss = getScoreGrid(spread, spread, dartBoard)
        plotHeatmap(gauss, "movie/{:04d}.png".format(x))


dartBoard = getDartBoard()

for sigma in [0.1, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0]:
    sigmaScaled = sigma / totalRadius

    scoreGrid = getScoreGrid(sigmaScaled, sigmaScaled, dartBoard)

    avgScore = getAvPoints(scoreGrid)
    print("Average score at sigma = {:3.0f} mm: {:.0f}".format(sigma, avgScore))

    plotHeatmap(scoreGrid, "board_{:03.0f}.png".format(sigma))
