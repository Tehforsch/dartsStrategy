import numpy as np
import bisect
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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
ALLOWSAVING = True

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
    gauss = multivariate_normal(mean=(0, 0), cov=np.array([[xSpread ** 2, 0],
                                                           [0, ySpread ** 2]]))
    xx, yy = getLinGrid()
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    gaussianGrid = gauss.pdf(xxyy).reshape((RESOLUTION, RESOLUTION))
    return convolve(gaussianGrid, dartBoard)

def getBestPosition(scoreGrid):
    index1D = np.argmax(scoreGrid)
    index2D = np.unravel_index(index1D, scoreGrid.shape)
    # Convoluted array is twice the size of the dartboard array / gaussian array - in the convoluted array, the dartboard starts at (RES/2, RES/2)
    return (index2D[1] - RESOLUTION / 2, index2D[0] - RESOLUTION / 2)

def plotHeatmap(data, filename=None):
    plt.clf()
    plt.imshow(data, cmap='inferno', interpolation='nearest')

    BOARD_COLOR = '#222222'

    ax = plt.gca()
    for radius in regions:
        circ = Circle((RESOLUTION, RESOLUTION),
                      radius * RESOLUTION / 2,
                      fill=False,
                      color=BOARD_COLOR,
                      linewidth=0.5)
        ax.add_patch(circ)

    for k in range(0, 20):
        phi = 2.0 * np.pi / 20 * k + np.pi / 20
        x1 = RESOLUTION + regions[2] * RESOLUTION / 2 * np.cos(phi)
        y1 = RESOLUTION + regions[2] * RESOLUTION / 2 * np.sin(phi)
        x2 = RESOLUTION + regions[-1] * RESOLUTION / 2 * np.cos(phi)
        y2 = RESOLUTION + regions[-1] * RESOLUTION / 2 * np.sin(phi)
        plt.plot([x1, x2], [y1, y2], '-', color=BOARD_COLOR, linewidth=0.5)

    linear_index = np.argmax(data)
    (y, x) = np.unravel_index(linear_index, data.shape)

    circ = Circle((x, y),
                  15.0,
                  fill=True,
                  color='#ff0000',
                  linewidth=0)
    ax.add_patch(circ)

    plotOrSave(filename)


def plotPosition(dartboard, positions, filename=None):
    plt.imshow(dartboard, cmap='hot', interpolation='nearest')
    sc = plt.scatter(positions[:,0], positions[:,1], c=positions[:,2], cmap=plt.cm.winter)
    plt.plot(positions[:,0], positions[:,1], 'b-')
    plt.colorbar(sc)
    plotOrSave(filename)

def plotOrSave(filename):
    if filename is None and ALLOWSAVING:
        plt.show()
    else:
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)

def getScaledSigmas(sigmas):
    return [sigma / totalRadius for sigma in sigmas]

dartBoard = getDartBoard()

# Create plot which shows optimal position as precision changes
for xyRatio in [1, 2, 4]:
    sigmas = np.logspace(-4, 1, num=50)
    scaledSigmas = getScaledSigmas(sigmas)
    positions = np.array([list(getBestPosition(getScoreGrid(sigma, sigma*xyRatio, dartBoard))) + [scaledSigma] for (sigma, scaledSigma) in zip(sigmas, scaledSigmas)])
    plotPosition(dartBoard, positions, filename="position{}.png".format(xyRatio))
    plt.clf()

sigmas = [0.001, 0.1, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0]
scaledSigmas = getScaledSigmas(sigmas)

# Show boards for different precisions
for (sigma, scaledSigma) in zip(sigmas, scaledSigmas):
    scoreGrid = getScoreGrid(scaledSigma, scaledSigma, dartBoard)
    plotHeatmap(scoreGrid, "board_{:03.0f}.png".format(sigma))
