# First the preliminary stuff. The modules that we are going to use.

from __future__ import division, print_function
import numpy as np
import csv
import subprocess
from datetime import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from nupic.encoders.date import DateEncoder
from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder
from nupic.algorithms.spatial_pooler import SpatialPooler


# Colormap for TM visualization later
# Not active, Active, Predicted, Winner
#          black,      grey,      yellow,   cyan

colors = [(0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 0), (0, 1, 1)]
tm_cmap = LinearSegmentedColormap.from_list('tm', colors, N=4)



timeOfDayEncoder = DateEncoder(timeOfDay=(21, 1))  # bucket [0] - width = 21 bucket [1] - radius = 1
weekendEncoder = DateEncoder(weekend=21)  # bucket width = 21
scalarEncoder = RandomDistributedScalarEncoder(resolution=0.88)  # bucket resolution = 0.88
# By default the other principal parameters are w=21, n =400.

# We are going to see how the encoder works, so lets encode fake data

record = ['7/2/10 0:00', '21.2']

# Convert date string into Python date object
dateString = dt.strptime(record[0], "%m/%d/%y %H:%M")
# Convert data value string into float
consuption = float(record[1])

# To encode I must set a bit array foreach encoder, so we create numpy arrays
timeOfDayBits = np.zeros(timeOfDayEncoder.getWidth())
weekendBits = np.zeros(weekendEncoder.getWidth())
consuptionBits = np.zeros(scalarEncoder.getWidth())

# Now we call the encoders to create bits representation foreach value
timeOfDayEncoder.encodeIntoArray(dateString, timeOfDayBits)
weekendEncoder.encodeIntoArray(dateString, weekendBits)
scalarEncoder.encodeIntoArray(consuption, consuptionBits)


# Concatenate the arrays
# encoding = np.concatenate((timeOfDayBits, weekendBits, consuptionBits))

# np.set_printoptions(threshold=np.nan)
# print(encoding.astype('int16'))
# np.set_printoptions(threshold=1000)

# Plot the bit array in a way it is better to analyze it
# plt.figure(figsize=(15, 2))
# plt.plot(encoding)
# plt.show()


def encode(file_record):
    dateString = dt.strptime(file_record[0], '%m/%d/%y %H:%M')
    consumption = float(file_record[1])

    timeOfDayBits = np.zeros(timeOfDayEncoder.getWidth())
    weekendBits = np.zeros(weekendEncoder.getWidth())
    consumptionBits = np.zeros(scalarEncoder.getWidth())

    timeOfDayEncoder.encodeIntoArray(dateString, timeOfDayBits)
    weekendEncoder.encodeIntoArray(dateString, weekendBits)
    scalarEncoder.encodeIntoArray(consumption, consumptionBits)

    return np.concatenate((timeOfDayBits, weekendBits, consumptionBits))


encodingWidth = timeOfDayEncoder.getWidth() + weekendEncoder.getWidth() + scalarEncoder.getWidth()

sp = SpatialPooler(
    inputDimensions=(encodingWidth,),
    columnDimensions=(2048,),
    potentialRadius=encodingWidth,  # -> let every column see every input cell
    potentialPct=0.85,  # -> but use only a random 85% of them
    globalInhibition=True,
    localAreaDensity=-1.0,
    numActiveColumnsPerInhArea=40.0,  # this / total column number = sparsity (40/2048 ~ 2%)
    stimulusThreshold=0,
    synPermInactiveDec=0.005,  # I've set the synapse growth-to-degradation
    synPermActiveInc=0.04,  # ratio to a little less than 10
    synPermConnected=0.1,
    minPctOverlapDutyCycle=0.001,
    dutyCyclePeriod=100,
    boostStrength=0.0,
    seed=42,
    spVerbosity=0,
    wrapAround=False
)

# After creating the SpatialPooler Instance we can create columns via the compute function
# First we need to build an array which will receive the sp method
activeColumns = np.zeros(2048)


# sp.compute(encoding, True, activeColumns)
# activeColumnsIndices = np.nonzero(activeColumns)[0]  # Note that the [0] is because the sp is poli dimensional so
# we just define the first dimension
# print(activeColumnsIndices)


def showSDR(file_record):
    encoding = encode(file_record)

    activeCols = np.zeros(sp.getColumnDimensions())  # in that case 2048
    sp.compute(encoding, False, activeCols)
    nEN = int(np.math.ceil(encodingWidth ** 0.5))
    nSP = int(np.math.ceil(sp.getColumnDimensions()[0] ** 0.5))
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    imgEN = np.ones(nEN ** 2) * -1
    imgEN[:len(encoding)] = encoding
    imgSP = np.ones(nSP ** 2) * -1
    imgSP[:len(activeCols)] = activeCols
    ax[0].imshow(imgEN.reshape(nEN, nEN))
    ax[1].imshow(imgSP.reshape(nSP, nSP))
    for a in ax:
        a.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
    ax[0].set_title('Encoder output', fontsize=20)
    ax[1].set_title('SDR', fontsize=20)
    ax[0].set_ylabel("{0}  --  {1}".format(*file_record), fontsize=20)
    plt.show()


showSDR(['7/2/10 2:00', 4.7])