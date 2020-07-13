# Preliminary stuff

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
from nupic.algorithms.temporal_memory import TemporalMemory
from nupic.algorithms.sdr_classifier_factory import SDRClassifierFactory
from nupic.algorithms.anomaly import Anomaly
from nupic.algorithms.anomaly_likelihood import AnomalyLikelihood


# Colormap for TM visualization later
# Not active, Active, Predicted, Winner
#      black,   grey,    yellow,   cyan
colors = [(0,0,0), (0.5,0.5,0.5), (1,1,0), (0,1,1)]
tm_cmap = LinearSegmentedColormap.from_list('tm', colors, N=4)

_INPUT_FILE_PATH = "PP_M0530031_PP_M0520117_tension_vector.csv"


_NUM_RECORDS = 5000

print("Creating encoders and encodings...")

scalarEncoder = RandomDistributedScalarEncoder(resolution=0.5)

encodingWidth =  scalarEncoder.getWidth()*2


foco1Bits = np.zeros(scalarEncoder.getWidth())
foco2Bits = np.zeros(scalarEncoder.getWidth())

activeColumns = np.zeros(2048)

print("Initializing Spatial Pooler...")
sp = SpatialPooler(
    inputDimensions = (encodingWidth,),
    columnDimensions=(2048,),
    potentialRadius = encodingWidth, # global coverage
    potentialPct = 0.85,
    globalInhibition = True,
    localAreaDensity = -1.0,
    numActiveColumnsPerInhArea = 40.0, # %2 sparsity
    stimulusThreshold = 0,
    synPermInactiveDec = 0.005,
    synPermActiveInc = 0.04,
    synPermConnected = 0.1,
    minPctOverlapDutyCycle = 0.001,
    dutyCyclePeriod = 100,
    boostStrength = 3.0, # boost a little
    seed = 42,
    spVerbosity = 0,
    wrapAround = False
)

print("Letting the spatial pooler learn the dataspace...")
n_train_for_SP = 3000
with open(_INPUT_FILE_PATH, 'r') as fin:
    reader = csv.reader(fin)

    for count, record in enumerate(reader):
        if count >= n_train_for_SP: break
        foco1 = float(record[0])
        foco2 = float(record[1])
        
        scalarEncoder.encodeIntoArray(foco1, foco1Bits)
	scalarEncoder.encodeIntoArray(foco2, foco2Bits)

        encoding = np.concatenate([foco1Bits, foco2Bits])
        sp.compute(encoding, True, activeColumns)

print("...all done. Turning off boosting")
sp.setBoostStrength(0.0)

print("Initializing Temporal Memory...")
tm = TemporalMemory(
    columnDimensions = sp.getColumnDimensions(),
    cellsPerColumn = 16,
    activationThreshold = 13,
    initialPermanence = 0.55,
    connectedPermanence = 0.5,
    minThreshold = 10,
    maxNewSynapseCount = 20,
    permanenceIncrement = 0.1,
    permanenceDecrement = 0.1,
    predictedSegmentDecrement = 0.0,
    seed = 42,
    maxSegmentsPerCell = 128,
    maxSynapsesPerSegment = 40
)


print("Initializing classification and anomaly calculators")
classifier = SDRClassifierFactory.create(steps=[1], alpha=0.01)
predictions = np.zeros(_NUM_RECORDS+2)

aScore = Anomaly(slidingWindowSize=25)
aLikely = AnomalyLikelihood(learningPeriod=600, historicWindowSize=313)
ascores = np.zeros(_NUM_RECORDS+1)
alhoods = np.zeros(_NUM_RECORDS+1)
alloghoods = np.zeros(_NUM_RECORDS+1)

with open(_INPUT_FILE_PATH, "r") as fin:
    reader = csv.reader(fin)

    print("Beginning record processing...")
    for count, record in enumerate(reader):

        if count > _NUM_RECORDS: break
        if count % 500 == 0:
            print("...processed {0:4d}/{1} records...".format(count, _NUM_RECORDS))

        foco1 = float(record[0])
        foco2 = float(record[1])

        scalarEncoder.encodeIntoArray(foco1, foco1Bits)
	scalarEncoder.encodeIntoArray(foco2, foco2Bits)

        encoding = np.concatenate([foco1Bits, foco2Bits])
        sp.compute(encoding, False, activeColumns)

        activeColumnIndices = np.nonzero(activeColumns)[0]
        predictedColumnIndices = [tm.columnForCell(cell) for cell in tm.getPredictiveCells()]
        tm.compute(activeColumnIndices, learn=True)
        ascores[count] = aScore.compute(activeColumnIndices, predictedColumnIndices)
        alhoods[count] = aLikely.anomalyProbability(foco2, ascores[count])
        alloghoods[count] = aLikely.computeLogLikelihood(alhoods[count])

        bucketIdx = scalarEncoder.getBucketIndices(foco2)[0]
        classifierResult = classifier.compute(recordNum=count, patternNZ=tm.getActiveCells(),
            classification={"bucketIdx": bucketIdx,"actValue": foco2},
            learn=count > 700, # let classifier learn once TM has learned a little
            infer=True
        )

        predConf, predictions[count+1] = sorted(zip(classifierResult[1], classifierResult["actualValues"]), reverse=True)[0]

foco2 = np.loadtxt(_INPUT_FILE_PATH,delimiter=',',skiprows=3,usecols=1)

plt.figure(figsize=(15,3))
plt.plot(foco2)
plt.xlim(0, 5000)
plt.xticks(range(0,4401,250))
plt.show()


possible_anomaly_indices = np.where(alloghoods >= 0.5)[0]
y = np.array([100,80,100]*len(possible_anomaly_indices))
possible_anomaly_indices = np.sort(np.concatenate((possible_anomaly_indices, possible_anomaly_indices, possible_anomaly_indices)))
fig, ax = plt.subplots(3,1,figsize=(15,10))
ax[0].plot(alhoods, label='normal')
ax[0].plot(alloghoods, label='log')
ax[1].plot(ascores)
ax[2].plot(foco2, label='actual',ls=':')
ax[2].plot(predictions, label='predicted')

ax[0].grid()
for a in ax:
    a.set_xlim((0,4400))
    a.set_xticks(range(0,4401,250))

ax[2].set_ylim((115000,135000))
ax[0].legend()
ax[2].legend()
ax[0].set_ylabel("Anomaly\nLikelihoods")
ax[1].set_ylabel("Anomaly\nScores")
ax[2].set_ylabel("Signal")
plt.show()


