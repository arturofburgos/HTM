import numpy as np
from nupic.encoders import ScalarEncoder

enc = ScalarEncoder(n=21, w=1, minval=2.5, maxval=97.5, clipInput=True, forced=True)

print "3 =  ", enc.encode(3)
print '4 =  ', enc.encode(4)
print '100= ', enc.encode(100)
print '1000=', enc.encode(1000)

from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder

rdse = RandomDistributedScalarEncoder(n=21, w=3, resolution=1, offset=2.5)

print '\n3 = ', rdse.encode(3)
print '1 = ', rdse.encode(1)

import datetime
from nupic.encoders.date import DateEncoder

de = DateEncoder(season=3)  

now = datetime.datetime.strptime('2020-05-11 14:57:35', '%Y-%m-%d %H:%M:%S')
print '\nnow = ', de.encode(now)
xmas = datetime.datetime.strptime('2020-12-25 13:05:25', '%Y-%m-%d %H:%M:%S')
print 'xmas =', de.encode(xmas)

from nupic.encoders.category import CategoryEncoder

categories = ('cat', 'dog', 'monkey', 'bird')
encoder = CategoryEncoder(w=3, categoryList=categories, forced=True)
cat = encoder.encode('cat')
dog = encoder.encode('dog')
monkey = encoder.encode('monkey')
bird = encoder.encode('bird')

print '\ncat = ', cat
print 'dog = ', dog
print 'monkey = ', monkey
print 'bird  = ', bird

print encoder.encode(None)
print encoder.encode('unknown')

print '\n'

print encoder.decode(cat)
print encoder.decode(monkey)

print '\n'

catdog = np.array([0,0,0,1,0,1,0,1,1,0,0,0,0,0,0])
print encoder.decode(catdog)
print '\n'

from nupic.algorithms.spatial_pooler import SpatialPooler
print SpatialPooler

print '\n'
print(len(cat))
print '\n'

sp = SpatialPooler(inputDimensions=(15,),
                   columnDimensions=(4,),
                   potentialRadius=15,
                   numActiveColumnsPerInhArea=1,
                   globalInhibition=True,
                   synPermActiveInc=0.03,
                   potentialPct=1.0,
                   stimulusThreshold=0,
                   seed = 97
                   )

for column in xrange(4):
    connected = np.zeros((15,), dtype="int")
    sp.getConnectedSynapses(column, connected)
    print connected

output = np.zeros((4,), dtype='int')
sp.compute(cat, learn=True, activeArray=output)
print output

print '\n'

for i in xrange(100):
    sp.compute(cat, learn=True, activeArray=output)

for column in xrange(4):
    connected = np.zeros((15,), dtype="int")
    sp.getConnectedSynapses(column, connected)
    print connected

output = np.zeros((4,), dtype='int')
sp.compute(cat, learn=True, activeArray=output)
print output
