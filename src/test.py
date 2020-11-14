# Imports
import numpy as np
from skmultiflow.drift_detection import DDM
ddm = DDM(warning_level=0.1)
# Simulating a data stream as a normal distribution of 1's and 0's
data_stream = np.random.randint(2, size=2000)
# Changing the data concept from index 999 to 1500, simulating an
# increase in error rate
for i in range(2000):
    data_stream[i] = 1
# Adding stream elements to DDM and verifying if drift occurred

for i in range(1000, 1400):
    data_stream[i] = i%2
for i in range(2000):
    ddm.add_element(data_stream[i])
    if ddm.detected_warning_zone():
        print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    if ddm.detected_change():
        print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))