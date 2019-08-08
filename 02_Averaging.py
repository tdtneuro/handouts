"""
# Averaging Example
Import stream and epoc data into Python using **read_block**\
Plot the average waveform around the epoc event using **epoc_filter**\  
Good for Evoked Potential detection
"""

"""
### Housekeeping

Import critial libraries
"""

import matplotlib.pyplot as plt  # standard Python plotting library
import numpy as np  # package for scientific computing, handles arrays and math

# import the primary functions from the tdt library only
from tdt import read_block, epoc_filter, download_demo_data

import handout # handout: exclude
import os # handout: exclude
fname = os.path.basename(__file__).replace('.py','') # handout: exclude
doc = handout.Handout(fname) # handout: exclude

"""
### Importing the Data
This example uses our [example data sets](https://www.tdt.com/files/examples/TDTExampleData.zip). To import your own data, replace BLOCK_PATH with the full path to your own data block.
 
In Synapse, you can find the block path in the database. Go to Menu > History. Find your block, then Right-Click > Copy path to clipboard.
"""

download_demo_data()
BLOCK_PATH = 'data/Algernon-180308-130351'

"""
### Set up the varibles for the data you want to extract.
We will extract channel 3 from the LFP1 stream data store, created by the Neural Stream Processor gizmo, and use our PulseGen epoc event ('PC0/') as our stimulus onset.
"""

REF_EPOC     = 'PC0/'
STREAM_STORE = 'LFP1'
ARTIFACT     = np.inf      # optionally set an artifact rejection level
CHANNEL      = 3
TRANGE       = [-0.3, 0.8] # [start time relative to epoc onset, window duration]

"""
### Now read the specified data from our block into a Python structure
"""

data = read_block(BLOCK_PATH, evtype=['epocs','scalars','streams'], channel=CHANNEL)

"""
## Use epoc_filter to extract data around our epoc event

Using the `t` parameter extracts data only from the time range around our epoc event. For stream events, the chunks of data are stored in a list.
"""

data = epoc_filter(data, 'PC0/', t=TRANGE)

"""
Optionally remove artifacts
"""

art1 = np.array([np.any(x>ARTIFACT)
                 for x in data.streams[STREAM_STORE].filtered], dtype=bool)
                 
art2 = np.array([np.any(x<-ARTIFACT)
                 for x in data.streams[STREAM_STORE].filtered], dtype=bool)

good = np.logical_not(art1) & np.logical_not(art2)

num_artifacts = np.sum(np.logical_not(good))
if num_artifacts == len(art1):
    raise Exception('all waveforms rejected as artifacts')

data.streams[STREAM_STORE].filtered = [data.streams[STREAM_STORE].filtered[i]
                                       for i in range(len(good)) if good[i]]

"""
Applying a time filter to a uniformly sampled signal means that the length of each segment could vary by one sample.  Let's find the minimum length so we can trim the excess off before calculating the mean.
"""

min_length = np.min([len(x) for x in data.streams[STREAM_STORE].filtered])
data.streams[STREAM_STORE].filtered = [x[:min_length]
                                       for x in data.streams[STREAM_STORE].filtered]

"""
Find the average signal
"""

all_signals = np.vstack(data.streams[STREAM_STORE].filtered)
mean_signal = np.mean(all_signals, axis=0)
std_signal = np.std(all_signals, axis=0)

"""
### Ready to plot
Create the time vector
"""

ts = TRANGE[0] + np.arange(0, min_length) / data.streams[STREAM_STORE].fs

"""
Plot all the signals as gray
"""

fig, ax1 = plt.subplots(1, 1, figsize=(8,5))
ax1.plot(ts, all_signals.T, color=(.85,.85,.85), linewidth=0.5)
doc.add_figure(fig) # handout: exclude
doc.show() # handout: exclude

"""
Plot vertical line at time=0
"""

ax1.plot([0, 0], [np.min(all_signals), np.max(all_signals)], color='r', linewidth=3)
doc.add_figure(fig) # handout: exclude
doc.show() # handout: exclude

"""
Plot the average signal
"""

ax1.plot(ts, mean_signal, color='b', linewidth=3)
doc.add_figure(fig) # handout: exclude
doc.show() # handout: exclude

"""
Plot the standard deviation bands
"""

ax1.plot(ts, mean_signal + std_signal, 'b--', ts, mean_signal - std_signal, 'b--')
doc.add_figure(fig) # handout: exclude
doc.show() # handout: exclude

"""
Finish up the plot
"""

ax1.axis('tight')
ax1.set_xlabel('Time, s',fontsize=12)
ax1.set_ylabel('V',fontsize=12)
ax1.set_title('{0} {1} Trials ({2} Artifacts Removed)'.format(
    STREAM_STORE,
    len(data.streams[STREAM_STORE].filtered),
    num_artifacts))
doc.add_figure(fig) # handout: exclude
doc.show() # handout: exclude

#index_file = os.path.join(fname, 'index.html') # handout: exclude
#os.system("start " + index_file) # handout: exclude