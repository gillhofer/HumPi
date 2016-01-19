#!/usr/bin/python2

from __future__ import print_function

import alsaaudio
import numpy as np
import array
import matplotlib.pyplot as plt


print(" ")
print("================================================================================")
print("Use me to find your ALSA Device number")
print("I will listen to all devices on your system and plot their current input signal.")
print("Use the device which best represents a sine wave.")
print("================================================================================")

devices = alsaaudio.pcms(alsaaudio.PCM_CAPTURE)
j = 1
for device in devices:
	try:
		print(j,"),  u'", device, "'", sep="")		
		recorder=alsaaudio.PCM(alsaaudio.PCM_CAPTURE,
                alsaaudio.PCM_NORMAL,
                device)
		recorder.setchannels(1)
		recorder.setrate(24000)
		recorder.setformat(alsaaudio.PCM_FORMAT_FLOAT_LE)  
		recorder.setperiodsize(512)

		buffer = array.array('f')
		for i in range(10):
   			buffer.fromstring(recorder.read()[1])
		data = np.array(buffer, dtype='f')
	
		plt.plot(data)
		plt.title(device) 
		plt.show()
	except: 
		print("Device", device, "led to an error")
	finally:
		j += 1

