#!/usr/bin/python2

import alsaaudio
import numpy as np
import array
import matplotlib.pyplot as plt

devices = alsaaudio.pcms(alsaaudio.PCM_CAPTURE)
print(" ")
print("======================================")
print("Use this tool to find your ALSA Device")
print("I will listen to all devices on your system and plot their current input signal.")
print("Use the device which best represents a sine wave.")
print("======================================")

for device in devices:
	try:
		print "u'{}'".format(device)		
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

