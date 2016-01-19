#HumPi
Using a Raspberry Pi to measure the frequency of the synchronous grid of continental europe

## Hardware requirements
* Raspberry Pi 2
* USB-Soundcard with a microphone or line input
* AC-Power supply with V_out smaller than ~10V.
* [Voltage divider](https://en.wikipedia.org/wiki/Voltage_divider) to
	30mV for microphone input, 1V RMS for line in
* [Phone connector](https://en.wikipedia.org/wiki/Phone_connector_%28audio%29)

## Software requirements
* Python2
* SciPy ([HowTo](http://wyolum.com/numpyscipymatplotlib-on-raspberry-pi/))
* [numexpr](https://github.com/pydata/numexpr) for 2x faster measurements
* alsaaudio and ntplib. You can install these via pip:
````
    $ pip install -r requirements.txt
````
==========================

### Version 0.5.1

Once started, HumPi captures the signal from an USB-Soundcard and calculates the frequency continuously by fitting a sine wave on the last second of 'sound'. Currently the data is stored to disk. In future versions it might get sent to the [netzsinus](https://github.com/netzsinus) project. 

### Install & Run
```
git clone https://github.com/gillhofer/HumPi.git
./HumPi/python/HumPi.py
```
If your ALSA Device is not found, use the script in the python directory to get your correct device.

## TODO
* Send data to the [netzsinus](https://github.com/netzsinus) project
* general improvements


