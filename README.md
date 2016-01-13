#HumPi
Using a Raspberry Pi to measure the frequency of the synchronous grid of continental europe

## Hardware requirements
* Raspberry Pi 2
* USB-Soundcard with a microphone or line input
* a [VoltageDivider](https://en.wikipedia.org/wiki/Voltage_divider) to 30mV
* a [Phone Connector](https://en.wikipedia.org/wiki/Phone_connector_%28audio%29)

## Software requirements
* SciPy ([HowTo](http://wyolum.com/numpyscipymatplotlib-on-raspberry-pi/))
* [numexpr](https://github.com/pydata/numexpr)
* pyaudio (`sudo apt-get install python-pyaudio` )

==========================

### Version 0.1

The demo within the python directory captures the signal from an usb-sound card and calculates the frequency.

###Currently working on
* Continuously record the signal while concurrently processing it .





