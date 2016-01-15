#HumPi
Using a Raspberry Pi to measure the frequency of the synchronous grid of continental europe

## Hardware requirements
* Raspberry Pi 2
* USB-Soundcard with a microphone or line input
* AC-Power supply with V_out <  ~10V.
* [Voltage divider](https://en.wikipedia.org/wiki/Voltage_divider) to 30mV
* [Phone connector](https://en.wikipedia.org/wiki/Phone_connector_%28audio%29)

## Software requirements
* SciPy ([HowTo](http://wyolum.com/numpyscipymatplotlib-on-raspberry-pi/))
* [numexpr](https://github.com/pydata/numexpr) for 2x faster measurements
* alsaaudio (`sudo pip install pyalsaaudio` )

==========================

### Version 0.4

The demo within the `python` directory captures the signal and calculates the frequency on the fly.

## TODO
* allow to adjust the measurement frequency
* Send data to the [netzsinus](https://github.com/netzsinus) project
* general improvements




