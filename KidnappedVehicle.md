# UnscentedKF and Catch_Runaway_Vehicle project
## GitHub Repo
[CarND-Unscented-Kalman-Filter-Project](https://github.com/QuantumCoherence/CarND-Unscented-Kalman-Filter-Project)

[CarND-Catch-Run-Away-Car-UKF](https://github.com/QuantumCoherence/CarND-Catch-Run-Away-Car-UKF)

This project was completed on Ubuntu 16.04.

For both github repo links, the ecbuild subfolder contains all ECLIPSE project files, including compiled binary.

The build subfolder contains all the binaries and the makefile for compiling the project at the prompt with no formal IDE. 

**Compiling on your system: CMakelists.txt**

Use the CMakeLists.txt in the repo root folder to create the make file and compile on your system.

## RMSE and NIS Result for the UKF project
The file rmse-nis.txt in the subfolder build contains the RMSE values for p_x, p_y, vx, vy and the nis values when run with the Dataset 1.

***Here below a plot of the RMSE values***
![RMSE Plot](https://github.com/QuantumCoherence/CarND-Unscented-Kalman-Filter-Project/blob/master/RMSE_plot.jpeg?raw=true)

***The file UKF-result.jpeg is a screendump of the simulator at the end of the Dataset 1 run.***

![Simulator Screen Dump](https://github.com/QuantumCoherence/CarND-Unscented-Kalman-Filter-Project/blob/master/UKF-result.jpeg?raw=true)


***The NIS_plot.jpeg shows the plot of the NIS values***
![NIS plot](https://github.com/QuantumCoherence/CarND-Unscented-Kalman-Filter-Project/blob/master/NIS_plot.jpeg?raw=true)


## UKF convergence tuning and NIS

**Paramters used for tuning**

	1. Initial State Vector Values
    2. Process Covariance Initial Values 
    3. Angular and linera Acceleration noise values

*Initial State Vector Values*

Initial values do help keeping the initial error low. It's easy to guess that initial values for psi and psid will be close to zero (the car start driving straight). FOr the speed V, the intiial value is set to 5, which is pretty much the constant speed of the vehicle. px and px are taken from the first measurement.

*Process Covariance Initial Values*

Well selected initial process covariance values help avoiding the likelihood of an unstable and longer convergence. Intuitively the intial value for the postion given by the sensor measurment, can be expected to have a good level of accuracy, hence a high confidence can be used as starting value. The intial values for speed, spi and psid were selected by a trial and error process, starting with relatively good confidence values. If the selected level of confidence is too high for the parameter, the outcome will not convegre smoothly, but rather first accumulate errors, then start converging (assuming the noise density levels for the random components have been selected close enough not to disrupt the whole process). Aftre trial and error, these are the values selected: px: 0.101, py: 0.11, v: 0.231, psi: 3, psid: 3.


*Acceleration Noise values*
The forward speed of the car is essentially constant, we can therefore expect little acceleration variation. Starting with a value of 1 for stda, that is, expecting longitudinal acceleration to be within +/-2m/s2, with trial and error the end value was selected at stda=0.84.
The std_yawdd was less intuitive to guess: the vehicle takes about 5 second to complete each portion of the 8 figure, hence 2 PI/5 =~ 1.25 Rad/s speed. This changes rapdily during the transition from left to right turn, in about 2-3 seconds(not measured, just visually guessed), hence the max yaw acceleration should roughly be around 0.8 Rad/s2 (from -1.25 Rad/s to +1.25Rad/s, it's 2.5 Rad/s speed range change in ~3 seconds, that is it's 0.8 Rad/s change each second, hence 0.8 Rad/s2s std_yawdd noise density). After trial and error the final value for std_yawdd was selcted at 0.55 Rad/s2, so close enough ;-)


**Remarks**

The non linear model of the vehicle motion allows for a much better tracking of the vehicle speed during the turns compared to the EKF, which only used a non linear model for the radar sensor. The positional error did not improve signficantly, while the error in speed estimation is 50% lower.

**Video**
[UKF Video](https://github.com/QuantumCoherence/CarND-Unscented-Kalman-Filter-Project/blob/master/vokoscreen-2018-04-23_13-40-23.mkv)

