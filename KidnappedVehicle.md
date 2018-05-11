# Kidnapped Vehicle project
## GitHub Repo
[CarND-Kidnapped-Vehicle-Project](https://github.com/QuantumCoherence/CarND-Kidnapped-Vehicle-Project)


This project was completed on Ubuntu 16.04.

For both github repo links, the ecbuild subfolder contains all ECLIPSE project files, including compiled binary.

The build subfolder contains all the binaries and the makefile for compiling the project at the prompt with no formal IDE. 

**Compiling on your system: CMakelists.txt**

Use the CMakeLists.txt in the repo root folder to create the make file and compile on your system.




## Particle Filter Coding Notes

*Parameters and Performance*

Besides the need for a best guess initial state, the only tunable paramters are the number of particles used, which is set at 50 (below 40, the accuracy perfomance is negatively affected) and  the noise density for position and yaw angle. 

The Resample algoirthm makes use of the discrete distribution random generator that comes standard with C++11. It simplifes the resampling algoirhtm, quite signficantly, but aslo might be a cause for lowered performance. The resample wheel algorithm discussed during Sebastian's class, is not exaclty uniformly distirbuted, , which seem to henance performace.
The noise simulation parameters for the yaw angle was increased to 0.04, those for X and Y to 0.32, for a better performance.
The filter achieves the goal within ~97 secs. The initial random distribution of the particles seems to have quite a significant impact. It appears not to be a symmetric distribution .. most of the time the filter complete below 97 seconds. In one occasion it failed by 0.06 seconds, and in one occasion ended by 81 seconds. System CPU time might be playing are role in these assymetric variations. 

**Video**
Download in ziped form, gunzip and play.
[Kidnapped Vehicle](https://github.com/QuantumCoherence/CarND-Kidnapped-Vehicle-Project/blob/master/vokoscreen-2018-05-10_21-48-25.mkv.gz)

