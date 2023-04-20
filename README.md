## Setup Instructions
To get the library working, first ensure that you have Fedml library installed on your system.
Once done and verified that your Fedml installation works, replace the fedml folder in site-packages/user packages of python environment on your system with this fedml folder.

The code in fedml/simulation/fedavg/fedavg_api.py is called for running fedavg simulations by the library. Replace the code inside this file with other files present in the same folder (having config as suggested in the file name) to get that config running in simulation.

For client selection method, make changes in the client selection function. Random selection, ratio select and round robin with ratio are implemented (2 of them need to be commented out at a given time.
