# pyPowerFlow
Python tool to solve power flow problems in power networks using numeric methods, given line and bus parameters.

## Suppored Numeric Methods
* Newton-Raphson (Port of Hadi Sa'adat's implementation)
* Gauss-Seidel

## Note
Code adapted from **Hadi Sa'adat's** MATLAB toolbox, published along with his book **Power System Analysis, third edition, McGraw-Hill, 1999**.

This code was ported and edited to solve a graded assignment in the course **Techno-Economical Problems in Power Systems**, in Tel-Aviv University, 2022.
The changes from the original `MATLAB` implementation are detailed below.


## Changes (from the original implementation)
* **No** tap-changer support
* Added all line currents and losses to the outputs
* Added a network plot, a voltage plot and convergence statistics.

_____________
# Installation

1. Create a virtual environment `python -m venv venv`
2. Activate it (On unix: `source venv/Scripts/activate`, On windows: `.\venv\Scripts\activate`)
3. Install the requirements: `pip install -r requirements.txt`
4. (Modify the line and bus parameters)
5. Run the code `python main.py`

# Usage (`main.py`)
* Input the bus and line data according to the example and headers. The format is consistent with Hadi Sa'adat's format.
* Set the accuracy constraint and the maximum number of iterations

* Set the mode:
`newton` for Newton-Raphson, `gauss` for Gauss-Seidel.