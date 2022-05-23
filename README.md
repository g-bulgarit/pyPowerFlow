# **pyPowerFlow**
Python tool to solve power flow problems in power networks using numeric methods, given line and bus parameters.

## **Suppored Numeric Methods**
* Newton-Raphson (Port of Hadi Sa'adat's implementation)
* Gauss-Seidel

## **Supported Generator Policies**
* "Tap-Changer": Play with voltages of busses that have generators on them, such that the reactive power `Q_gen` stays within the realm of `Q_min` and `Q_max`.
* "Clamp": Keep the voltages of busses with generators on them **constant**, and clamp `Q_gen` to the closest of {`Q_min`, `Q_max`}.

## **Note**
Code adapted from **Hadi Sa'adat's** MATLAB toolbox, published along with his book **Power System Analysis, third edition, McGraw-Hill, 1999**.

This code was ported and edited to solve a graded assignment in the course **Techno-Economical Problems in Power Systems**, in Tel-Aviv University, 2022.
The changes from the original `MATLAB` implementation are detailed below.


## Changes (from the original implementation)
* **No** tap-changer support (apart from the policy described earlier), i.e can't input tap changer stops for each generator.
* Added all line currents and losses to the outputs
* Added a network plot, a voltage plot and convergence statistics for each mode.

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

# Example initialization
```
my_pf =     PowerFlowNetwork(bus_params,
                             line_params,
                             base_power_reference,
                             accuracy, 
                             max_iterations,
                             mode="newton",
                             policy="tap_changer",
                             bus_csv_name="newton_bus_data.csv",
                             line_csv_name="newton_line_data.csv")
```