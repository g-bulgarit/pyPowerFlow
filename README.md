# pyPowerFlow
Python utility to solve power flow problems in power networks, given line and bus parameters.

Code adapted from **Hadi Sa'adat's** MATLAB toolbox, published along with his book **Power System Analysis, third edition, McGraw-Hill, 1999**.

This MATLAB port lets you solve using the Newton-Raphson method iterativly, with additional plotting capabilities not present in Sa'adat's original publication.

## **Note:**
This code was ported and edited to solve a graded assignment in the course **Techno-Economical Problems in Power Systems**, in Tel-Aviv University, 2022.

_____________
# Installation

1. Create a virtual environment `python -m venv venv`
2. Activate it (On unix: `source venv/Scripts/activate`, On windows: `.\venv\Scripts\activate`)
3. Install the requirements: `pip install -r requirements.txt`
4. (Modify the line and bus parameters)
5. Run the code `python main.py`