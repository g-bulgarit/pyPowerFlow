import numpy as np
from matplotlib import pyplot as plt

from powerflow import PowerFlowNetwork

bus_params = np.asarray(
#        Convention for bus type:
#        0: Just load on the bus
#        1: Slack Bus
#        2: Generator and load on the same bus
#
#        Bus Bus    Voltage Angle     ---Load----     -------Generator------    Static Mvar
#        No  type   Mag.    Degree    MW     Mvar     MW     Mvar  Qmin  Qmax    +Qc/-Ql
        [1,   1,    1.025,    0.0,    51.0,  41.0,    0.0,   0.0,  0,    0,        4,
         2,   2,    1.020,    0.0,    22.0,  15.0,    79.0,  0.0,  40.0, 250.0,    0,
         3,   2,    1.025,    0.0,    64.0,  50.0,    20.0,  0.0,  40.0, 150.0,    0,
         4,   2,    1.050,    0.0,    25.0,  10.0,    100.0, 0.0,  40.0, 80.0,     2,
         5,   2,    1.045,    0.0,    50.0,  30.0,    300.0, 0.0,  40.0, 160.0,    5,
         6,   0,    1.0,      0.0,    76.0,  29.0,    0,     0,    0,    0,        2,
         7,   0,    1.0,      0.0,    0.0,   0.0,     0,     0,    0,    0,        0,
         8,   0,    1.0,      0.0,    0.0,   0.0,     0,     0,    0,    0,        0,
         9,   0,    1.0,      0.0,    89.0,  50.0,    0,     0,    0,    0,        0,
        10,   0,    1.0,      0.0,    0.0,   0.0,     0,     0,    0,    0,        0,
        11,   0,    1.0,      0.0,    25.0,  15.0,    0,     0,    0,    0,        1.5,
        12,   0,    1.0,      0.0,    89.0,  48.0,    0,     0,    0,    0,        2,
        13,   0,    1.0,      0.0,    31.0,  15.0,    0,     0,    0,    0,        0,
        14,   0,    1.0,      0.0,    24.0,  12.0,    0,     0,    0,    0,        0,
        15,   0,    1.0,      0.0,    70.0,  31.0,    0,     0,    0,    0,        0.5,
        16,   0,    1.0,      0.0,    55.0,  27.0,    0,     0,    0,    0,        0,
        17,   0,    1.0,      0.0,    78.0,  38.0,    0,     0,    0,    0,        0,
        18,   0,    1.0,      0.0,    153.0, 67.0,    0,     0,    0,    0,        0,
        19,   0,    1.0,      0.0,    75.0,  15.0,    0,     0,    0,    0,        5,
        20,   0,    1.0,      0.0,    48.0,  27.0,    0,     0,    0,    0,        0,
        21,   0,    1.0,      0.0,    46.0,  23.0,    0,     0,    0,    0,        0,
        22,   0,    1.0,      0.0,    45.0,  22.0,    0,     0,    0,    0,        0,
        23,   0,    1.0,      0.0,    25.0,  12.0,    0,     0,    0,    0,        0,
        24,   0,    1.0,      0.0,    54.0,  27.0,    0,     0,    0,    0,        0,
        25,   0,    1.0,      0.0,    28.0,  13.0,    0,     0,    0,    0,        0,
        26,   2,    1.015,    0.0,    40.0,  20.0,    60.0,  0.0,  15.0, 50.0,     0]
).reshape((26, 11))

line_params = np.asarray(
#      Bus   bus     R       X      1/2 B
#      left  right  p.u.    p.u.    p.u.
       [1,   2,    0.0005, 0.0048,  0.03,
        1,   18,   0.0013, 0.0110,  0.06,
        2,   3,    0.0014, 0.0513,  0.05,
        2,   7,    0.0103, 0.0586,  0.018,
        2,   8,    0.0074, 0.0321,  0.039,
        2,   13,   0.0035, 0.0967,  0.025,
        2,   26,   0.0323, 0.1967,  0.0,
        3,   13,   0.0007, 0.0054,  0.0005,
        4,   8,    0.0008, 0.0240,  0.0001,
        4,   12,   0.0016, 0.0207,  0.015,
        5,   6,    0.0069, 0.0300,  0.099,
        6,   7,    0.0053, 0.0306,  0.001,
        6,   11,   0.0097, 0.0570,  0.0001,
        6,   18,   0.0037, 0.0222,  0.0012,
        6,   19,   0.0035, 0.0660,  0.045,
        6,   21,   0.0050, 0.0900,  0.0226,
        7,   8,    0.0012, 0.0069,  0.0001,
        7,   9,    0.0009, 0.0429,  0.025,
        8,   12,   0.0020, 0.0180,  0.02,
        9,   10,   0.0010, 0.0493,  0.001,
        10,  12,   0.0024, 0.0132,  0.01,
        10,  19,   0.0547, 0.2360,  0.000,
        10,  20,   0.0066, 0.0160,  0.001,
        10,  22,   0.0069, 0.0298,  0.005,
        11,  25,   0.0960, 0.2700,  0.010,
        11,  26,   0.0165, 0.0970,  0.004,
        12,  14,   0.0327, 0.0802,  0.000,
        12,  15,   0.0180, 0.0598,  0.000,
        13,  14,   0.0046, 0.0271,  0.001,
        13,  15,   0.0116, 0.0610,  0.000,
        13,  16,   0.0179, 0.0888,  0.001,
        14,  15,   0.0069, 0.0382,  0.000,
        15,  16,   0.0209, 0.0512,  0.000,
        16,  17,   0.0990, 0.0600,  0.000,
        16,  20,   0.0239, 0.0585,  0.000,
        17,  18,   0.0032, 0.0600,  0.038,
        17,  21,   0.2290, 0.4450,  0.000,
        19,  23,   0.0300, 0.1310,  0.000,
        19,  24,   0.0300, 0.1250,  0.002,
        19,  25,   0.1190, 0.2249,  0.004,
        20,  21,   0.0657, 0.157,   0.000,
        20,  22,   0.0150, 0.0366,  0.000,
        21,  24,   0.0476, 0.1510,  0.000,
        22,  23,   0.0290, 0.0990,  0.000,
        22,  24,   0.0310, 0.0880,  0.000,
        23,  25,   0.0987, 0.1168,  0.000]
).reshape((46, 5))


if __name__ == '__main__':
    base_power_reference = 100  # MVA
    accuracy = 1e-4  # delta
    max_iterations = 400

    # Get solved PF network:
    pf = PowerFlowNetwork(bus_params, line_params, base_power_reference, accuracy, max_iterations,
                          mode="gauss", policy="tap_changer")
    pf.export_bus_data(printout=True)

    # Plot stuff
    pf.plot_convergence_graph()
    pf.plot_voltages(pu=True, minimum_voltage=0.99)
    pf.plot_voltage_angles()
    pf.plot_network_graph(minimum_voltage_pu=0.99, label_edges=False)
    pf.print_line_currents()
    plt.show()
