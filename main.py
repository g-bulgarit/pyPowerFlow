import numpy as np
from matplotlib import pyplot as plt

from powerflow import PowerFlowNetwork
bus_params = np.asarray(
# Bus type: 1 Slack, 0: just load, 2: generator and load
#        Bus Bus    Voltage Angle     ---Load----     -------Generator------    Static Mvar
#        No  type   Mag.    Degree    MW     Mvar     MW     Mvar  Qmin  Qmax    +Qc/-Ql
        [1,   1,    1.05,    0.0,    0,  0,    0.0,   0.0,  0,    0,        0,
         2,   0,    1,    0.0,    400,  250,    0,  0.0,  0, 0,    0,
         3,   2,    1.04,    0.0,    0,  0,    200.0,  0.0, 0, 0,    0]
).reshape((3, 11))

line_params = np.asarray(
#      Bus   bus     R       X      1/2 B
#      left  right  p.u.    p.u.    p.u.
       [1,   2,    0.02, 0.04,  0.00,
        1,   3,   0.01, 0.03,  0.00,
        2,   3,    0.0125, 0.025,  0.00,
]
).reshape((3, 5))


if __name__ == '__main__':
    base_power_reference = 100  # MVA
    accuracy = 1e-8  # delta
    max_iterations = 200

    # Get solved PF network:
    pf = PowerFlowNetwork(bus_params, line_params, base_power_reference, accuracy, max_iterations, mode="gauss")
    pf.export_bus_data(printout=True)

    # Plot stuff
    pf.plot_convergence_graph()
    pf.plot_voltages(pu=True, minimum_voltage=0.99)
    pf.plot_network_graph(minimum_voltage_pu=0.99, label_edges=True)
    pf.print_line_currents()
    plt.show()
