import numpy as np


class PowerFlowNetwork:
    def __init__(self, bus_parameters, line_parameters, base_power, accuracy, max_iterations):
        # Keep line and bus parameters in the object
        self.bus_data = bus_parameters
        self.line_data = line_parameters

        # Parse Newton-Raphson parameters
        self.accuracy = accuracy
        self.maximumIterations = max_iterations

        # Parse general parameters
        self.basePower = base_power

        # Parse line parameters into arrays
        self.nl = line_parameters[:, 0]
        self.nr = line_parameters[:, 1]
        self.R = line_parameters[:, 2]
        self.X = line_parameters[:, 3]
        self.B_c = 1j * line_parameters[:, 4]
        self.a = line_parameters[:, 5]
        self.nbr = len(line_parameters[:, 1])

        # Parse bus parameters
        self.nbus = np.max([np.max(self.nl), np.max(self.nr)])

        # Calculate admittances and impedances
        self.Z = self.R + 1j * self.X
        self.y = np.ones(self.nbr).T / self.Z

        # Find admittance matrix
        self.ybus = self.calculate_admittance_matrix()

        # Initialize output parameters
        self.V = np.zeros(int(self.nbus))
        self.P = np.zeros(int(self.nbus))
        self.Q = np.zeros(int(self.nbus))
        self.S = np.zeros(int(self.nbus))
        self.kb = np.zeros(int(self.nbus))
        self.Vm = np.zeros(int(self.nbus))
        self.delta = np.zeros(int(self.nbus))
        self.Pd = np.zeros(int(self.nbus))
        self.Qd = np.zeros(int(self.nbus))
        self.Pg = np.zeros(int(self.nbus))
        self.Qg = np.zeros(int(self.nbus))
        self.Q_min = np.zeros(int(self.nbus))
        self.Q_max = np.zeros(int(self.nbus))
        self.Q_shunt = np.zeros(int(self.nbus))

        # Solve:
        self.newton_raphson_solver()

    def calculate_admittance_matrix(self) -> np.ndarray:
        """
        Calculate admittance matrix from given line and bus data
        :return: nBus x nBus matrix as numpy complex128 array.
        """

        # Prepare admittance matrix placeholder:
        ybus = np.zeros((int(self.nbus), int(self.nbus)), dtype=np.complex128)

        # Fill gaps:
        for n in range(0, int(self.nbr)):
            if self.a[n] <= 0:
                self.a[n] = 1

        # Assign values in the admittance matrix that are not on the diagonal:
        for k in range(0, int(self.nbr)):
            ybus[int(self.nl[k]) - 1, int(self.nr[k]) - 1] = ybus[int(self.nl[k]) - 1, int(self.nr[k]) - 1] -\
                                                             self.y[k]/self.a[k]
            ybus[int(self.nr[k]) - 1, int(self.nl[k]) - 1] = ybus[int(self.nl[k]) - 1, int(self.nr[k]) - 1]

        # Assign values in the admittance matrix that are on the diagonal
        for n in range(0, int(self.nbus)):
            for k in range(0, int(self.nbr)):
                if (self.nl[k] - 1) == n:
                    ybus[n, n] = ybus[n, n] + self.B_c[k] + self.y[k]/((self.a[k]) ** 2)
                elif (self.nr[k] - 1) == n:
                    ybus[n, n] = ybus[n, n] + self.y[k] + self.B_c[k]
                else:
                    continue
        return ybus

    def newton_raphson_solver(self):
        # Find voltage, real and imaginary power, and total power in complex units
        for k in range(0, int(self.nbus)):
            n = int(self.bus_data[k, 0] - 1)
            # Parse:
            self.kb[n] = self.bus_data[k, 1]
            self.Vm[n] = self.bus_data[k, 2]
            self.delta[n] = self.bus_data[k, 3]
            self.Pd[n] = self.bus_data[k, 4]
            self.Qd[n] = self.bus_data[k, 5]
            self.Pg[n] = self.bus_data[k, 6]
            self.Qg[n] = self.bus_data[k, 7]
            self.Q_min[n] = self.bus_data[k, 8]
            self.Q_max[n] = self.bus_data[k, 9]
            self.Q_shunt[n] = self.bus_data[k, 10]

            if self.Vm[n] <= 0:
                self.Vm[n] = 1
                self.V[n] = 1 + 0j

            else:
                self.delta[n] = (np.pi / 180) * self.delta[n];
                self.V[n] = self.Vm[n] * (np.cos(self.delta[n]) + 1j * np.sin(self.delta[n]))
                self.P[n] = (self.Pg[n] - self.Pd[n]) / self.basePower
                self.P[n] = (self.Qg[n] - self.Qd[n] + self.Q_shunt[n]) / self.basePower
                self.S[n] = self.P[n] + 1j * self.Q[n]

        pass
