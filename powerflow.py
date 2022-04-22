import numpy as np


class PowerFlowNetwork:
    def __init__(self, bus_parameters, line_parameters, base_power, accuracy, max_iterations):
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
