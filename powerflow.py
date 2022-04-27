import numpy as np
from matplotlib import pyplot as plt
import csv
import networkx as nx


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
        self.convergenceDeltas = []
        self.V = np.zeros(int(self.nbus), dtype=np.complex128)
        self.P = np.zeros(int(self.nbus))
        self.Q = np.zeros(int(self.nbus))
        self.S = np.zeros(int(self.nbus), dtype=np.complex128)
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
        self.Pgg = np.zeros(int(self.nbus))
        self.Qgg = np.zeros(int(self.nbus))
        self.ngs = np.zeros(int(self.nbus))
        self.nss = np.zeros(int(self.nbus))
        self.ns = 0
        self.ng = 0
        self.delta_degrees = np.zeros(int(self.nbus))
        self.y_load = np.zeros(int(self.nbus), dtype=np.complex128)
        self.Pg_total = 0
        self.Qg_total = 0
        self.Pd_total = 0
        self.Qd_total = 0
        self.Q_shunt_total = 0

        # Solve:
        self.newton_raphson_solver()
        self.calculate_line_flow()

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
            ybus[int(self.nl[k]) - 1, int(self.nr[k]) - 1] = ybus[int(self.nl[k]) - 1, int(self.nr[k]) - 1] - \
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

            # Create initial guess for voltages - set all voltages to 1.0 p.u
            if self.Vm[n] <= 0:
                self.Vm[n] = 1
                self.V[n] = 1 + 0j

            else:
                self.delta[n] = (np.pi / 180) * self.delta[n]
                self.V[n] = self.Vm[n] * (np.cos(self.delta[n]) + 1j * np.sin(self.delta[n]))
                self.P[n] = (self.Pg[n] - self.Pd[n]) / self.basePower
                self.Q[n] = (self.Qg[n] - self.Qd[n] + self.Q_shunt[n]) / self.basePower
                self.S[n] = self.P[n] + 1j * self.Q[n]

        # Do something
        for k in range(0, int(self.nbus)):
            if self.kb[k] == 1:
                self.ns += 1
            if self.kb[k] == 2:
                self.ng += 1

            self.ngs[k] = self.ng
            self.nss[k] = self.ns

        # Convert to a phasor-vector, separated to two vectors, one for magnitude and one for phase
        Ym = np.abs(self.ybus)
        t = np.angle(self.ybus)
        m = int(2 * self.nbus - self.ng - (2 * self.ns))
        max_error = 1
        converge = 1
        iteration = 0

        # Start iterating over solution
        while max_error >= self.accuracy and iteration <= self.maximumIterations:
            jacobian_matrix = np.zeros((m, m))
            dc_vec = np.zeros(m)

            iteration += 1  # increment iteration

            # Start calculation
            for n in range(0, int(self.nbus)):
                nn = int(n - self.nss[n])
                lm = int(self.nbus + n - self.ngs[n] - self.nss[n] - self.ns)

                # Jacobian elements:
                j11 = 0
                j22 = 0
                j33 = 0
                j44 = 0

                for i in range(0, int(self.nbr)):  # check -1 here
                    if (self.nl[i] - 1) == n or (self.nr[i] - 1) == n:
                        if (self.nl[i] - 1) == n:
                            l = int(self.nr[i]) - 1
                        if (self.nr[i] - 1) == n:
                            l = int(self.nl[i]) - 1
                        j11 += self.Vm[n] * self.Vm[l] * Ym[n, l] * np.sin(t[n, l] - self.delta[n] + self.delta[l])
                        j33 += self.Vm[n] * self.Vm[l] * Ym[n, l] * np.cos(t[n, l] - self.delta[n] + self.delta[l])

                        if self.kb[n] != 1:
                            j22 += self.Vm[l] * Ym[n, l] * np.cos(t[n, l] - self.delta[n] + self.delta[l])
                            j44 += self.Vm[l] * Ym[n, l] * np.sin(t[n, l] - self.delta[n] + self.delta[l])

                        if self.kb[n] != 1 and self.kb[l] != 1:
                            lk = int(self.nbus + l - self.ngs[l] - self.nss[l] - self.ns)
                            ll = int(l - self.nss[l])
                            # Calculate the elements off of the diagonal:
                            jacobian_matrix[nn, ll] = -1 * self.Vm[n] * self.Vm[l] * Ym[n, l] * np.sin(t[n, l] - self.delta[n] + self.delta[l])
                            if self.kb[l] == 0:
                                jacobian_matrix[nn, lk] = self.Vm[n] * Ym[n, l] * np.cos(t[n, l] - self.delta[n] + self.delta[l])
                            if self.kb[n] == 0:
                                jacobian_matrix[lm, ll] = -1 * self.Vm[n] * self.Vm[l] * Ym[n, l] * np.cos(t[n, l] - self.delta[n] + self.delta[l])
                            if self.kb[n] == 0 and self.kb[l] == 0:
                                jacobian_matrix[lm, lk] = -1 * self.Vm[n] * Ym[n, l] * np.sin(t[n, l] - self.delta[n] + self.delta[l])

                Pk = (self.Vm[n] ** 2) * Ym[n, n] * np.cos(t[n, n]) + j33
                Qk = -1 * (self.Vm[n] ** 2) * Ym[n, n] * np.sin(t[n, n]) - j11

                # Handle the swing bus
                if self.kb[n] == 1:
                    self.P[n] = Pk
                    self.Q[n] = Qk

                if self.kb[n] == 2:
                    self.Q[n] = Qk
                    if self.Q_max[n] != 0:
                        Qgc = (self.Q[n] * self.basePower) + self.Qd[n] - self.Q_shunt[n]
                        if iteration <= 7:
                            if iteration > 2:
                                if Qgc < self.Q_min[n]:
                                    self.Vm[n] += 0.01
                                elif Qgc > self.Q_max[n]:
                                    self.Vm[n] -= 0.01

                if int(self.kb[n]) != 1:
                    jacobian_matrix[nn, nn] = j11
                    dc_vec[nn] = self.P[n] - Pk

                if int(self.kb[n]) == 0:
                    jacobian_matrix[nn, lm] = (2 * self.Vm[n] * Ym[n, n] * np.cos(t[n, n])) + j22
                    jacobian_matrix[lm, nn] = j33
                    jacobian_matrix[lm, lm] = (-2 * self.Vm[n] * Ym[n, n] * np.sin(t[n, n])) - j44
                    dc_vec[lm] = self.Q[n] - Qk

            # Solve with least-squares
            dx_vec = np.linalg.lstsq(jacobian_matrix, dc_vec.T, rcond=None)[0]

            for n in range(0, int(self.nbus)):
                nn = int(n - self.nss[n])
                lm = int(self.nbus + n - self.ngs[n] - self.nss[n] - self.ns)
                if self.kb[n] != 1:
                    self.delta[n] += dx_vec[nn]
                if self.kb[n] == 0:
                    self.Vm[n] += dx_vec[lm]

            max_error = np.max(np.abs(dc_vec))
            self.convergenceDeltas.append(max_error)

            # Check if we are diverging beyond the allowed limit:
            if iteration == self.maximumIterations and max_error > self.accuracy:
                print(f"Solution did not converge after {iteration} iterations...")
                converge = 0

        if converge == 1:
            print(f"Solution converged after {iteration} iterations!")
            self.V = self.Vm * np.cos(self.delta) + 1j * self.Vm * np.sin(self.delta)
            self.delta_degrees = (180 / np.pi) * self.delta

            k = 0
            for n in range(0, int(self.nbus)):
                if self.kb[n] == 1:
                    k += 1
                    self.S[n] = self.P[n] + 1j * self.Q[n]
                    self.Pg[n] = self.P[n] * self.basePower + self.Pd[n]
                    self.Qg[n] = self.Q[n] * self.basePower + self.Qd[n] - self.Q_shunt[n]
                    self.Pgg[k] = self.Pg[n]
                    self.Qgg[k] = self.Qg[n]
                self.y_load[n] = (self.Pd[n] - (1j * self.Qd[n]) + (1j * self.Q_shunt[n])) / (self.basePower * (self.Vm[n] ** 2))
            self.bus_data[:, 2] = self.Vm.T
            self.bus_data[:, 3] = self.delta_degrees.T
            self.Pg_total = np.sum(self.Pg)
            self.Qg_total = np.sum(self.Qg)
            self.Pd_total = np.sum(self.Pd)
            self.Qd_total = np.sum(self.Qd)
            self.Q_shunt_total = np.sum(self.Q_shunt)

    def calculate_line_flow(self):
        slt = 0
        outlines = ["From, To, MW, MVAR, MVA, MW Loss, MVAR Loss, Tap\n"]

        for n in range(0, int(self.nbus)):
            bus_body = 0
            for l in range(0, int(self.nbr)):
                k = 0
                if not bus_body:
                    # Print header:
                    outlines.append(f"{n + 1}, , {self.P[n] * self.basePower:.3f}, {self.Q[n] * self.basePower:.3f},"
                                    f" {np.abs(self.S[n] * self.basePower):.3f}\n")
                    bus_body = 1

                if self.nl[l] - 1 == n:
                    k = int(self.nr[l]) - 1
                    i_n = (self.V[n] - self.a[l] * self.V[k]) * self.y[l] / (self.a[l] ** 2) + \
                        self.B_c[l] / (self.a[l] ** 2) * self.V[n]
                    i_k = (self.V[k] - self.V[n] / self.a[l]) * self.y[l] + self.B_c[l] * self.V[k]
                    s_nk = self.V[n] * np.conj(i_n) * self.basePower
                    s_kn = self.V[k] * np.conj(i_k) * self.basePower
                    sl = s_nk + s_kn
                    slt += s_nk + s_kn

                elif self.nr[l] - 1 == n:
                    k = int(self.nl[l]) - 1
                    i_n = (self.V[n] - self.V[k] / self.a[l] ) * self.y[l] + \
                          self.B_c[l] * self.V[n]
                    i_k = (self.V[k] - self.a[l] * self.V[n]) * self.y[l] / (self.a[l] ** 2) + \
                          self.B_c[l] / (self.a[l] ** 2) * self.V[k]
                    s_nk = self.V[n] * np.conj(i_n) * self.basePower
                    s_kn = self.V[k] * np.conj(i_k) * self.basePower
                    sl = s_nk + s_kn
                    slt += s_nk + s_kn

                if self.nl[l] - 1 == n or self.nr[l] - 1 == n:
                    out_str = f" , {k + 1}, {np.real(s_nk):.3f}, {np.imag(s_nk):.3f}," \
                              f" {np.abs(s_nk):.3f}, {np.real(sl):.3f}, "
                    if self.nl[l] - 1 == n and self.a[l] != 1:
                        out_str += f"{np.imag(sl):.3f}, {self.a[l]:.3f}\n"
                    else:
                        out_str += f"{np.imag(sl):.3f}, ,\n"
                    outlines.append(out_str)

                with open("line_outputs.csv", "w+") as csv_out:
                    csv_out.writelines(outlines)

    def plot_convergence_graph(self):
        plt.figure()
        plt.plot(self.convergenceDeltas, label="Delta")
        plt.title("Delta between iterations as a function of iterations")
        plt.xlabel("Iteration Number [#]")
        plt.ylabel("Absolute Delta")
        plt.legend()

    def plot_voltages(self, pu=True, minimum_voltage=0.97):
        plt.figure()
        voltages_pu = np.abs(self.V)
        plt.ylabel("Voltage [pu]")

        if not pu:
            voltages_pu = np.real(self.V) * self.basePower
            plt.ylabel("Voltage [V]")

        x_axis = list(range(1, voltages_pu.size + 1))
        plt.title("Network Voltage Distribution")
        for idx, _ in enumerate(voltages_pu):
            if voltages_pu[idx] < minimum_voltage:
                color = "r"
            else:
                color = "g"
            plt.scatter(x_axis[idx], voltages_pu[idx], color=color)
            plt.annotate(idx+1, (x_axis[idx], voltages_pu[idx]))
        plt.grid(visible=True, which="both", axis="y")
        plt.minorticks_on()
        plt.xlabel("Bus Number [#]")
        plt.axhline(y=minimum_voltage, color='r', linestyle='--')

    def plot_network_graph(self, minimum_voltage_pu=0.97):
        plt.figure()
        graph = nx.Graph()
        voltages = np.abs(self.V)
        colors = []
        labels = dict()
        # Add nodes
        for i in range(int(self.nbus)):
            # Create this bus as a node on the graph
            graph.add_node(i)
            labels[i] = i+1
            if i == 0:
                colors.append("blue")
            elif voltages[i] > minimum_voltage_pu:
                colors.append('green')
            else:
                colors.append('red')

        # Add edges
        for row in self.line_data:
            graph.add_edge(int(row[0]) - 1, int(row[1]) - 1)

        layout_pos = nx.planar_layout(graph)
        plt.title("Network Graph: Busses and Lines")
        nx.draw(graph, pos=layout_pos, labels=labels, with_labels=True, node_color=colors)

    def export_bus_data(self, printout=True):
        outlines = ["Bus #, Voltage, Angle, Load MW, Load MVAR, Generator MW, Generator MVAR, Injected MVAR\n"]
        for i in range(int(self.nbus)):
            outlines.append(f"{i}, {self.Vm[i]:.3f}, {self.delta_degrees[i]:.3f}, "
                            f"{self.Pd[i]:.3f}, {self.Qd[i]:.3f}, {self.Pg[i]:.3f},"
                            f"{self.Qg[i]:.3f}, {self.Q_shunt[i]:.3f}\n")

        with open("bus_outputs.csv", "w+") as csvfile:
            csvfile.writelines(outlines)

        if printout:
            [print(line) for line in outlines]
            print(f"Bus Totals:\n"
                  f"Total Dispersed Power: {self.Pd_total:.3f}[MW], {self.Qd_total:.3f}[MVAR]\n"
                  f"Total Generated Power: {self.Pg_total:.3f}[MW], {self.Qg_total:.3f}[MVAR],"
                  f"With Capacitors:{self.Q_shunt_total:.3f}[MVAR]")




