import numpy as np
from matplotlib import pyplot as plt
import networkx as nx


class PowerFlowNetwork:
    def __init__(self, bus_parameters, line_parameters, base_power, accuracy, max_iterations, mode="newton"):
        # Keep line and bus parameters in the object
        self.bus_data = bus_parameters
        self.line_data = line_parameters

        # Parse Newton-Raphson parameters
        self.accuracy = accuracy
        self.maximumIterations = max_iterations
        self.converge = 1

        # Parse general parameters
        self.basePower = base_power
        self.current = dict()

        # Parse line parameters into arrays
        self.line_left = line_parameters[:, 0]
        self.line_right = line_parameters[:, 1]
        self.R = line_parameters[:, 2]
        self.X = line_parameters[:, 3]
        self.B_c = 1j * line_parameters[:, 4]
        self.nlines = len(line_parameters[:, 1])

        # Parse bus parameters
        self.nbus = np.max(bus_parameters[:, 0])

        # Calculate admittances and impedances
        self.Z = self.R + 1j * self.X
        self.y = np.ones(self.nlines).T / self.Z

        # Find admittance matrix
        self.ybus = self.calculate_admittance_matrix()
        self.y_to_gnd = self.calculate_admittance_to_gnd()

        # Initialize output parameters
        self.convergenceDeltas = []
        self.V = np.zeros(int(self.nbus), dtype=np.complex128)
        self.P = np.zeros(int(self.nbus))
        self.Q = np.zeros(int(self.nbus))
        self.S = np.zeros(int(self.nbus), dtype=np.complex128)
        self.bus_type = np.zeros(int(self.nbus))
        self.V_mag = np.zeros(int(self.nbus))
        self.delta = np.zeros(int(self.nbus))
        self.P_load = np.zeros(int(self.nbus))
        self.Q_load = np.zeros(int(self.nbus))
        self.P_gen = np.zeros(int(self.nbus))
        self.Q_gen = np.zeros(int(self.nbus))
        self.Q_min = np.zeros(int(self.nbus))
        self.Q_max = np.zeros(int(self.nbus))
        self.Q_shunt = np.zeros(int(self.nbus))
        self.num_gen_seen = np.zeros(int(self.nbus))
        self.num_slack_seen = np.zeros(int(self.nbus))
        self.num_slack = 0
        self.num_gen = 0
        self.nload = 0
        self.delta_degrees = np.zeros(int(self.nbus))
        self.P_gen_total = 0
        self.Q_gen_total = 0
        self.P_load_total = 0
        self.Q_load_total = 0
        self.Q_shunt_total = 0
        self.linepq = dict()

        # Parse bus parameters
        self.init_bus_data()

        # Todo: init line data

        # Solve:
        if mode == "newton":
            self.newton_raphson_solver()
        elif mode == "gauss":
            self.gauss_solver()
        self.calculate_line_flow()

    def calculate_admittance_matrix(self) -> np.ndarray:
        """
        Calculate admittance matrix from given line and bus data
        :return: nBus x nBus matrix as numpy complex128 array.
        """

        # Prepare admittance matrix placeholder:
        ybus = np.zeros((int(self.nbus), int(self.nbus)), dtype=np.complex128)

        # Assign values in the admittance matrix that are not on the diagonal, by inputting the line data:
        for k in range(0, int(self.nlines)):
            ybus[int(self.line_left[k]) - 1, int(self.line_right[k]) - 1] = -1 * self.y[k]
            ybus[int(self.line_right[k]) - 1, int(self.line_left[k]) - 1] = ybus[
                int(self.line_left[k]) - 1, int(self.line_right[k]) - 1]

        # Assign values in the admittance matrix that are on the diagonal
        for bus_idx in range(0, int(self.nbus)):
            for line_idx in range(0, int(self.nlines)):
                if (self.line_left[line_idx] - 1) == bus_idx:
                    ybus[bus_idx, bus_idx] += self.y[line_idx] + self.B_c[line_idx]

                elif (self.line_right[line_idx] - 1) == bus_idx:
                    ybus[bus_idx, bus_idx] += self.y[line_idx] + self.B_c[line_idx]
                else:
                    continue
        return ybus

    def calculate_admittance_to_gnd(self) -> np.ndarray:
        # Return a vector of total capacitance (in admittance) in every bus

        # Prepare admittance matrix placeholder:
        y_to_gnd = np.zeros(int(self.nbus), dtype=np.complex128)

        for n in range(0, int(self.nbus)):
            y_to_gnd[n] = np.sum(self.ybus[n, :])
        return y_to_gnd

    def init_bus_data(self):
        # Prepare for numeric calculation by reading all inputs in the correct format.

        # Find voltage, real and imaginary power, and total power in complex units
        for k in range(0, int(self.nbus)):
            n = int(self.bus_data[k, 0] - 1)
            # Parse:
            self.bus_type[n] = self.bus_data[k, 1]
            self.V_mag[n] = self.bus_data[k, 2]
            self.delta_degrees[n] = self.bus_data[k, 3]            # Phasor angle (in degrees)
            self.delta[n] = (np.pi / 180) * self.delta_degrees[n]  # Convert to radians

            # All power units in p.u
            self.P_load[n] = self.bus_data[k, 4] / self.basePower
            self.Q_load[n] = self.bus_data[k, 5] / self.basePower
            self.P_gen[n] = self.bus_data[k, 6] / self.basePower
            self.Q_gen[n] = self.bus_data[k, 7] / self.basePower
            self.Q_min[n] = self.bus_data[k, 8] / self.basePower
            self.Q_max[n] = self.bus_data[k, 9] / self.basePower
            self.Q_shunt[n] = self.bus_data[k, 10] / self.basePower

            # Override(!) bus parameters based on type (load, slack, gen)
            if self.bus_type[n] == 0:  # Load
                self.V_mag[n] = 1
                self.delta[n] = 0
            elif self.bus_type[n] == 1:  # Slack
                self.P_gen[n] = 0
                self.Q_gen[n] = 0
            elif self.bus_type[n] == 2:  # Generator and Load
                self.delta[n] = 0
                self.Q_gen[n] = 0
            else:
                exit("Wrong bus type in input")

            self.V[n] = self.V_mag[n] * (np.cos(self.delta[n]) + 1j * np.sin(self.delta[n]))
            self.P[n] = (self.P_gen[n] - self.P_load[n])
            self.Q[n] = (self.Q_gen[n] - (self.Q_load[n] - self.Q_shunt[n]))
            self.S[n] = self.P[n] + 1j * self.Q[n]

    def newton_raphson_solver(self):
        for k in range(0, int(self.nbus)):
            if self.bus_type[k] == 1:
                self.num_slack += 1
            elif self.bus_type[k] == 2:
                self.num_gen += 1
            elif self.bus_type[k] == 0:
                self.nload += 1

            self.num_gen_seen[k] = self.num_gen
            self.num_slack_seen[k] = self.num_slack

        # Convert to a phasor-vector, separated to two vectors, one for magnitude and one for phase
        y_mag = np.abs(self.ybus)
        y_angle = np.angle(self.ybus)

        # Calculate jacobian matrix size based on how many variables we need to find
        num_unknowns = int(2 * self.nload + self.num_gen)

        max_error = 1
        converge = 1
        iteration = 0

        # Start iterating over solution
        while max_error >= self.accuracy and iteration <= self.maximumIterations:
            jacobian_matrix = np.zeros((num_unknowns, num_unknowns))
            dc_vec = np.zeros(num_unknowns)

            iteration += 1  # increment iteration

            # Start calculation
            for n in range(0, int(self.nbus)):
                nn = int(n - self.num_slack_seen[n])
                lm = int(self.nbus + n - self.num_gen_seen[n] - self.num_slack_seen[n] - self.num_slack)

                # Jacobian elements:
                j11 = 0
                j22 = 0
                j33 = 0
                j44 = 0

                for i in range(0, int(self.nlines)):
                    if (self.line_left[i] - 1) == n or (self.line_right[i] - 1) == n:
                        if (self.line_left[i] - 1) == n:
                            l = int(self.line_right[i]) - 1
                        if (self.line_right[i] - 1) == n:
                            l = int(self.line_left[i]) - 1
                        j11 += self.V_mag[n] * self.V_mag[l] * y_mag[n, l] * np.sin(
                            y_angle[n, l] - self.delta[n] + self.delta[l])
                        j33 += self.V_mag[n] * self.V_mag[l] * y_mag[n, l] * np.cos(
                            y_angle[n, l] - self.delta[n] + self.delta[l])

                        if self.bus_type[n] != 1:
                            j22 += self.V_mag[l] * y_mag[n, l] * np.cos(y_angle[n, l] - self.delta[n] + self.delta[l])
                            j44 += self.V_mag[l] * y_mag[n, l] * np.sin(y_angle[n, l] - self.delta[n] + self.delta[l])

                        if self.bus_type[n] != 1 and self.bus_type[l] != 1:
                            lk = int(self.nbus + l - self.num_gen_seen[l] - self.num_slack_seen[l] - self.num_slack)
                            ll = int(l - self.num_slack_seen[l])
                            # Calculate the elements off of the diagonal:
                            jacobian_matrix[nn, ll] = -1 * self.V_mag[n] * self.V_mag[l] * y_mag[n, l] * np.sin(
                                y_angle[n, l] - self.delta[n] + self.delta[l])
                            if self.bus_type[l] == 0:
                                jacobian_matrix[nn, lk] = self.V_mag[n] * y_mag[n, l] * np.cos(
                                    y_angle[n, l] - self.delta[n] + self.delta[l])
                            if self.bus_type[n] == 0:
                                jacobian_matrix[lm, ll] = -1 * self.V_mag[n] * self.V_mag[l] * y_mag[n, l] * np.cos(
                                    y_angle[n, l] - self.delta[n] + self.delta[l])
                            if self.bus_type[n] == 0 and self.bus_type[l] == 0:
                                jacobian_matrix[lm, lk] = -1 * self.V_mag[n] * y_mag[n, l] * np.sin(
                                    y_angle[n, l] - self.delta[n] + self.delta[l])

                Pk = (self.V_mag[n] ** 2) * y_mag[n, n] * np.cos(y_angle[n, n]) + j33
                Qk = -1 * (self.V_mag[n] ** 2) * y_mag[n, n] * np.sin(y_angle[n, n]) - j11

                # Handle the slack bus
                if self.bus_type[n] == 1:
                    self.P[n] = Pk
                    self.Q[n] = Qk

                if self.bus_type[n] == 2:
                    self.Q[n] = Qk
                    if self.Q_max[n] != 0:
                        Q_genc = (self.Q[n] * self.basePower) + self.Q_load[n] - self.Q_shunt[n]
                        if iteration <= 7:
                            if iteration > 2:
                                if Q_genc < self.Q_min[n]:
                                    self.V_mag[n] += 0.01
                                elif Q_genc > self.Q_max[n]:
                                    self.V_mag[n] -= 0.01

                if int(self.bus_type[n]) != 1:
                    jacobian_matrix[nn, nn] = j11
                    dc_vec[nn] = self.P[n] - Pk

                if int(self.bus_type[n]) == 0:
                    jacobian_matrix[nn, lm] = (2 * self.V_mag[n] * y_mag[n, n] * np.cos(y_angle[n, n])) + j22
                    jacobian_matrix[lm, nn] = j33
                    jacobian_matrix[lm, lm] = (-2 * self.V_mag[n] * y_mag[n, n] * np.sin(y_angle[n, n])) - j44
                    dc_vec[lm] = self.Q[n] - Qk

            # Solve with least-squares
            dx_vec = np.linalg.lstsq(jacobian_matrix, dc_vec.T, rcond=None)[0]

            for n in range(0, int(self.nbus)):
                nn = int(n - self.num_slack_seen[n])
                lm = int(self.nbus + n - self.num_gen_seen[n] - self.num_slack_seen[n] - self.num_slack)
                if self.bus_type[n] != 1:
                    self.delta[n] += dx_vec[nn]
                if self.bus_type[n] == 0:
                    self.V_mag[n] += dx_vec[lm]

            max_error = np.max(np.abs(dc_vec))
            self.convergenceDeltas.append(max_error)

            # Check if we are diverging beyond the allowed limit:
            if iteration == self.maximumIterations and max_error > self.accuracy:
                print(f"Solution did not converge after {iteration} iterations...")
                self.converge = 0
                converge = 0

        if converge == 1:
            print(f"Solution converged after {iteration} iterations!")
            self.V = self.V_mag * (np.cos(self.delta) + 1j *  np.sin(self.delta))
            self.delta_degrees = (180 / np.pi) * self.delta

            k = 0
            for n in range(0, int(self.nbus)):
                if self.bus_type[n] == 1:
                    k += 1
                    self.S[n] = self.P[n] + 1j * self.Q[n]
                    self.P_gen[n] = self.P[n] * self.basePower + self.P_load[n]
                    self.Q_gen[n] = self.Q[n] * self.basePower + self.Q_load[n] - self.Q_shunt[n]
            self.bus_data[:, 2] = self.V_mag.T
            self.bus_data[:, 3] = self.delta_degrees.T
            self.P_gen_total = np.sum(self.P_gen)
            self.Q_gen_total = np.sum(self.Q_gen)
            self.P_load_total = np.sum(self.P_load)
            self.Q_load_total = np.sum(self.Q_load)
            self.Q_shunt_total = np.sum(self.Q_shunt)

    def gauss_solver(self):
        max_error = 1
        converge = 0  # Flag
        iteration = 0

        # Start iterating over solution
        while max_error >= self.accuracy and iteration <= self.maximumIterations:
            v_previous = np.copy(self.V)
            iteration += 1
            # Iteratively find the voltages in all busses
            for n in range(0, int(self.nbus)):
                if self.bus_type[n] == 1:  # Slack bus
                    continue
                elif self.bus_type[n] == 0:  # Only load
                    self.V[n] = self.calc_v(n)
                elif self.bus_type[n] == 2:  # Generator and Load (known voltage)
                    self.Q[n] = -1 * np.imag(np.conj(self.V[n]) * self.calc_current(n))
                    self.S[n] = self.P[n] + 1j * self.Q[n]
                    V_imag = np.imag(self.calc_v(n))
                    V_real = np.sqrt(np.square(self.V_mag[n]) - np.square(V_imag))
                    self.V[n] = V_real + 1j * V_imag

            max_error = np.max(np.abs(self.V - v_previous))
            self.convergenceDeltas.append(max_error)

        if max_error < self.accuracy:
            self.converge = 1
        # Find other parameters from voltage and other known param
        for bus_idx in range(0, int(self.nbus)):
            self.V_mag[bus_idx] = np.abs(self.V[bus_idx])
            self.delta[bus_idx] = np.angle(self.V[bus_idx])

            if self.bus_type[bus_idx] == 0:  # Only load
                continue

            if self.bus_type[bus_idx] == 1:  # Slack bus calculations
                self.P[bus_idx] = np.real(np.conj(self.V[bus_idx]) * self.calc_current(bus_idx))
                self.Q[bus_idx] = -1 * np.imag(np.conj(self.V[bus_idx]) * self.calc_current(bus_idx))
                self.S[bus_idx] = self.P[bus_idx] + 1j * self.Q[bus_idx]
                self.P_gen[bus_idx] = self.P[bus_idx] + self.P_load[bus_idx]
                self.Q_gen[bus_idx] = self.Q[bus_idx] + self.Q_load[bus_idx] - self.Q_shunt[bus_idx]

            elif self.bus_type[bus_idx] == 2:  # Load and generator (P is given)
                self.Q_gen[bus_idx] = self.Q[bus_idx] + self.Q_load[bus_idx] - self.Q_shunt[bus_idx]

        self.P_gen_total = np.sum(self.P_gen)
        self.Q_gen_total = np.sum(self.Q_gen)
        self.P_load_total = np.sum(self.P_load)
        self.Q_load_total = np.sum(self.Q_load)
        self.Q_shunt_total = np.sum(self.Q_shunt)

    def calc_v(self, bus_idx: int):
        denominator = self.ybus[bus_idx, bus_idx]
        numerator = np.conj(self.S[bus_idx] / self.V[bus_idx])
        temp = 0
        for k in range(0, int(self.nbus)):
            if k == bus_idx:
                continue
            temp += self.V[k] * self.ybus[bus_idx, k]
        numerator -= temp
        return numerator / denominator

    def calc_current(self, bus_idx: int):
        current = self.V[bus_idx] * self.ybus[bus_idx, bus_idx]
        temp = 0
        for k in range(0, int(self.nbus)):
            if k == bus_idx:
                continue
            temp += self.V[k] * self.ybus[bus_idx, k]  # ybus is negative
        current += temp
        return current

    def calculate_line_flow(self):
        if not self.converge:
            return
        slt = 0
        outlines = ["From, To, MW, MVAR, MVA, MW Loss, MVAR Loss, Tap\n"]

        for n in range(0, int(self.nbus)):
            bus_body = 0
            for l in range(0, int(self.nlines)):
                k = 0
                if not bus_body:
                    # Print header:
                    outlines.append(f"{n + 1}, , {self.P[n] * self.basePower:.3f}, {self.Q[n] * self.basePower:.3f},"
                                    f" {np.abs(self.S[n] * self.basePower):.3f}\n")
                    bus_body = 1

                if self.line_left[l] - 1 == n:
                    # Do calculation
                    k = int(self.line_right[l]) - 1
                    i_n = (self.V[n] - self.V[k]) * self.y[l] + self.B_c[l] * self.V[n]
                    i_k = (self.V[k] - self.V[n]) * self.y[l] + self.B_c[l] * self.V[k]
                    s_nk = self.V[n] * np.conj(i_n) * self.basePower
                    s_kn = self.V[k] * np.conj(i_k) * self.basePower
                    sl = s_nk + s_kn
                    slt += s_nk + s_kn
                    self.current[(n, k)] = np.conj(sl / self.basePower) / np.conj(self.V[n] - self.V[k])  # TODO

                elif self.line_right[l] - 1 == n:
                    # Do calculation
                    k = int(self.line_left[l]) - 1

                    # Calc current:
                    i_n = (self.V[n] - self.V[k]) * self.y[l] + self.B_c[l] * self.V[n]
                    i_k = (self.V[k] - self.V[n]) * self.y[l] + self.B_c[l] * self.V[k]
                    s_nk = self.V[n] * np.conj(i_n) * self.basePower
                    s_kn = self.V[k] * np.conj(i_k) * self.basePower
                    sl = s_nk + s_kn

                    # calculate current also
                    self.current[(n, k)] = np.conj(sl / self.basePower) / np.conj(self.V[n] - self.V[k])  # TODO
                    slt += s_nk + s_kn

                if self.line_left[l] - 1 == n or self.line_right[l] - 1 == n:
                    out_str = f" , {k + 1}, {np.real(s_nk):.3f}, {np.imag(s_nk):.3f}," \
                              f" {np.abs(s_nk):.3f}, {np.real(sl):.3f}, "

                    out_str += f"{np.imag(sl):.3f}, ,\n"
                    outlines.append(out_str)
                self.linepq[(n, l)] = [np.real(s_nk), np.imag(s_nk)]

        pairs = list(self.current.keys())

        for idx in range(int(self.nbus)):
            outbound_current = 0
            inbound_current = 0
            for pair in pairs:
                if pair[0] == idx:
                    # this is an outbound connection
                    outbound_current += self.current[pair]
                if pair[1] == idx:
                    # this is an inbound connection
                    inbound_current += self.current[pair]
            print(f"Node: {idx + 1}, In: {inbound_current}, Out: {outbound_current}, "
                  f"Delta: {outbound_current + inbound_current}")

        with open("line_outputs.csv", "w+") as csv_out:
            csv_out.writelines(outlines)

    def plot_convergence_graph(self):
        """
        in pu
        :return:
        """
        plt.figure()
        plt.plot(self.convergenceDeltas, label="Delta")
        plt.title("Convergence Graph: Newton-Raphson")
        plt.xlabel("Iteration Number [#]")
        plt.ylabel("Max difference of power between iterations")
        plt.legend()

    def plot_voltages(self, pu=True, minimum_voltage=0.97):
        if not self.converge:
            return
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
            plt.annotate(idx + 1, (x_axis[idx], voltages_pu[idx]))
        plt.grid(visible=True, which="both", axis="y")
        plt.minorticks_on()
        plt.xlabel("Bus Number [#]")
        plt.axhline(y=minimum_voltage, color='r', linestyle='--')

    def plot_network_graph(self, minimum_voltage_pu=0.97, label_edges=False):
        if not self.converge:
            return
        plt.figure()
        graph = nx.DiGraph()
        voltages = np.abs(self.V)
        colors = []
        labels = dict()
        edge_labels = dict()
        # Add nodes
        for i in range(int(self.nbus)):
            # Create this bus as a node on the graph
            graph.add_node(i)
            labels[i] = i + 1
            if i == 0:
                colors.append("blue")
            elif voltages[i] > minimum_voltage_pu:
                colors.append('green')
            else:
                colors.append('red')

        # Add edges
        for row in self.line_data:
            start_point = int(row[0]) - 1
            end_point = int(row[1]) - 1
            graph.add_edge(start_point, end_point)
            if label_edges:
                edge_labels[(start_point, end_point)] = f"{self.linepq[(start_point, end_point)][0]:.1f}[MW], " \
                                                        f"{self.linepq[(start_point, end_point)][1]:.1f}[MVAR]"

        # layout_pos = nx.planar_layout(graph, scale=4)
        layout_pos = nx.spring_layout(graph, k=3)
        plt.title("Network Graph: Busses and Lines")
        nx.draw(graph, pos=layout_pos, labels=labels, with_labels=True, node_color=colors,
                node_size=280, node_shape='o', edgecolors="black", font_color="white")
        if label_edges:
            nx.draw_networkx_edge_labels(graph, pos=layout_pos, edge_labels=edge_labels, rotate=False,
                                         font_size=6, verticalalignment="center_baseline", alpha=0.5)

    def export_bus_data(self, printout=True):
        if not self.converge:
            return
        outlines = ["Bus #, Voltage, Angle, Load MW, Load MVAR, Generator MW, Generator MVAR, Injected MVAR\n"]
        for i in range(int(self.nbus)):
            outlines.append(f"{i + 1}, {self.V_mag[i]:.3f}, {self.delta_degrees[i]:.3f}, "
                            f"{self.P_load[i]:.3f}, {self.Q_load[i]:.3f}, {self.P_gen[i]:.3f},"
                            f"{self.Q_gen[i]:.3f}, {self.Q_shunt[i]:.3f}\n")

        with open("bus_outputs.csv", "w+") as csvfile:
            csvfile.writelines(outlines)

        if printout:
            [print(line) for line in outlines]
            print(f"Bus Totals:\n"
                  f"Total Dissipated Power: {self.P_load_total:.3f}[MW], {self.Q_load_total:.3f}[MVAR]\n"
                  f"Total Generated Power: {self.P_gen_total:.3f}[MW], {self.Q_gen_total:.3f}[MVAR], "
                  f"With Capacitors: {self.Q_shunt_total:.3f}[MVAR]")
            print(
                f"Difference of {self.P_gen_total - self.P_load_total:.3f}[MW] "
                f"between generated and dissipated power.\n")

    def print_line_currents(self):
        if not self.converge:
            return

        for pair in list(self.current.keys()):
            abs_current = np.abs(self.current[pair])
            phase_current = np.angle(self.current[pair])
            print(f"{pair[0] + 1} -> {pair[1] + 1}: {abs_current}[A, pu] with angle {phase_current}[rad?]")