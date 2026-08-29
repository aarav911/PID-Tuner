import sys
import numpy as np

from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QSlider,
    QPushButton,
    QGroupBox,
    QComboBox,
)

import pyqtgraph as pg


# ============================================================
# PLOT CONFIGURATION
# ============================================================

pg.setConfigOption("background", "#181825")
pg.setConfigOption("foreground", "#cdd6f4")
pg.setConfigOptions(antialias=False)


# ============================================================
# RING BUFFER
# ============================================================

class RingBuffer:

    def __init__(self, size):
        self.data = np.zeros(size, dtype=np.float32)
        self.size = size
        self.index = 0
        self.count = 0

    def append(self, value):
        self.data[self.index] = value
        self.index = (self.index + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def values(self):
        if self.count == 0:
            return self.data[:0]

        if self.count < self.size:
            return self.data[:self.count]

        return np.concatenate(
            (
                self.data[self.index:],
                self.data[:self.index],
            )
        )


# ============================================================
# PID CONTROLLER
# ============================================================

class PIDController:

    def __init__(
        self,
        kp=5.0,
        ki=1.2,
        kd=0.4,
        output_min=-10.0,
        output_max=10.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.output_min = output_min
        self.output_max = output_max

        self.integral = 0.0
        self.previous_error = 0.0

    def update(
        self,
        setpoint,
        measurement,
        dt,
    ):
        error = setpoint - measurement

        self.integral += error * dt

        derivative = (error - self.previous_error) / dt

        control = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        control = np.clip(
            control,
            self.output_min,
            self.output_max,
        )

        self.previous_error = error

        return control

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0


# ============================================================
# SIMULATED PLANT
# ============================================================

class FirstOrderPlant:
    """
    First-order system:
        dy/dt = (u - y) / tau

    u = controller output
    y = system output
    """

    def __init__(
        self,
        initial_value=0.0,
        time_constant=0.8,
    ):
        self.initial_value = initial_value
        self.time_constant = time_constant
        self.y = initial_value

    def update(
        self,
        control,
        dt,
    ):
        dydt = (control - self.y) / self.time_constant
        self.y += dydt * dt
        return self.y

    def reset(self):
        self.y = self.initial_value


# ============================================================
# SETPOINT GENERATORS (REALISTIC MODES)
# ============================================================

class SetpointEngine:
    """
    Handles multiple setpoint profiles:
      1. Classic Random Steps (with noise)
      2. Kinematic S-Curve (Rate & Accel Bounded Steps)
      3. Stochastic Drift & Turbulences (Ornstein-Uhlenbeck + Harmonics)
    """

    @staticmethod
    def get_step(t, seed=None):
        step_duration = 4.0
        step_index = int(t // step_duration)

        rng = np.random.default_rng(step_index + (42 if seed is None else seed))
        target = rng.uniform(0.5, 3.0)
        noise = np.random.normal(0.0, 0.03) if seed is None else 0.0
        return target + noise

    @staticmethod
    def get_scurve(t, seed=None):
        """Smooth, continuous rate-limited trajectory with 2nd-order dynamics."""
        step_duration = 5.0
        step_index = int(t // step_duration)
        tau = (t % step_duration) / step_duration  # Normalized phase [0, 1)

        # Previous and target positions
        rng_prev = np.random.default_rng(step_index + (42 if seed is None else seed))
        rng_curr = np.random.default_rng(step_index + 1 + (42 if seed is None else seed))

        p_start = rng_prev.uniform(0.5, 3.0)
        p_end = rng_curr.uniform(0.5, 3.0)

        # Quintic smooth polynomial (Zero velocity & acceleration at transitions)
        # S(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5 for tau in [0, 1]
        transition_duration = 1.8  # Seconds to complete the ramp
        phase = min(1.0, (t % step_duration) / transition_duration)
        s_curve = 10.0 * (phase ** 3) - 15.0 * (phase ** 4) + 6.0 * (phase ** 5)

        target = p_start + (p_end - p_start) * s_curve
        noise = np.random.normal(0.0, 0.015) if seed is None else 0.0
        return target + noise

    @staticmethod
    def get_stochastic_drift(t, seed=None):
        """
        Ornstein-Uhlenbeck Mean-Reverting Drift + Dynamic Harmonics + Gusts.
        Simulates atmospheric turbulence / roll guidance drift.
        """
        base_center = 1.8

        # Deterministic multi-frequency harmonics
        w1 = 0.35 * np.sin(0.4 * t)
        w2 = 0.20 * np.sin(1.1 * t + 1.2)
        w3 = 0.10 * np.cos(2.3 * t + 0.7)

        # Discrete gust/shear events every 8 seconds
        gust_index = int(t // 8.0)
        rng_gust = np.random.default_rng(gust_index + (99 if seed is None else seed))
        gust_amp = rng_gust.uniform(-0.8, 0.8)
        gust_phase = (t % 8.0)
        gust = gust_amp * np.exp(-((gust_phase - 2.0) ** 2) / 0.8)

        # Fast continuous jitter
        rng_jitter = np.random.default_rng(int(t * 100) + (13 if seed is None else seed))
        jitter = rng_jitter.normal(0.0, 0.02) if seed is None else 0.0

        return base_center + w1 + w2 + w3 + gust + jitter


# ============================================================
# EXTENDABLE OPTIMIZER FRAMEWORK
# ============================================================

class BaseCostFunction:
    def evaluate(self, errors, controls, outputs, setpoints, dt):
        raise NotImplementedError


class WeightedPerformanceCost(BaseCostFunction):
    def __init__(self, w_error=1.0, w_control=0.005, w_overshoot=2.0):
        self.w_error = w_error
        self.w_control = w_control
        self.w_overshoot = w_overshoot

    def evaluate(self, errors, controls, outputs, setpoints, dt):
        ise = np.sum(errors ** 2) * dt
        control_energy = np.sum(controls ** 2) * dt
        overshoot = np.maximum(0.0, (outputs - setpoints) * np.sign(setpoints))
        overshoot_penalty = np.sum(overshoot ** 2) * dt

        return (
            self.w_error * ise
            + self.w_control * control_energy
            + self.w_overshoot * overshoot_penalty
        )


class BaseOptimizer:
    def step(self, current_params):
        raise NotImplementedError


class FiniteDifferenceGradientDescent(BaseOptimizer):
    def __init__(
        self,
        eval_fn,
        bounds,
        learning_rate=0.4,
        delta=1e-3,
        momentum=0.6,
    ):
        self.eval_fn = eval_fn
        self.bounds = np.array(bounds, dtype=np.float64)
        self.learning_rate = learning_rate
        self.delta = delta
        self.momentum = momentum
        self.velocity = np.zeros(len(bounds), dtype=np.float64)

    def compute_gradient(self, params):
        grad = np.zeros_like(params)
        base_cost = self.eval_fn(params)

        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()

            p_plus[i] += self.delta
            p_minus[i] -= self.delta

            cost_plus = self.eval_fn(p_plus)
            cost_minus = self.eval_fn(p_minus)

            grad[i] = (cost_plus - cost_minus) / (2.0 * self.delta)

        return grad, base_cost

    def step(self, current_params):
        params = np.array(current_params, dtype=np.float64)
        grad, current_cost = self.compute_gradient(params)

        grad_norm = np.linalg.norm(grad)
        grad_scaled = grad / max(1.0, grad_norm / 5.0) if grad_norm > 1e-6 else grad

        self.velocity = self.momentum * self.velocity + self.learning_rate * grad_scaled
        new_params = params - self.velocity
        new_params = np.clip(new_params, self.bounds[:, 0], self.bounds[:, 1])

        return new_params, current_cost


class OptimizerWorker(QThread):
    progress = Signal(int, float, float, float, float)
    finished_opt = Signal(float, float, float)

    def __init__(
        self,
        initial_params,
        bounds,
        mode="Kinematic S-Curve",
        sim_duration=12.0,
        dt=0.01,
        max_iters=40,
        parent=None,
    ):
        super().__init__(parent)
        self.params = np.array(initial_params, dtype=np.float64)
        self.bounds = bounds
        self.mode = mode
        self.sim_duration = sim_duration
        self.dt = dt
        self.max_iters = max_iters
        self.cost_fn = WeightedPerformanceCost()

    def get_setpoint_val(self, t):
        if self.mode == "Kinematic S-Curve":
            return SetpointEngine.get_scurve(t, seed=42)
        elif self.mode == "Stochastic Drift & Gusts":
            return SetpointEngine.get_stochastic_drift(t, seed=42)
        return SetpointEngine.get_step(t, seed=42)

    def simulate_and_evaluate(self, params):
        kp, ki, kd = params
        test_pid = PIDController(kp=kp, ki=ki, kd=kd)
        test_plant = FirstOrderPlant()

        steps = int(self.sim_duration / self.dt)
        errors = np.zeros(steps)
        controls = np.zeros(steps)
        outputs = np.zeros(steps)
        setpoints = np.zeros(steps)

        for step in range(steps):
            t = step * self.dt
            target = self.get_setpoint_val(t)
            control = test_pid.update(setpoint=target, measurement=test_plant.y, dt=self.dt)
            output = test_plant.update(control=control, dt=self.dt)

            errors[step] = target - output
            controls[step] = control
            outputs[step] = output
            setpoints[step] = target

        return self.cost_fn.evaluate(errors, controls, outputs, setpoints, self.dt)

    def run(self):
        optimizer = FiniteDifferenceGradientDescent(
            eval_fn=self.simulate_and_evaluate,
            bounds=self.bounds,
            learning_rate=0.35,
            delta=1e-3,
        )

        current_params = self.params.copy()
        for i in range(self.max_iters):
            new_params, cost = optimizer.step(current_params)
            current_params = new_params
            self.progress.emit(
                i + 1,
                float(current_params[0]),
                float(current_params[1]),
                float(current_params[2]),
                float(cost),
            )
            self.msleep(25)

        self.finished_opt.emit(
            float(current_params[0]),
            float(current_params[1]),
            float(current_params[2]),
        )


# ============================================================
# PID TUNING UI
# ============================================================

class PIDSlider(QWidget):

    def __init__(
        self,
        name,
        minimum,
        maximum,
        value,
        decimals=2,
    ):
        super().__init__()

        self.minimum = minimum
        self.maximum = maximum
        self.decimals = decimals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.name_label = QLabel(name)
        self.name_label.setFixedWidth(35)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setValue(self.value_to_slider(value))

        self.value_label = QLabel()
        self.value_label.setFixedWidth(70)
        self.value_label.setAlignment(Qt.AlignRight)

        self.update_label()

        layout.addWidget(self.name_label)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.value_label)

        self.slider.valueChanged.connect(self.update_label)

    def value_to_slider(self, value):
        ratio = (value - self.minimum) / (self.maximum - self.minimum)
        return int(ratio * 1000)

    def slider_to_value(self):
        ratio = self.slider.value() / 1000.0
        return self.minimum + ratio * (self.maximum - self.minimum)

    def update_label(self):
        value = self.slider_to_value()
        self.value_label.setText(f"{value:.{self.decimals}f}")

    def set_value(self, value):
        self.slider.setValue(self.value_to_slider(value))

    def value(self):
        return self.slider_to_value()


# ============================================================
# MAIN PID TOOL
# ============================================================

class PIDTuningTool(QWidget):

    def __init__(
        self,
        time_window=10.0,
        simulation_rate=100.0,
        render_rate=30.0,
    ):
        super().__init__()

        self.time_window = time_window
        self.simulation_rate = simulation_rate
        self.render_rate = render_rate
        self.dt = 1.0 / simulation_rate

        self.buffer_size = int(np.ceil(time_window * simulation_rate))
        self.t = 0.0
        self.optimizer_worker = None

        # ====================================================
        # DATA BUFFERS
        # ====================================================
        self.time_buffer = RingBuffer(self.buffer_size)
        self.setpoint_buffer = RingBuffer(self.buffer_size)
        self.output_buffer = RingBuffer(self.buffer_size)
        self.control_buffer = RingBuffer(self.buffer_size)
        self.error_buffer = RingBuffer(self.buffer_size)

        # ====================================================
        # PID + PLANT
        # ====================================================
        self.pid = PIDController(
            kp=5.0,
            ki=1.2,
            kd=0.4,
            output_min=-10.0,
            output_max=10.0,
        )

        self.plant = FirstOrderPlant(
            initial_value=0.0,
            time_constant=0.8,
        )

        # ====================================================
        # MAIN LAYOUT
        # ====================================================
        main_layout = QVBoxLayout(self)

        # ====================================================
        # TOP: PID CONTROLS & SETPOINT SELECTION
        # ====================================================
        tuning_box = QGroupBox("Controller & Reference Configuration")
        tuning_layout = QGridLayout(tuning_box)

        # Sliders
        self.kp_slider = PIDSlider("Kp", minimum=0.0, maximum=20.0, value=5.0, decimals=2)
        self.ki_slider = PIDSlider("Ki", minimum=0.0, maximum=10.0, value=1.2, decimals=2)
        self.kd_slider = PIDSlider("Kd", minimum=0.0, maximum=10.0, value=0.4, decimals=2)

        tuning_layout.addWidget(self.kp_slider, 0, 0)
        tuning_layout.addWidget(self.ki_slider, 1, 0)
        tuning_layout.addWidget(self.kd_slider, 2, 0)

        # Setpoint selector dropdown
        self.profile_combo = QComboBox()
        self.profile_combo.addItems([
            "Kinematic S-Curve",
            "Stochastic Drift & Gusts",
            "Classic Random Steps",
        ])
        tuning_layout.addWidget(QLabel("Setpoint Mode:"), 0, 1)
        tuning_layout.addWidget(self.profile_combo, 0, 2)

        # Buttons
        self.reset_button = QPushButton("Reset Simulation")
        self.optimize_button = QPushButton("Find Optimal Parameters")
        self.optimize_button.clicked.connect(self.find_optimal_parameters)

        tuning_layout.addWidget(self.reset_button, 1, 1, 2, 1)
        tuning_layout.addWidget(self.optimize_button, 1, 2, 2, 1)

        main_layout.addWidget(tuning_box)

        # ====================================================
        # MAIN OUTPUT PLOT
        # ====================================================
        self.output_plot = pg.PlotWidget()
        self.output_plot.showGrid(x=True, y=True, alpha=0.25)
        self.output_plot.setLabel("bottom", "Simulation Time", units="s")
        self.output_plot.setLabel("left", "System Output")
        self.output_plot.disableAutoRange()
        self.output_plot.addLegend(offset=(15, 15))

        self.setpoint_curve = self.output_plot.plot(
            name="Setpoint",
            pen=pg.mkPen("#f9e2af", width=2),
        )
        self.output_curve = self.output_plot.plot(
            name="System Output",
            pen=pg.mkPen("#89b4fa", width=2),
        )

        main_layout.addWidget(self.output_plot, stretch=5)

        # ====================================================
        # CONTROL SIGNAL PLOT
        # ====================================================
        self.control_plot = pg.PlotWidget()
        self.control_plot.showGrid(x=True, y=True, alpha=0.25)
        self.control_plot.setLabel("bottom", "Simulation Time", units="s")
        self.control_plot.setLabel("left", "Control Signal")
        self.control_plot.disableAutoRange()
        self.control_plot.addLegend(offset=(15, 15))

        self.control_curve = self.control_plot.plot(
            name="u(t)",
            pen=pg.mkPen("#a6e3a1", width=2),
        )

        main_layout.addWidget(self.control_plot, stretch=2)

        # ====================================================
        # METRICS
        # ====================================================
        metrics_box = QGroupBox("Live Performance")
        metrics_layout = QHBoxLayout(metrics_box)

        self.error_label = QLabel("Error: 0.000")
        self.abs_error_label = QLabel("Abs. Error: 0.000")
        self.control_label = QLabel("Control: 0.000")
        self.output_label = QLabel("Output: 0.000")

        metrics_layout.addWidget(self.error_label)
        metrics_layout.addWidget(self.abs_error_label)
        metrics_layout.addWidget(self.control_label)
        metrics_layout.addWidget(self.output_label)

        main_layout.addWidget(metrics_box)

        # ====================================================
        # CONNECTIONS
        # ====================================================
        self.kp_slider.slider.valueChanged.connect(self.parameters_changed)
        self.ki_slider.slider.valueChanged.connect(self.parameters_changed)
        self.kd_slider.slider.valueChanged.connect(self.parameters_changed)
        self.reset_button.clicked.connect(self.reset)

        # ====================================================
        # SIMULATION TIMER
        # ====================================================
        self.simulation_timer = QTimer(self)
        self.simulation_timer.timeout.connect(self.simulation_step)
        self.simulation_timer.start(round(1000.0 / simulation_rate))

        # ====================================================
        # RENDER TIMER
        # ====================================================
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.render)
        self.render_timer.start(round(1000.0 / render_rate))

    def get_active_setpoint(self, t):
        mode = self.profile_combo.currentText()
        if mode == "Kinematic S-Curve":
            return SetpointEngine.get_scurve(t)
        elif mode == "Stochastic Drift & Gusts":
            return SetpointEngine.get_stochastic_drift(t)
        return SetpointEngine.get_step(t)

    # ========================================================
    # UPDATE PID PARAMETERS
    # ========================================================

    def parameters_changed(self):
        self.pid.kp = self.kp_slider.value()
        self.pid.ki = self.ki_slider.value()
        self.pid.kd = self.kd_slider.value()

    # ========================================================
    # SIMULATION STEP
    # ========================================================

    def simulation_step(self):
        self.t += self.dt
        target = self.get_active_setpoint(self.t)
        control = self.pid.update(setpoint=target, measurement=self.plant.y, dt=self.dt)
        output = self.plant.update(control=control, dt=self.dt)
        error = target - output

        self.time_buffer.append(self.t)
        self.setpoint_buffer.append(target)
        self.output_buffer.append(output)
        self.control_buffer.append(control)
        self.error_buffer.append(error)

        self.error_label.setText(f"Error: {error:+.3f}")
        self.abs_error_label.setText(f"Abs. Error: {abs(error):.3f}")
        self.control_label.setText(f"Control: {control:+.3f}")
        self.output_label.setText(f"Output: {output:+.3f}")

    # ========================================================
    # RENDER
    # ========================================================

    def render(self):
        if self.time_buffer.count == 0:
            return

        time_data = self.time_buffer.values()
        setpoint_data = self.setpoint_buffer.values()
        output_data = self.output_buffer.values()
        control_data = self.control_buffer.values()

        self.setpoint_curve.setData(time_data, setpoint_data)
        self.output_curve.setData(time_data, output_data)
        self.control_curve.setData(time_data, control_data)

        x_max = self.t
        x_min = max(0.0, x_max - self.time_window)
        if self.t < self.time_window:
            x_max = self.time_window

        right_padding = 0.5
        self.output_plot.setXRange(x_min, x_max + right_padding, padding=0)
        self.control_plot.setXRange(x_min, x_max + right_padding, padding=0)

        minimum = min(float(setpoint_data.min()), float(output_data.min()))
        maximum = max(float(setpoint_data.max()), float(output_data.max()))

        if maximum - minimum < 1e-6:
            center = (minimum + maximum) * 0.5
            minimum = center - 0.5
            maximum = center + 0.5

        margin = max(0.2, 0.1 * (maximum - minimum))
        self.output_plot.setYRange(minimum - margin, maximum + margin, padding=0)

        control_min = float(control_data.min())
        control_max = float(control_data.max())
        control_margin = max(0.5, 0.1 * (control_max - control_min))
        self.control_plot.setYRange(control_min - control_margin, control_max + control_margin, padding=0)

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        self.simulation_timer.stop()
        self.render_timer.stop()

        self.t = 0.0
        self.plant.reset()
        self.pid.reset()

        self.time_buffer = RingBuffer(self.buffer_size)
        self.setpoint_buffer = RingBuffer(self.buffer_size)
        self.output_buffer = RingBuffer(self.buffer_size)
        self.control_buffer = RingBuffer(self.buffer_size)
        self.error_buffer = RingBuffer(self.buffer_size)

        self.output_plot.setXRange(0, self.time_window, padding=0)
        self.control_plot.setXRange(0, self.time_window, padding=0)

        self.simulation_timer.start(round(1000.0 / self.simulation_rate))
        self.render_timer.start(round(1000.0 / self.render_rate))

    # ========================================================
    # OPTIMIZATION INTEGRATION
    # ========================================================

    def find_optimal_parameters(self):
        if self.optimizer_worker and self.optimizer_worker.isRunning():
            return

        initial_params = [
            self.kp_slider.value(),
            self.ki_slider.value(),
            self.kd_slider.value(),
        ]
        bounds = [
            (self.kp_slider.minimum, self.kp_slider.maximum),
            (self.ki_slider.minimum, self.ki_slider.maximum),
            (self.kd_slider.minimum, self.kd_slider.maximum),
        ]

        active_mode = self.profile_combo.currentText()

        self.optimize_button.setEnabled(False)
        self.optimize_button.setText("Optimizing...")

        self.optimizer_worker = OptimizerWorker(
            initial_params=initial_params,
            bounds=bounds,
            mode=active_mode,
            sim_duration=12.0,
            dt=0.01,
            max_iters=40,
        )

        self.optimizer_worker.progress.connect(self._on_optimizer_progress)
        self.optimizer_worker.finished_opt.connect(self._on_optimizer_finished)
        self.optimizer_worker.start()

    def _on_optimizer_progress(self, iteration, kp, ki, kd, cost):
        self.kp_slider.set_value(kp)
        self.ki_slider.set_value(ki)
        self.kd_slider.set_value(kd)
        self.optimize_button.setText(f"Iter {iteration} (J={cost:.2f})")

    def _on_optimizer_finished(self, kp, ki, kd):
        self.kp_slider.set_value(kp)
        self.ki_slider.set_value(ki)
        self.kd_slider.set_value(kd)
        self.optimize_button.setText("Find Optimal Parameters")
        self.optimize_button.setEnabled(True)


# ============================================================
# MAIN WINDOW
# ============================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PID Controller — Tuning & Visualization")
        self.resize(1200, 850)

        self.tool = PIDTuningTool(
            time_window=10.0,
            simulation_rate=100.0,
            render_rate=30.0,
        )
        self.setCentralWidget(self.tool)


# ============================================================
# APPLICATION
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())