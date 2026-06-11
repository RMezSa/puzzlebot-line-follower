#!/usr/bin/env python3
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Float32

import numpy as np
import skfuzzy as fuzz


LABELS = {0: "Sin color", 1: "Rojo", 2: "Amarillo", 3: "Verde"}

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONTROLADOR SUGENO
# ─────────────────────────────────────────────────────────────────────────────

class SugenoLineController:
    def __init__(self):
        # Universo de entrada
        self.E_universe = np.arange(-320, 321, 1)

        # Funciones de membresía de la entrada E
        self.mucho_error_negativo = fuzz.zmf(
            self.E_universe,
            -250,
            -70
        )

        self.poco_error_negativo = fuzz.trimf(
            self.E_universe,
            [-100, -80, -8]
        )

        self.sin_error = fuzz.trimf(
            self.E_universe,
            [-15, 0, 15]
        )

        self.poco_error_positivo = fuzz.trimf(
            self.E_universe,
            [8, 80, 100]
        )

        self.mucho_error_positivo = fuzz.smf(
            self.E_universe,
            70,
            250
        )

        # Consecuentes Sugeno de orden cero
        #
        # Equivalencia aproximada con tu Mamdani:
        #
        # mucho_error_negativo -> V poca,   W mucho positivo
        # poco_error_negativo  -> V normal, W poco positivo
        # sin_error            -> V normal, W nada
        # poco_error_positivo  -> V normal, W poco negativo
        # mucho_error_positivo -> V poca,   W mucho negativo

        self.v_values = np.array([
            0.65,   # mucho_error_negativo
            1.20,   # poco_error_negativo
            1.50,   # sin_error
            1.20,   # poco_error_positivo
            0.65    # mucho_error_positivo
        ])

        self.w_values = np.array([
            2.30,    # mucho_error_negativo
            0.80,    # poco_error_negativo
            0.00,    # sin_error
            -0.80,   # poco_error_positivo
            -2.30    # mucho_error_positivo
        ])

    def compute(self, error_px: float):
        e = float(np.clip(error_px, -320.0, 320.0))

        # Grados de activación
        mu1 = fuzz.interp_membership(
            self.E_universe,
            self.mucho_error_negativo,
            e
        )

        mu2 = fuzz.interp_membership(
            self.E_universe,
            self.poco_error_negativo,
            e
        )

        mu3 = fuzz.interp_membership(
            self.E_universe,
            self.sin_error,
            e
        )

        mu4 = fuzz.interp_membership(
            self.E_universe,
            self.poco_error_positivo,
            e
        )

        mu5 = fuzz.interp_membership(
            self.E_universe,
            self.mucho_error_positivo,
            e
        )

        weights = np.array([mu1, mu2, mu3, mu4, mu5])
        weight_sum = np.sum(weights)

        if weight_sum <= 1e-9:
            return 0.0, 0.0, e

        # Promedio ponderado Sugeno
        v = np.sum(weights * self.v_values) / weight_sum
        w = np.sum(weights * self.w_values) / weight_sum

        return float(v), float(w), e


# ─────────────────────────────────────────────────────────────────────────────
# 2. NODO ROS2
# ─────────────────────────────────────────────────────────────────────────────

class LineFollowerController(Node):
    def __init__(self):
        super().__init__("line_follower_controller")
        self.get_logger().info("Line follower controller Sugeno started...")

        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 1)

        self.create_subscription(
            Int32,
            "/semaforo",
            self.cb_traffic_light,
            10
        )

        self.create_subscription(
            Float32,
            "/line_error_px",
            self.cb_line_error,
            10
        )

        # Controlador Sugeno
        self.sugeno = SugenoLineController()

        # Parámetros de aceleración
        self.declare_parameter("accel_rate_up", 0.05)
        self.declare_parameter("accel_rate_down", 0.30)
        self.declare_parameter("soft_start_v_threshold", 0.08)
        self.declare_parameter("soft_start_min_factor", 0.25)

        # Parámetros de línea
        self.declare_parameter("line_lost_timeout", 0.5)
        self.declare_parameter("line_lost_speed", 0.10)

        # Escalamiento de salidas Sugeno
        #
        # Sugeno entrega:
        #   V ≈ 0.0 a 1.5
        #   W ≈ -2.3 a 2.3
        #
        # Estos factores lo bajan a valores razonables para el Puzzlebot.
        self.declare_parameter("sugeno_v_scale", 0.10)
        self.declare_parameter("sugeno_w_scale", 0.25)

        # Parámetros del semáforo
        self.declare_parameter("tl_buffer_size", 20)
        self.declare_parameter("tl_yellow_ratio", 0.5)
        self.declare_parameter("tl_red_ratio", 0.0)

        self._cache_params()
        self.add_on_set_parameters_callback(self._on_params_changed)

        self.current_v = 0.0
        self.line_error = None
        self.prev_error = 0.0

        self.last_error_t = self.get_clock().now()
        self.prev_control_t = self.get_clock().now()

        # Estado del semáforo
        self.traffic_status = 0
        self.detected_color = 0
        self.prev_color = None
        self.raw_buffer = deque(maxlen=int(self.p_tl_buffer_size))

        self.create_timer(0.1, self.control_loop)

    def _cache_params(self):
        self.p_accel_rate_up = float(
            self.get_parameter("accel_rate_up").value
        )

        self.p_accel_rate_down = float(
            self.get_parameter("accel_rate_down").value
        )

        self.p_ss_v_threshold = float(
            self.get_parameter("soft_start_v_threshold").value
        )

        self.p_ss_min_factor = float(
            self.get_parameter("soft_start_min_factor").value
        )

        self.p_line_lost_timeout = float(
            self.get_parameter("line_lost_timeout").value
        )

        self.p_line_lost_speed = float(
            self.get_parameter("line_lost_speed").value
        )

        self.p_sugeno_v_scale = float(
            self.get_parameter("sugeno_v_scale").value
        )

        self.p_sugeno_w_scale = float(
            self.get_parameter("sugeno_w_scale").value
        )

        self.p_tl_buffer_size = int(
            self.get_parameter("tl_buffer_size").value
        )

        self.p_tl_yellow_ratio = float(
            self.get_parameter("tl_yellow_ratio").value
        )

        self.p_tl_red_ratio = float(
            self.get_parameter("tl_red_ratio").value
        )

    def _on_params_changed(self, params):
        self._cache_params()

        self.raw_buffer = deque(
            self.raw_buffer,
            maxlen=int(self.p_tl_buffer_size)
        )

        self.get_logger().info(
            f"Params updated → "
            f"sugeno_v_scale={self.p_sugeno_v_scale}  "
            f"sugeno_w_scale={self.p_sugeno_w_scale}"
        )

        return SetParametersResult(successful=True)

    def _valid_transition(self, current, new):
        """
        Ciclo válido:
        Verde(3) → Amarillo(2) → Rojo(1) → Verde(3)
        """

        next_in_cycle = {3: 2, 2: 1, 1: 3}

        if current == new:
            return True

        if current == 0:
            if self.prev_color is None:
                return new in (1, 2, 3)

            return new == next_in_cycle.get(self.prev_color)

        if new == 0:
            return True

        return new == next_in_cycle.get(current)

    def cb_traffic_light(self, msg):
        self.raw_buffer.append(msg.data)

        counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for v in self.raw_buffer:
            counts[v] = counts.get(v, 0) + 1

        majority_color = max(counts, key=counts.get)

        if counts[majority_color] <= len(self.raw_buffer) // 2:
            majority_color = 0

        if majority_color == self.detected_color:
            return

        if not self._valid_transition(self.detected_color, majority_color):
            self.get_logger().warn(
                f"Transición inválida ignorada: "
                f"{LABELS[self.detected_color]} → {LABELS[majority_color]}"
            )
            return

        self.detected_color = majority_color

        if majority_color == 0:
            self.get_logger().info(
                f"Semáforo: Sin color | Estado efectivo mantenido: "
                f"{LABELS[self.traffic_status]}"
            )
        else:
            self.prev_color = majority_color
            self.traffic_status = majority_color
            self.get_logger().info(f"Semáforo: {LABELS[majority_color]}")

    def cb_line_error(self, msg):
        val = float(msg.data)

        if not math.isnan(val):
            self.line_error = val
            self.last_error_t = self.get_clock().now()

    def _traffic_light_factor(self) -> float:
        """
        Factor multiplicativo sobre velocidad lineal.

        Verde / sin color: 1.0
        Amarillo: tl_yellow_ratio
        Rojo: tl_red_ratio
        """

        if self.traffic_status == 1:
            return self.p_tl_red_ratio

        if self.traffic_status == 2:
            return self.p_tl_yellow_ratio

        return 1.0

    def _compute_sugeno_control(self, error_px: float):
        """
        Calcula velocidades usando Sugeno.

        Sugeno entrega valores teóricos:
            v_raw ≈ 0.65 a 1.50
            w_raw ≈ -2.30 a 2.30

        Después se escalan para el robot:
            target_v = v_raw * sugeno_v_scale
            angular_z = w_raw * sugeno_w_scale
        """

        v_raw, w_raw, error_clamped = self.sugeno.compute(error_px)

        target_v = v_raw * self.p_sugeno_v_scale
        angular_z = w_raw * self.p_sugeno_w_scale

        return target_v, angular_z, v_raw, w_raw, error_clamped

    def control_loop(self):
        now = self.get_clock().now()

        dt_sec = (now - self.prev_control_t).nanoseconds / 1e9
        self.prev_control_t = now

        if dt_sec <= 0.0:
            dt_sec = 0.1

        # 1. Detectar pérdida de línea
        age = (now - self.last_error_t).nanoseconds / 1e9
        line_lost = age > self.p_line_lost_timeout

        # 2. Error activo
        if not line_lost and self.line_error is not None:
            active_error = self.line_error
        else:
            active_error = self.prev_error

        self.prev_error = active_error

        # 3. Control Sugeno
        target_v, angular_z, v_raw, w_raw, error_clamped = (
            self._compute_sugeno_control(active_error)
        )

        # 4. Si se perdió la línea, reducir velocidad
        if line_lost:
            target_v *= self.p_line_lost_speed

        # 5. Aplicar factor del semáforo
        tl_factor = self._traffic_light_factor()
        target_v *= tl_factor

        # 6. Rampa de aceleración asimétrica con arranque suave
        error_v = target_v - self.current_v

        if error_v >= 0.0:
            if self.p_ss_v_threshold > 1e-6:
                ratio = min(self.current_v / self.p_ss_v_threshold, 1.0)
            else:
                ratio = 1.0

            ss_factor = (
                self.p_ss_min_factor
                + (1.0 - self.p_ss_min_factor) * ratio
            )

            max_step = self.p_accel_rate_up * ss_factor * dt_sec

        else:
            max_step = self.p_accel_rate_down * dt_sec

        if abs(error_v) <= max_step:
            self.current_v = target_v
        else:
            if error_v > 0:
                self.current_v += max_step
            else:
                self.current_v -= max_step

        # 7. Si está en rojo y se detuvo, no girar en sitio
        if self.traffic_status == 1 and self.current_v <= 1e-3:
            self.current_v = 0.0
            angular_z = 0.0

        # 8. Si no hay velocidad lineal, tampoco girar
        if self.current_v <= 1e-3:
            angular_z = 0.0

        # 9. Publicar comando
        cmd = Twist()
        cmd.linear.x = max(0.0, self.current_v)
        cmd.angular.z = angular_z

        self.pub_cmd.publish(cmd)

        self.get_logger().info(
            f"v={self.current_v:.3f}  "
            f"w={angular_z:.3f}  "
            f"error={active_error:.1f}px  "
            f"error_clamped={error_clamped:.1f}px  "
            f"v_raw={v_raw:.3f}  "
            f"w_raw={w_raw:.3f}  "
            f"lost={line_lost}  "
            f"semaforo={LABELS.get(self.traffic_status, '?')}  "
            f"tl_factor={tl_factor:.2f}",
            throttle_duration_sec=0.5
        )


def main(args=None):
    rclpy.init(args=args)

    node = LineFollowerController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()