#!/usr/bin/env python3
"""
LineFollowerController — ROS 2 Humble
Control difuso (Mamdani) + State Machine de cruce de zebra.

Estados:
  FOLLOW_LINE   — seguimiento normal con fuzzy
  SIGN_LATCHED  — señal latcheada, sigue línea normalmente
  STOP_SIGN     — alto por señal stop durante 3 segundos
  STOP_PRE      — pausa antes de maniobra
  CROSSING      — maniobra open-loop según señal latcheada
  STOP_POST     — pausa después de maniobra
  RECOVERING    — re-adquisición de línea tras cruce

Semáforo:
  red    = 1 → stop
  yellow = 2 → reduce velocidad
  green  = 3 → go
  none   = 0 → go

Latches:
  - Usa flechas detectadas por YOLO.
  - Si no hay flecha detectada, default = flecha_derecho.
"""

import math
import json
from collections import deque, Counter
from enum import Enum

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Float32, Bool, String

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control


LABELS = {0: "Sin color", 1: "Rojo", 2: "Amarillo", 3: "Verde"}

LATCHABLE_SIGNS = {"flecha_derecha", "flecha_izquierda", "flecha_derecho"}
STOP_SIGNS = {"stop"}


class State(Enum):
    FOLLOW_LINE  = "FOLLOW_LINE"
    SIGN_LATCHED = "SIGN_LATCHED"
    STOP_SIGN    = "STOP_SIGN"
    STOP_PRE     = "STOP_PRE"
    CROSSING     = "CROSSING"
    STOP_POST    = "STOP_POST"
    RECOVERING   = "RECOVERING"


def build_fuzzy_system():
    E = control.Antecedent(np.arange(-320, 321, 1), "E")
    V = control.Consequent(np.arange(0.0, 1.51, 0.01), "Velocidad_lineal")
    W = control.Consequent(np.arange(-3.0, 3.01, 0.01), "Velocidad_angular")

    E["mucho_error_negativo"] = fuzz.zmf(E.universe, -250, -70)
    E["poco_error_negativo"]  = fuzz.trimf(E.universe, [-100, -80, -8])
    E["sin_error"]            = fuzz.trimf(E.universe, [-15, 0, 15])
    E["poco_error_positivo"]  = fuzz.trimf(E.universe, [8, 80, 100])
    E["mucho_error_positivo"] = fuzz.smf(E.universe, 70, 250)

    V["nada"]   = fuzz.trimf(V.universe, [0.0, 0.0, 0.4])
    V["poca"]   = fuzz.trimf(V.universe, [0.4, 0.65, 0.9])
    V["normal"] = fuzz.trapmf(V.universe, [0.9, 1.5, 1.5, 1.5])

    W["mucho_negativo"] = fuzz.trimf(W.universe, [-3.0, -3.0, -1.5])
    W["poco_negativo"]  = fuzz.trimf(W.universe, [-1.5, -0.8, -0.3])
    W["nada"]           = fuzz.trimf(W.universe, [-0.3, 0.0, 0.3])
    W["poco_positivo"]  = fuzz.trimf(W.universe, [0.3, 0.8, 1.5])
    W["mucho_positivo"] = fuzz.trimf(W.universe, [1.5, 3.0, 3.0])

    r1 = control.Rule(E["mucho_error_negativo"], (V["poca"],   W["mucho_positivo"]))
    r2 = control.Rule(E["poco_error_negativo"],  (V["normal"], W["poco_positivo"]))
    r3 = control.Rule(E["sin_error"],            (V["normal"], W["nada"]))
    r4 = control.Rule(E["poco_error_positivo"],  (V["normal"], W["poco_negativo"]))
    r5 = control.Rule(E["mucho_error_positivo"], (V["poca"],   W["mucho_negativo"]))

    sistema = control.ControlSystem([r1, r2, r3, r4, r5])
    return control.ControlSystemSimulation(sistema)


class LineFollowerController(Node):
    def __init__(self):
        super().__init__("line_follower_controller")
        self.get_logger().info("Line follower controller started...")

        self.pub_cmd      = self.create_publisher(Twist, "/cmd_vel",         1)
        self.pub_crossing = self.create_publisher(Bool,  "/crossing_active", 1)
        self._crossing_active = False

        self.create_subscription(Int32,   "/semaforo",        self.cb_traffic_light, 10)
        self.create_subscription(Float32, "/line_error_px",   self.cb_line_error,    10)
        self.create_subscription(Bool,    "/zebra_detected",  self.cb_zebra,         10)
        self.create_subscription(String,  "/sign_detections", self.cb_signs,         10)

        self.fuzzy_sim = build_fuzzy_system()

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 1 — ANTI-OSCILACIÓN  (los más importantes del tuning)   ║
        # ║  Estos tres atacan directamente el zigzag del robot.            ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("error_filter_alpha", 0.45)  # suaviza el error de línea (0=muy suave, 1=crudo)
        self.declare_parameter("angular_slew_rate",  1.40)  # máximo cambio de giro por segundo (evita latigazos)
        self.declare_parameter("max_angular_z",      0.55)  # límite absoluto de giro rad/s

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 2 — VELOCIDAD EN CURVAS                                  ║
        # ║  Evitan que intente corregir a alta velocidad en curvas.         ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("curve_slowdown_error", 90.0)  # error (px) a partir del cual empieza a bajar v
        self.declare_parameter("curve_min_v_factor",   0.55)  # en curva máxima, baja hasta el 55% de la v normal

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 3 — ESCALA FUZZY                                         ║
        # ║  Controlan la agresividad general del controlador.               ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("fuzzy_v_scale", 0.10)  # multiplica la velocidad lineal que da el fuzzy
        self.declare_parameter("fuzzy_w_scale", 0.20)  # multiplica la velocidad angular que da el fuzzy
        self.declare_parameter("error_gain",    0.75)  # reduce cuánto "siente" el error (evita paranoia)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 4 — RAMPA DE VELOCIDAD LINEAL                            ║
        # ║  Controlan la aceleración/frenada del robot.                     ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("accel_rate_up",          0.05)  # qué tan rápido acelera (bajo = suave)
        self.declare_parameter("accel_rate_down",        0.30)  # qué tan rápido frena (alto = frena rápido)
        self.declare_parameter("soft_start_v_threshold", 0.08)  # velocidad hasta la que aplica soft-start
        self.declare_parameter("soft_start_min_factor",  0.25)  # factor mínimo de aceleración en soft-start

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 5 — COMPORTAMIENTO SI SE PIERDE LA LÍNEA                 ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("line_lost_timeout", 0.5)   # segundos sin señal para declarar línea perdida
        self.declare_parameter("line_lost_speed",   0.10)  # velocidad de búsqueda cuando pierde la línea

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 6 — SEMÁFORO                                             ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("traffic_light_enabled",  True)
        self.declare_parameter("yellow_slowdown_factor", 0.50)  # amarillo = 50% de velocidad
        self.declare_parameter("tl_buffer_size",         10)    # ventana de frames para filtrar color
        self.declare_parameter("tl_red_ratio",           0.50)  # fracción del buffer para confirmar rojo
        self.declare_parameter("tl_yellow_ratio",        0.50)
        self.declare_parameter("tl_green_ratio",         0.50)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 7 — SEÑAL STOP                                           ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("stop_sign_enabled",  True)
        self.declare_parameter("stop_sign_duration", 3.0)  # segundos de alto al ver stop

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 8 — MANIOBRAS OPEN-LOOP EN CRUCE                        ║
        # ║  Cada maniobra: fwd1 (avanza) → turn (gira) → fwd2 (avanza)    ║
        # ╚══════════════════════════════════════════════════════════════════╝
        # Flecha derecha → gira derecha
        self.declare_parameter("cross_right_fwd1_v",  0.12)
        self.declare_parameter("cross_right_fwd1_t",  3.5)
        self.declare_parameter("cross_right_turn_w", -0.6)  # negativo = gira derecha
        self.declare_parameter("cross_right_turn_t",  2.6)
        self.declare_parameter("cross_right_fwd2_v",  0.12)
        self.declare_parameter("cross_right_fwd2_t",  2.0)

        # Flecha izquierda → gira izquierda
        self.declare_parameter("cross_left_fwd1_v",  0.12)
        self.declare_parameter("cross_left_fwd1_t",  3.5)
        self.declare_parameter("cross_left_turn_w",   0.6)  # positivo = gira izquierda
        self.declare_parameter("cross_left_turn_t",   2.6)
        self.declare_parameter("cross_left_fwd2_v",  0.12)
        self.declare_parameter("cross_left_fwd2_t",  2.0)

        # Flecha derecho → sigue recto
        self.declare_parameter("cross_straight_v", 0.12)
        self.declare_parameter("cross_straight_t", 5.0)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 9 — RE-ADQUISICIÓN DE LÍNEA TRAS CRUCE                  ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("recover_stable_frames", 5)   # frames consecutivos con línea para salir de RECOVERING
        self.declare_parameter("recover_timeout",       5.0) # timeout máximo si nunca re-adquiere

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 10 — LATCH DE SEÑAL                                      ║
        # ║  Qué flecha ejecutar al cruzar la zebra.                        ║
        # ╚══════════════════════════════════════════════════════════════════╝
        # perma_latch: "none" = usa YOLO | "flecha_derecha/izquierda/derecho" = forzar (debug)
        self.declare_parameter("perma_latch",   "none")
        self.declare_parameter("default_latch", "flecha_derecho")  # si YOLO no vio flecha, va recto

        # ── Pausa antes/después de maniobra (debug) ──────────────────────
        self.declare_parameter("debug_stop_duration", 0.1)

        self._cache_params()
        self.add_on_set_parameters_callback(self._on_params_changed)

        # ── Estado fuzzy/línea ───────────────────────────────────────────────
        self.current_v      = 0.0
        self.current_w      = 0.0
        self.line_error     = None
        self.prev_error     = 0.0
        self.filtered_error = 0.0
        self.last_error_t   = self.get_clock().now()
        self.prev_control_t = self.get_clock().now()

        # Último comando publicado
        self.last_v = 0.0
        self.last_w = 0.0

        # ── Estado semáforo ──────────────────────────────────────────────────
        self.detected_color   = 0
        self.raw_buffer       = deque(maxlen=int(self.p_tl_buffer_size))
        self._tl_latched_stop = False   # True = stopped by red, waiting for green/yellow

        # ── Estado de señal STOP ─────────────────────────────────────────────
        self.stop_sign_active   = False
        self.stop_sign_consumed = False
        self._stop_sign_t0      = None

        # ── State machine ────────────────────────────────────────────────────
        self.state        = State.SIGN_LATCHED
        self.latched_sign = self.p_default_latch
        self.zebra_active = False

        self.zebra_consumed = False

        self._maneuver_steps = []
        self._maneuver_idx   = 0
        self._maneuver_t0    = None

        self._stop_t0 = None

        self._recover_stable = 0
        self._recover_t0     = None

        self.create_timer(0.1, self.control_loop)

    # ─────────────────────────────────────────────────────────────────────────
    # Caché de parámetros
    # ─────────────────────────────────────────────────────────────────────────
    def _cache_params(self):
        self.p_accel_rate_up     = float(self.get_parameter("accel_rate_up").value)
        self.p_accel_rate_down   = float(self.get_parameter("accel_rate_down").value)
        self.p_ss_v_threshold    = float(self.get_parameter("soft_start_v_threshold").value)
        self.p_ss_min_factor     = float(self.get_parameter("soft_start_min_factor").value)

        self.p_line_lost_timeout = float(self.get_parameter("line_lost_timeout").value)
        self.p_line_lost_speed   = float(self.get_parameter("line_lost_speed").value)

        self.p_fuzzy_v_scale = float(self.get_parameter("fuzzy_v_scale").value)
        self.p_fuzzy_w_scale = float(self.get_parameter("fuzzy_w_scale").value)
        self.p_error_gain    = float(self.get_parameter("error_gain").value)

        self.p_error_filter_alpha = float(self.get_parameter("error_filter_alpha").value)
        self.p_angular_slew_rate  = float(self.get_parameter("angular_slew_rate").value)
        self.p_max_angular_z      = float(self.get_parameter("max_angular_z").value)

        self.p_curve_slowdown_error = float(self.get_parameter("curve_slowdown_error").value)
        self.p_curve_min_v_factor   = float(self.get_parameter("curve_min_v_factor").value)

        self.p_traffic_light_enabled  = bool(self.get_parameter("traffic_light_enabled").value)
        self.p_yellow_slowdown_factor = float(self.get_parameter("yellow_slowdown_factor").value)
        self.p_tl_buffer_size         = int(self.get_parameter("tl_buffer_size").value)
        self.p_tl_red_ratio           = float(self.get_parameter("tl_red_ratio").value)
        self.p_tl_yellow_ratio        = float(self.get_parameter("tl_yellow_ratio").value)
        self.p_tl_green_ratio         = float(self.get_parameter("tl_green_ratio").value)

        self.p_stop_sign_enabled  = bool(self.get_parameter("stop_sign_enabled").value)
        self.p_stop_sign_duration = float(self.get_parameter("stop_sign_duration").value)

        self.p_cross_right = [
            (
                float(self.get_parameter("cross_right_fwd1_v").value),
                0.0,
                float(self.get_parameter("cross_right_fwd1_t").value),
            ),
            (
                0.0,
                float(self.get_parameter("cross_right_turn_w").value),
                float(self.get_parameter("cross_right_turn_t").value),
            ),
            (
                float(self.get_parameter("cross_right_fwd2_v").value),
                0.0,
                float(self.get_parameter("cross_right_fwd2_t").value),
            ),
        ]

        self.p_cross_left = [
            (
                float(self.get_parameter("cross_left_fwd1_v").value),
                0.0,
                float(self.get_parameter("cross_left_fwd1_t").value),
            ),
            (
                0.0,
                float(self.get_parameter("cross_left_turn_w").value),
                float(self.get_parameter("cross_left_turn_t").value),
            ),
            (
                float(self.get_parameter("cross_left_fwd2_v").value),
                0.0,
                float(self.get_parameter("cross_left_fwd2_t").value),
            ),
        ]

        self.p_cross_straight = [
            (
                float(self.get_parameter("cross_straight_v").value),
                0.0,
                float(self.get_parameter("cross_straight_t").value),
            ),
        ]

        self.p_recover_stable_frames = int(self.get_parameter("recover_stable_frames").value)
        self.p_recover_timeout       = float(self.get_parameter("recover_timeout").value)

        raw_pl = str(self.get_parameter("perma_latch").value).strip().lower()
        self.p_perma_latch = raw_pl if raw_pl in LATCHABLE_SIGNS else None

        raw_default = str(self.get_parameter("default_latch").value).strip().lower()
        self.p_default_latch = raw_default if raw_default in LATCHABLE_SIGNS else "flecha_derecho"

        self.p_debug_stop_duration = float(self.get_parameter("debug_stop_duration").value)

    def _on_params_changed(self, _params):
        self._cache_params()
        self.raw_buffer = deque(self.raw_buffer, maxlen=int(self.p_tl_buffer_size))

        self.get_logger().info(
            f"Params updated → "
            f"error_gain={self.p_error_gain}  "
            f"fuzzy_v_scale={self.p_fuzzy_v_scale}  "
            f"fuzzy_w_scale={self.p_fuzzy_w_scale}  "
            f"error_filter_alpha={self.p_error_filter_alpha}  "
            f"angular_slew_rate={self.p_angular_slew_rate}  "
            f"max_angular_z={self.p_max_angular_z}  "
            f"traffic_light_enabled={self.p_traffic_light_enabled}  "
            f"yellow_slowdown_factor={self.p_yellow_slowdown_factor}  "
            f"stop_sign_enabled={self.p_stop_sign_enabled}  "
            f"perma_latch={self.p_perma_latch}  "
            f"default_latch={self.p_default_latch}"
        )

        return SetParametersResult(successful=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────
    def cb_traffic_light(self, msg):
        """
        /semaforo:
          0 = ninguno
          1 = rojo
          2 = amarillo
          3 = verde
        """
        val = int(msg.data)
        if val not in (0, 1, 2, 3):
            val = 0

        self.raw_buffer.append(val)

        counts = Counter(self.raw_buffer)

        n = max(1, len(self.raw_buffer))
        red_ratio    = counts[1] / n
        yellow_ratio = counts[2] / n
        green_ratio  = counts[3] / n

        # Prioridad: rojo > amarillo > verde > none
        if red_ratio >= self.p_tl_red_ratio:
            new_color = 1
        elif yellow_ratio >= self.p_tl_yellow_ratio:
            new_color = 2
        elif green_ratio >= self.p_tl_green_ratio:
            new_color = 3
        else:
            new_color = 0

        if new_color == self.detected_color:
            return

        self.detected_color = new_color

        self.get_logger().info(
            f"[Semáforo] {LABELS.get(new_color, '?')} "
            f"(raw={val}, R={red_ratio:.2f}, Y={yellow_ratio:.2f}, G={green_ratio:.2f})",
            throttle_duration_sec=1.0,
        )

    def cb_line_error(self, msg):
        val = float(msg.data)

        if not math.isnan(val):
            self.line_error   = val
            self.last_error_t = self.get_clock().now()

    def cb_zebra(self, msg):
        new_val = bool(msg.data)

        if not new_val:
            self.zebra_consumed = False

        self.zebra_active = new_val

    def cb_signs(self, msg):
        """
        Procesa:
          - stop: activa rutina de alto 3 s
          - flechas: latch normal si perma_latch está apagado

        Aunque perma_latch esté activo, sí procesamos STOP.
        """
        try:
            detections = json.loads(msg.data)
        except Exception:
            return

        # ── STOP sign ────────────────────────────────────────────────────────
        stop_candidates = [
            d for d in detections
            if str(d.get("class_name", "")).lower() in STOP_SIGNS
        ]

        if stop_candidates:
            self.stop_sign_active = True
        else:
            self.stop_sign_active = False
            self.stop_sign_consumed = False

        # ── Si perma_latch está activo, no usamos flechas de YOLO ─────────────
        if self.p_perma_latch is not None:
            return

        if self.state not in (State.FOLLOW_LINE, State.SIGN_LATCHED):
            return

        candidates = [
            d for d in detections
            if d.get("class_name") in LATCHABLE_SIGNS
        ]

        if not candidates:
            return

        # flecha_derecho (straight) is frequently confused with the turn arrows.
        # When a turn arrow is also detected, the turn is almost certainly correct.
        turn_candidates = [
            d for d in candidates
            if d["class_name"] in {"flecha_derecha", "flecha_izquierda"}
        ]
        if turn_candidates:
            candidates = turn_candidates

        best = max(candidates, key=lambda d: d.get("bbox_area", 0))
        new_sign = best["class_name"]

        if new_sign != self.latched_sign:
            self.latched_sign = new_sign
            self.state        = State.SIGN_LATCHED

            self.get_logger().info(f"[SM] Señal latcheada desde YOLO: {self.latched_sign}")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_fuzzy_control(self, error_px: float):
        error_scaled  = error_px * self.p_error_gain
        error_clamped = min(max(error_scaled, -320.0), 320.0)

        self.fuzzy_sim.input["E"] = error_clamped
        self.fuzzy_sim.compute()

        target_v = (
            float(self.fuzzy_sim.output["Velocidad_lineal"])
            * self.p_fuzzy_v_scale
        )

        angular_z = (
            float(self.fuzzy_sim.output["Velocidad_angular"])
            * self.p_fuzzy_w_scale
        )

        return target_v, angular_z, error_clamped

    def _filter_line_error(self, raw_error: float):
        alpha = min(max(self.p_error_filter_alpha, 0.0), 1.0)

        self.filtered_error = (
            alpha * raw_error
            + (1.0 - alpha) * self.filtered_error
        )

        return self.filtered_error

    def _limit_angular_rate(self, target_w: float, dt_sec: float):
        target_w = min(max(target_w, -self.p_max_angular_z), self.p_max_angular_z)

        max_step = self.p_angular_slew_rate * dt_sec
        delta = target_w - self.current_w

        if abs(delta) <= max_step:
            self.current_w = target_w
        else:
            self.current_w += max_step if delta > 0.0 else -max_step

        return self.current_w

    def _curve_speed_factor(self, error_px: float):
        denom = max(self.p_curve_slowdown_error, 1.0)
        ratio = min(abs(error_px) / denom, 1.0)

        factor = 1.0 - (1.0 - self.p_curve_min_v_factor) * ratio
        return min(max(factor, self.p_curve_min_v_factor), 1.0)

    def _publish(self, v: float, w: float):
        self.last_v = max(0.0, float(v))
        self.last_w = float(w)

        cmd = Twist()
        cmd.linear.x  = self.last_v
        cmd.angular.z = self.last_w

        self.pub_cmd.publish(cmd)

    def _hard_stop(self):
        self.current_v = 0.0
        self.current_w = 0.0
        self._publish(0.0, 0.0)

    def _elapsed(self, t0, now):
        return (now - t0).nanoseconds / 1e9

    def _enter_stop_post(self, now):
        self._hard_stop()
        self._stop_t0 = now
        self.state    = State.STOP_POST
        self.get_logger().info(
            f"[SM] → STOP_POST "
            f"({self.p_debug_stop_duration:.2f} s pause after maneuver)"
        )

    def _set_crossing(self, active: bool):
        if active == self._crossing_active:
            return

        self._crossing_active = active

        msg = Bool()
        msg.data = active
        self.pub_crossing.publish(msg)

    def _traffic_should_stop(self):
        if not self.p_traffic_light_enabled:
            return False

        if self._tl_latched_stop:
            # Only release on explicit green or yellow; none/lost keeps us stopped.
            if self.detected_color in (2, 3):
                self._tl_latched_stop = False
                self.get_logger().info(
                    f"[Semáforo] Latch liberado por {LABELS.get(self.detected_color, '?')}"
                )
                return False
            return True

        if self.detected_color == 1:
            self._tl_latched_stop = True
            return True

        return False

    def _traffic_speed_factor(self):
        if not self.p_traffic_light_enabled:
            return 1.0

        # Amarillo: baja velocidad sin detener.
        if self.detected_color == 2:
            return min(max(self.p_yellow_slowdown_factor, 0.0), 1.0)

        return 1.0

    def _ensure_latch(self):
        """
        Mantiene siempre un latch válido.

        - Si perma_latch está activo, fuerza ese valor.
        - Si no hay perma_latch y no se ha visto flecha, usa default_latch.
        """
        if self.p_perma_latch is not None:
            self.latched_sign = self.p_perma_latch
        elif self.latched_sign not in LATCHABLE_SIGNS:
            self.latched_sign = self.p_default_latch

        if self.state == State.FOLLOW_LINE and self.latched_sign in LATCHABLE_SIGNS:
            self.state = State.SIGN_LATCHED

    def _reset_to_default_latch(self):
        """
        Después de una maniobra, vuelve a default_latch.
        Si YOLO ve otra flecha después, cb_signs la sobrescribe.
        """
        if self.p_perma_latch is not None:
            self.latched_sign = self.p_perma_latch
        else:
            self.latched_sign = self.p_default_latch

        self.state = State.SIGN_LATCHED

    def _start_stop_sign_routine(self):
        self.state = State.STOP_SIGN
        self._stop_sign_t0 = self.get_clock().now()
        self.stop_sign_consumed = True
        self._set_crossing(False)
        self._hard_stop()
        self.get_logger().info(
            f"[SM] STOP_SIGN detectado → alto {self.p_stop_sign_duration:.1f} s"
        )

    def _start_maneuver(self):
        self._ensure_latch()
        sign = self.latched_sign

        if sign == "flecha_derecha":
            self._maneuver_steps = self.p_cross_right
        elif sign == "flecha_izquierda":
            self._maneuver_steps = self.p_cross_left
        else:
            self._maneuver_steps = self.p_cross_straight

        self._maneuver_idx = 0
        self._stop_t0      = self.get_clock().now()
        self.state         = State.STOP_PRE

        self._set_crossing(True)
        self._hard_stop()

        self.get_logger().info(
            f"[SM] STOP_PRE → '{sign}' "
            f"({self.p_debug_stop_duration:.2f} s pause before maneuver)"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Control loop
    # ─────────────────────────────────────────────────────────────────────────
    def control_loop(self):
        now    = self.get_clock().now()
        dt_sec = self._elapsed(self.prev_control_t, now)
        self.prev_control_t = now

        if dt_sec <= 0.0:
            dt_sec = 0.1

        # Mantener latch válido: YOLO si existe, default recto si no.
        if self.state in (State.FOLLOW_LINE, State.SIGN_LATCHED):
            self._ensure_latch()

        # ── STOP_SIGN ────────────────────────────────────────────────────────
        if self.state == State.STOP_SIGN:
            self._hard_stop()

            if self._elapsed(self._stop_sign_t0, now) >= self.p_stop_sign_duration:
                self._reset_to_default_latch()
                self.get_logger().info("[SM] STOP_SIGN terminado → SIGN_LATCHED/default")

            return

        # ── STOP_PRE ─────────────────────────────────────────────────────────
        if self.state == State.STOP_PRE:
            self._hard_stop()

            if self._elapsed(self._stop_t0, now) >= self.p_debug_stop_duration:
                self._maneuver_t0 = now
                self.state        = State.CROSSING

                self.get_logger().info(
                    f"[SM] → CROSSING ({len(self._maneuver_steps)} pasos)"
                )

            return

        # ── CROSSING ─────────────────────────────────────────────────────────
        if self.state == State.CROSSING:
            if self._maneuver_idx >= len(self._maneuver_steps):
                self._enter_stop_post(now)
                return

            v, w, duration = self._maneuver_steps[self._maneuver_idx]

            if self._elapsed(self._maneuver_t0, now) >= duration:
                self._maneuver_idx += 1
                self._maneuver_t0  = now

                self.get_logger().info(
                    f"[SM] Paso {self._maneuver_idx}/"
                    f"{len(self._maneuver_steps)} completado"
                )

                if self._maneuver_idx >= len(self._maneuver_steps):
                    self._enter_stop_post(now)
                    return

                v, w, _ = self._maneuver_steps[self._maneuver_idx]

            self._publish(v, w)
            return

        # ── STOP_POST ────────────────────────────────────────────────────────
        if self.state == State.STOP_POST:
            self._hard_stop()

            if self._elapsed(self._stop_t0, now) >= self.p_debug_stop_duration:
                self._set_crossing(False)   # re-enable line follower so it can publish error data
                self.state           = State.RECOVERING
                self._recover_stable = 0
                self._recover_t0     = now

                self.get_logger().info("[SM] → RECOVERING")

            return

        # ── RECOVERING ───────────────────────────────────────────────────────
        if self.state == State.RECOVERING:
            age = self._elapsed(self.last_error_t, now)

            line_ok = (
                age < self.p_line_lost_timeout
                and self.line_error is not None
            )

            self._recover_stable = (
                self._recover_stable + 1
                if line_ok
                else 0
            )

            if self._elapsed(self._recover_t0, now) > self.p_recover_timeout:
                self.get_logger().warn(
                    "[SM] RECOVERING timeout → default latch forzado"
                )

                self._set_crossing(False)
                self.zebra_active = False
                self._reset_to_default_latch()
                return

            if self._recover_stable >= self.p_recover_stable_frames:
                self.get_logger().info(
                    f"[SM] Línea re-adquirida "
                    f"({self._recover_stable} frames) → default latch"
                )

                self._set_crossing(False)
                self.zebra_active = False
                self._reset_to_default_latch()
                return

            self._hard_stop()
            return

        # ── FOLLOW_LINE / SIGN_LATCHED ───────────────────────────────────────

        # STOP sign tiene prioridad sobre semáforo y seguimiento normal.
        if (
            self.p_stop_sign_enabled
            and self.state in (State.FOLLOW_LINE, State.SIGN_LATCHED)
            and self.stop_sign_active
            and not self.stop_sign_consumed
        ):
            self._start_stop_sign_routine()
            return

        # Semáforo: solo rojo detiene. Amarillo reduce velocidad más abajo.
        if self.state in (State.FOLLOW_LINE, State.SIGN_LATCHED) and self._traffic_should_stop():
            self._hard_stop()
            self.get_logger().info(
                f"[TRAFFIC_STOP] Detenido por semáforo: {LABELS.get(self.detected_color, '?')}",
                throttle_duration_sec=0.5,
            )
            return

        # Zebra + señal latcheada → maniobra.
        if (
            self.state == State.SIGN_LATCHED
            and self.zebra_active
            and not self.zebra_consumed
        ):
            self.zebra_consumed = True
            self._start_maneuver()
            return

        age       = self._elapsed(self.last_error_t, now)
        line_lost = age > self.p_line_lost_timeout

        # En SIGN_LATCHED con línea perdida: congelar último comando.
        if self.state == State.SIGN_LATCHED and line_lost:
            self._publish(self.last_v, self.last_w)

            self.get_logger().info(
                f"[SIGN_LATCHED] Línea perdida → congelando "
                f"v={self.last_v:.3f} w={self.last_w:.3f}",
                throttle_duration_sec=0.5,
            )

            return

        active_error_raw = (
            self.line_error
            if (not line_lost and self.line_error is not None)
            else self.prev_error
        )

        self.prev_error = active_error_raw

        active_error = self._filter_line_error(active_error_raw)

        target_v, angular_z, error_clamped = self._compute_fuzzy_control(active_error)

        if line_lost:
            target_v *= self.p_line_lost_speed
        else:
            target_v *= self._curve_speed_factor(active_error)

        # Semáforo amarillo: reducir velocidad sin detener.
        target_v *= self._traffic_speed_factor()

        # Rampa asimétrica de velocidad lineal.
        error_v = target_v - self.current_v

        if error_v >= 0.0:
            if self.p_ss_v_threshold > 1e-6:
                ratio = min(self.current_v / self.p_ss_v_threshold, 1.0)
            else:
                ratio = 1.0

            ss_factor = self.p_ss_min_factor + (1.0 - self.p_ss_min_factor) * ratio
            max_step  = self.p_accel_rate_up * ss_factor * dt_sec
        else:
            max_step = self.p_accel_rate_down * dt_sec

        if abs(error_v) <= max_step:
            self.current_v = target_v
        else:
            self.current_v += max_step if error_v > 0 else -max_step

        # Suavizado/límite del giro final.
        if self.current_v <= 1e-3:
            self.current_w = 0.0
            angular_z = 0.0
        else:
            angular_z = self._limit_angular_rate(angular_z, dt_sec)

        self._publish(self.current_v, angular_z)

        self.get_logger().info(
            f"[{self.state.value}]  "
            f"v={self.current_v:.3f}  "
            f"err_raw={active_error_raw:.1f}px  "
            f"err_filt={active_error:.1f}px  "
            f"err_gain={error_clamped:.1f}px  "
            f"w={angular_z:.3f}  "
            f"lost={line_lost}  "
            f"latch={self.latched_sign}  "
            f"default={self.p_default_latch}  "
            f"zebra={self.zebra_active}  "
            f"zebra_consumed={self.zebra_consumed}  "
            f"stop={self.stop_sign_active}/{self.stop_sign_consumed}  "
            f"tl={LABELS.get(self.detected_color, '?')}",
            throttle_duration_sec=0.5,
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