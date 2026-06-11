#!/usr/bin/env python3
"""
LineFollowerController — ROS 2 Humble
Control difuso (Mamdani) + State Machine de cruce de zebra.

Estados:
  FOLLOW_LINE   — seguimiento normal con fuzzy (sin señal latcheada)
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
  - Si no hay flecha detectada, default = "none" (sin latch).
  - La maniobra de zebra SOLO se ejecuta si hay un latch real
    (flecha_derecha / flecha_izquierda / flecha_derecho).
  - Sin latch el robot sigue la línea normalmente aunque detecte zebra.

CAMBIOS vs versión anterior:
  - default_latch: "flecha_derecho" → "none"
  - Estado inicial: SIGN_LATCHED → FOLLOW_LINE
  - _ensure_latch(): ya NO fuerza un latch cuando no hay señal detectada.
    Si perma_latch está activo sí se mantiene el comportamiento anterior.
  - _reset_to_default_latch(): tras una maniobra vuelve a latched="none" + FOLLOW_LINE.
  - Trigger de zebra: guarda extra `self.latched_sign in LATCHABLE_SIGNS`
    para que una zebra sin latch previo sea ignorada silenciosamente.
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
PAUSE_SIGNS     = {"stop", "give_way"}
ROADWORK_SIGNS  = {"roadwork_ahead"}


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
        # ║  GRUPO 1 — ANTI-OSCILACIÓN                                      ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("error_filter_alpha", 0.45)
        self.declare_parameter("angular_slew_rate",  1.40)
        self.declare_parameter("max_angular_z",      0.55)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 2 — VELOCIDAD EN CURVAS                                  ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("curve_slowdown_error", 90.0)
        self.declare_parameter("curve_min_v_factor",   0.55)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 3 — ESCALA FUZZY                                         ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("fuzzy_v_scale", 0.10)
        self.declare_parameter("fuzzy_w_scale", 0.20)
        self.declare_parameter("error_gain",    0.75)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 4 — RAMPA DE VELOCIDAD LINEAL                            ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("accel_rate_up",          0.05)
        self.declare_parameter("accel_rate_down",        0.30)
        self.declare_parameter("soft_start_v_threshold", 0.08)
        self.declare_parameter("soft_start_min_factor",  0.25)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 5 — COMPORTAMIENTO SI SE PIERDE LA LÍNEA                 ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("line_lost_timeout", 0.5)
        self.declare_parameter("line_lost_speed",   0.10)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 6 — SEMÁFORO                                             ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("traffic_light_enabled",  True)
        self.declare_parameter("yellow_slowdown_factor", 0.50)
        self.declare_parameter("tl_buffer_size",         10)
        self.declare_parameter("tl_red_ratio",           0.50)
        self.declare_parameter("tl_yellow_ratio",        0.50)
        self.declare_parameter("tl_green_ratio",         0.50)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 7 — SEÑALES STOP / GIVE WAY                             ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("stop_sign_enabled",     True)
        self.declare_parameter("stop_sign_duration",    3.0)
        self.declare_parameter("stop_sign_cooldown",    7.0)
        self.declare_parameter("give_way_sign_enabled", True)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 11 — SEÑAL ROADWORK AHEAD                                ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("roadwork_sign_enabled", True)
        self.declare_parameter("roadwork_duration",     3.0)
        self.declare_parameter("roadwork_cooldown",     7.0)
        self.declare_parameter("roadwork_speed_factor", 0.5)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 8 — MANIOBRAS OPEN-LOOP EN CRUCE                        ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("cross_right_fwd1_v",  0.12)
        self.declare_parameter("cross_right_fwd1_t",  3.5)
        self.declare_parameter("cross_right_turn_w", -0.6)
        self.declare_parameter("cross_right_turn_t",  2.6)
        self.declare_parameter("cross_right_fwd2_v",  0.12)
        self.declare_parameter("cross_right_fwd2_t",  2.0)

        self.declare_parameter("cross_left_fwd1_v",  0.12)
        self.declare_parameter("cross_left_fwd1_t",  3.5)
        self.declare_parameter("cross_left_turn_w",   0.6)
        self.declare_parameter("cross_left_turn_t",   2.6)
        self.declare_parameter("cross_left_fwd2_v",  0.12)
        self.declare_parameter("cross_left_fwd2_t",  2.0)

        self.declare_parameter("cross_straight_v", 0.12)
        self.declare_parameter("cross_straight_t", 5.0)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 9 — RE-ADQUISICIÓN DE LÍNEA TRAS CRUCE                  ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("recover_stable_frames", 5)
        self.declare_parameter("recover_timeout",       5.0)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  GRUPO 10 — LATCH DE SEÑAL                                      ║
        # ║                                                                  ║
        # ║  CAMBIO: default_latch es ahora "none".                         ║
        # ║    "none"            = sin latch, sigue línea sin ejecutar       ║
        # ║                        maniobra aunque haya zebra.              ║
        # ║    "flecha_derecha"  = gira derecha al cruzar                   ║
        # ║    "flecha_izquierda"= gira izquierda al cruzar                 ║
        # ║    "flecha_derecho"  = sigue recto al cruzar                    ║
        # ║                                                                  ║
        # ║  perma_latch: "none" = usa YOLO | nombre = forzar (debug)       ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self.declare_parameter("perma_latch",   "none")
        self.declare_parameter("default_latch", "none")   # ← CAMBIO: era "flecha_derecho"

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

        self.last_v = 0.0
        self.last_w = 0.0

        # ── Estado semáforo ──────────────────────────────────────────────────
        self.detected_color   = 0
        self.raw_buffer       = deque(maxlen=int(self.p_tl_buffer_size))
        self._tl_latched_stop = False

        # ── Estado de señal STOP / GIVE_WAY ─────────────────────────────────
        self.stop_sign_active      = False
        self.stop_sign_consumed    = False
        self._stop_sign_t0         = None
        self._stop_sign_cooldown_t = None

        # ── Estado de señal ROADWORK AHEAD ───────────────────────────────────
        self._roadwork_trigger_t = None

        # ── State machine ────────────────────────────────────────────────────
        # CAMBIO: estado inicial FOLLOW_LINE (sin latch), no SIGN_LATCHED
        self.state        = State.FOLLOW_LINE
        self.latched_sign = "none"          # ← CAMBIO: era self.p_default_latch

        self.zebra_active   = False
        self.zebra_consumed = False

        self._maneuver_steps = []
        self._maneuver_idx   = 0
        self._maneuver_t0    = None

        self._stop_t0 = None

        self._recover_stable = 0
        self._recover_t0     = None

        self._tl_just_released = False

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

        self.p_stop_sign_enabled     = bool(self.get_parameter("stop_sign_enabled").value)
        self.p_stop_sign_duration    = float(self.get_parameter("stop_sign_duration").value)
        self.p_stop_sign_cooldown    = float(self.get_parameter("stop_sign_cooldown").value)
        self.p_give_way_sign_enabled = bool(self.get_parameter("give_way_sign_enabled").value)
        self.p_roadwork_sign_enabled = bool(self.get_parameter("roadwork_sign_enabled").value)
        self.p_roadwork_duration     = float(self.get_parameter("roadwork_duration").value)
        self.p_roadwork_cooldown     = float(self.get_parameter("roadwork_cooldown").value)
        self.p_roadwork_speed_factor = float(self.get_parameter("roadwork_speed_factor").value)

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
        # CAMBIO: "none" es ahora un valor válido de default_latch
        self.p_default_latch = (
            raw_default if raw_default in LATCHABLE_SIGNS else "none"
        )

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
        val = int(msg.data)
        if val not in (0, 1, 2, 3):
            val = 0

        self.raw_buffer.append(val)

        counts = Counter(self.raw_buffer)
        n = max(1, len(self.raw_buffer))

        red_ratio    = counts[1] / n
        yellow_ratio = counts[2] / n
        green_ratio  = counts[3] / n

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
          - stop / give_way: activa rutina de pausa + cooldown anti-loop 7 s
          - roadwork_ahead: reduce velocidad al 50 % por 3 s + cooldown 7 s
          - flechas: latch normal; flecha_derecho no puede override un giro ya latcheado

        Aunque perma_latch esté activo, sí procesamos señales de pausa/roadwork.
        """
        try:
            detections = json.loads(msg.data)
        except Exception:
            return

        # ── STOP / GIVE_WAY ──────────────────────────────────────────────────
        stop_candidates = [
            d for d in detections
            if str(d.get("class_name", "")).lower() == "stop"
            and self.p_stop_sign_enabled
        ]
        give_way_candidates = [
            d for d in detections
            if str(d.get("class_name", "")).lower() == "give_way"
            and self.p_give_way_sign_enabled
        ]

        if stop_candidates or give_way_candidates:
            self.stop_sign_active = True
        else:
            self.stop_sign_active = False
            self.stop_sign_consumed = False

        # ── ROADWORK AHEAD ───────────────────────────────────────────────────
        if self.p_roadwork_sign_enabled:
            roadwork_candidates = [
                d for d in detections
                if str(d.get("class_name", "")).lower() in ROADWORK_SIGNS
            ]
            if roadwork_candidates:
                now_t = self.get_clock().now()
                cooldown_ok = (
                    self._roadwork_trigger_t is None
                    or self._elapsed(self._roadwork_trigger_t, now_t)
                       >= (self.p_roadwork_duration + self.p_roadwork_cooldown)
                )
                if cooldown_ok:
                    self._roadwork_trigger_t = now_t
                    self.get_logger().info(
                        f"[SM] Roadwork ahead → velocidad al "
                        f"{self.p_roadwork_speed_factor*100:.0f}% "
                        f"por {self.p_roadwork_duration:.1f}s"
                    )

        # ── Si perma_latch está activo, no usamos flechas de YOLO ────────────
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

        # Preferir flechas de giro sobre flecha_derecho cuando ambas presentes
        turn_candidates = [
            d for d in candidates
            if d["class_name"] in {"flecha_derecha", "flecha_izquierda"}
        ]
        if turn_candidates:
            candidates = turn_candidates

        best = max(candidates, key=lambda d: d.get("bbox_area", 0))
        new_sign = best["class_name"]

        # flecha_derecho no puede overridear un giro ya confirmado
        if (
            new_sign == "flecha_derecho"
            and self.latched_sign in {"flecha_derecha", "flecha_izquierda"}
        ):
            return

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
        delta    = target_w - self.current_w

        if abs(delta) <= max_step:
            self.current_w = target_w
        else:
            self.current_w += max_step if delta > 0.0 else -max_step

        return self.current_w

    def _curve_speed_factor(self, error_px: float):
        denom  = max(self.p_curve_slowdown_error, 1.0)
        ratio  = min(abs(error_px) / denom, 1.0)
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
            if self.detected_color in (2, 3):
                self._tl_latched_stop  = False
                self._tl_just_released = True
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
        if self.detected_color == 2:
            return min(max(self.p_yellow_slowdown_factor, 0.0), 1.0)
        return 1.0

    def _ensure_latch(self):
        """
        CAMBIO vs versión anterior:
          - Si perma_latch está activo, fuerza ese valor (comportamiento idéntico).
          - Si NO hay perma_latch, ya NO se fuerza ningún default.
            El robot simplemente permanece en FOLLOW_LINE sin latch.
          - Solo transiciona a SIGN_LATCHED si perma_latch está activo.
        """
        if self.p_perma_latch is not None:
            self.latched_sign = self.p_perma_latch
            if self.state == State.FOLLOW_LINE:
                self.state = State.SIGN_LATCHED

        # Sin perma_latch: no hacer nada. El latch solo llega desde cb_signs (YOLO).

    def _reset_to_default_latch(self):
        """
        Después de una maniobra, vuelve al estado base.

        CAMBIO vs versión anterior:
          - Si perma_latch activo: igual que antes (mantiene el latch forzado).
          - Si no: latched_sign = "none" y estado = FOLLOW_LINE.
            El robot sigue la línea normalmente hasta que YOLO detecte otra flecha.
        """
        if self.p_perma_latch is not None:
            self.latched_sign = self.p_perma_latch
            self.state        = State.SIGN_LATCHED
        else:
            self.latched_sign = "none"
            self.state        = State.FOLLOW_LINE

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

        # _ensure_latch solo actúa si perma_latch está activo
        if self.state in (State.FOLLOW_LINE, State.SIGN_LATCHED):
            self._ensure_latch()

        # ── STOP_SIGN ────────────────────────────────────────────────────────
        if self.state == State.STOP_SIGN:
            self._hard_stop()
            if self._elapsed(self._stop_sign_t0, now) >= self.p_stop_sign_duration:
                self._stop_sign_cooldown_t = now
                self._reset_to_default_latch()
                self.get_logger().info(
                    f"[SM] STOP_SIGN terminado → {self.state.value} "
                    f"(cooldown {self.p_stop_sign_cooldown:.1f}s)"
                )
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
                self._set_crossing(False)
                self.state           = State.RECOVERING
                self._recover_stable = 0
                self._recover_t0     = now
                self.get_logger().info("[SM] → RECOVERING")
            return

        # ── RECOVERING ───────────────────────────────────────────────────────
        if self.state == State.RECOVERING:
            age     = self._elapsed(self.last_error_t, now)
            line_ok = (
                age < self.p_line_lost_timeout
                and self.line_error is not None
            )

            self._recover_stable = (
                self._recover_stable + 1 if line_ok else 0
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
                    f"({self._recover_stable} frames) → {self.state.value}"
                )
                self._set_crossing(False)
                self.zebra_active = False
                self._reset_to_default_latch()
                return

            self._hard_stop()
            return

        # ── FOLLOW_LINE / SIGN_LATCHED ───────────────────────────────────────

        # STOP / GIVE_WAY tienen prioridad.
        if (
            self.state in (State.FOLLOW_LINE, State.SIGN_LATCHED)
            and self.stop_sign_active
            and not self.stop_sign_consumed
            and (
                self._stop_sign_cooldown_t is None
                or self._elapsed(self._stop_sign_cooldown_t, now) >= self.p_stop_sign_cooldown
            )
        ):
            self._start_stop_sign_routine()
            return

        # Semáforo rojo: detener.
        if self.state in (State.FOLLOW_LINE, State.SIGN_LATCHED) and self._traffic_should_stop():
            self._hard_stop()
            self.get_logger().info(
                f"[TRAFFIC_STOP] Detenido por semáforo: {LABELS.get(self.detected_color, '?')}",
                throttle_duration_sec=0.5,
            )
            return

        # ── Trigger de zebra ──────────────────────────────────────────────────
        # CAMBIO: guarda extra `self.latched_sign in LATCHABLE_SIGNS`.
        # Si no hay latch real (latched_sign == "none"), la zebra se ignora
        # y el robot sigue la línea con normalidad.
        if (
            self.state == State.SIGN_LATCHED
            and self.latched_sign in LATCHABLE_SIGNS    # ← nueva guarda
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
        active_error    = self._filter_line_error(active_error_raw)

        target_v, angular_z, error_clamped = self._compute_fuzzy_control(active_error)

        if line_lost:
            target_v *= self.p_line_lost_speed
        else:
            target_v *= self._curve_speed_factor(active_error)

        target_v *= self._traffic_speed_factor()

        # Roadwork ahead: velocidad reducida.
        if (
            self._roadwork_trigger_t is not None
            and self._elapsed(self._roadwork_trigger_t, now) < self.p_roadwork_duration
        ):
            target_v *= self.p_roadwork_speed_factor

        # Rampa asimétrica de velocidad lineal.
        error_v = target_v - self.current_v

        if error_v >= 0.0:
            if self._tl_just_released:
                ss_factor = 1.0
                if self.current_v >= self.p_ss_v_threshold:
                    self._tl_just_released = False
            elif self.p_ss_v_threshold > 1e-6:
                ratio     = min(self.current_v / self.p_ss_v_threshold, 1.0)
                ss_factor = self.p_ss_min_factor + (1.0 - self.p_ss_min_factor) * ratio
            else:
                ss_factor = 1.0

            max_step = self.p_accel_rate_up * ss_factor * dt_sec
        else:
            max_step = self.p_accel_rate_down * dt_sec

        if abs(error_v) <= max_step:
            self.current_v = target_v
        else:
            self.current_v += max_step if error_v > 0 else -max_step

        if self.current_v <= 1e-3:
            self.current_w = 0.0
            angular_z      = 0.0
        else:
            angular_z = self._limit_angular_rate(angular_z, dt_sec)

        self._publish(self.current_v, angular_z)

        rw_elapsed = (
            self._elapsed(self._roadwork_trigger_t, now)
            if self._roadwork_trigger_t is not None else -1.0
        )
        stop_cd = (
            self._elapsed(self._stop_sign_cooldown_t, now)
            if self._stop_sign_cooldown_t is not None else -1.0
        )
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
            f"stop_cd={stop_cd:.1f}s  "
            f"rw={rw_elapsed:.1f}s  "
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