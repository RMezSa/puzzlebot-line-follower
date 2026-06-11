#!/usr/bin/env python3
"""
Fuzzy controller — control curve analysis (v1) + 3D surface analysis (v2).
No ROS required. Run with:  python3 curve_analysis.py

Part 1 (v1 — single input E):
  - Membership function plots for the error input
  - Control curves v(e) and w(e) as actually commanded to the robot
  - Candidate mathematical approximations overlaid for comparison
  - Printed key values useful for fitting the math functions

Part 2 (v2 — two inputs E, DE):
  - 3D control surfaces v(E, DE) and w(E, DE)
  - Slice curves at DE=0 (same as v1 steady-state) and DE=±100 (transient)
"""

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control

# ── Parameters — must match line_follower_controller.py ──────────────────────
FUZZY_V_SCALE = 0.10
FUZZY_W_SCALE = 0.20
ERROR_GAIN    = 0.75
MAX_ANGULAR_Z = 0.55

# ── Parameters — must match sugeno_v1.py ─────────────────────────────────────
SUGENO_V_SCALE = 0.10
SUGENO_W_SCALE = 0.25


# ── Fuzzy system — exact copy of build_fuzzy_system() ────────────────────────
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

    return control.ControlSystemSimulation(control.ControlSystem([r1, r2, r3, r4, r5]))


# ── Sweep ─────────────────────────────────────────────────────────────────────
def sweep(sim):
    errors = np.arange(-320, 321, 2, dtype=float)
    v_cmd  = np.zeros_like(errors)
    w_cmd  = np.zeros_like(errors)
    v_raw  = np.zeros_like(errors)
    w_raw  = np.zeros_like(errors)

    for i, e_px in enumerate(errors):
        e_in = float(np.clip(e_px * ERROR_GAIN, -320, 320))
        sim.input["E"] = e_in
        sim.compute()

        vf = float(sim.output["Velocidad_lineal"])
        wf = float(sim.output["Velocidad_angular"])

        v_raw[i] = vf
        w_raw[i] = wf
        v_cmd[i] = vf * FUZZY_V_SCALE
        w_cmd[i] = float(np.clip(wf * FUZZY_W_SCALE, -MAX_ANGULAR_Z, MAX_ANGULAR_Z))

    return errors, v_raw, w_raw, v_cmd, w_cmd


# ── Candidate math functions ───────────────────────────────────────────────────
# v(e): the fuzzy curve is a step, not a parabola.
# Best fit: sigmoid centered at e_thresh with steepness k.
V_MAX    = 0.130   # v at e=0
V_MIN    = 0.065   # v at large error (exactly V_MAX/2 by construction)
E_THRESH = 125.0   # px — midpoint of the transition
K_SIG    = 0.215   # steepness of the sigmoid

# w(e): flat near zero then sharp rise — tanh(a·e) alone doesn't capture
# the dead-zone. Better fit: tanh of a cubic term.
# Simple linear: good enough for small errors, fails at saturation.
K_W_LINEAR = 0.0035   # slope for linear fit
# Tanh of cubic: captures the flat-then-steep shape
K_W_TANH   = 0.500    # amplitude (matches observed max ~0.50 rad/s)
A_TANH     = 0.010    # tune: larger = steeper transition


def v_math(e):
    """Sigmoid fit: flat at V_MAX for small error, steps to V_MIN at large."""
    return V_MIN + (V_MAX - V_MIN) / (1.0 + np.exp(K_SIG * (np.abs(e) - E_THRESH)))


def w_linear(e):
    return np.clip(-K_W_LINEAR * e, -MAX_ANGULAR_Z, MAX_ANGULAR_Z)


def w_tanh(e):
    """tanh of cubic term — produces flat-near-zero then steep rise."""
    E_SCALE = 150.0   # normalization reference
    return -K_W_TANH * np.tanh(A_TANH * e**3 / E_SCALE**2)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot(errors, v_raw, w_raw, v_cmd, w_cmd):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Controlador Difuso Mamdani — Curvas de Control', fontsize=14, fontweight='bold')

    # 1 ── Membership functions (error input)
    ax1 = fig.add_subplot(2, 3, 1)
    eu = np.arange(-320, 321, 1)
    ax1.plot(eu, fuzz.zmf(eu, -250, -70),         label='mucho_neg')
    ax1.plot(eu, fuzz.trimf(eu, [-100, -80, -8]), label='poco_neg')
    ax1.plot(eu, fuzz.trimf(eu, [-15, 0, 15]),    label='sin_error')
    ax1.plot(eu, fuzz.trimf(eu, [8, 80, 100]),    label='poco_pos')
    ax1.plot(eu, fuzz.smf(eu, 70, 250),           label='mucho_pos')
    ax1.set_title('FP — Entrada de error (px)')
    ax1.set_xlabel('Error (px)')
    ax1.set_ylabel('μ')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='k', lw=0.8, ls='--')

    # 2 ── Raw fuzzy v output (before scale)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(errors, v_raw, 'b-', lw=2)
    ax2.set_title('Salida difusa bruta — V\n(antes de ×fuzzy_v_scale)')
    ax2.set_xlabel('Error (px)')
    ax2.set_ylabel('V difuso [0 – 1.5]')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color='k', lw=0.8, ls='--')

    # 3 ── Raw fuzzy w output (before scale)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(errors, w_raw, 'r-', lw=2)
    ax3.set_title('Salida difusa bruta — W\n(antes de ×fuzzy_w_scale)')
    ax3.set_xlabel('Error (px)')
    ax3.set_ylabel('W difuso [−3 – 3]')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(0, color='k', lw=0.8, ls='--')
    ax3.axhline(0, color='k', lw=0.8, ls='--')

    # 4 ── Control curve v(e) with math approximation
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.plot(errors, v_cmd,        'b-',  lw=2.5, label='Mamdani v(e)')
    ax4.plot(errors, v_math(errors), 'b--', lw=1.5,
             label=f'sigmoide\n(v_min={V_MIN}, v_max={V_MAX}, umbral={E_THRESH:.0f}px)')
    ax4.set_title('Curva de control — v(e)  [cmd_vel.linear.x]')
    ax4.set_xlabel('Error (px)')
    ax4.set_ylabel('v  (m/s)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.axvline(0, color='k', lw=0.8, ls='--')

    # 5 ── Control curve w(e) with math approximations
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.plot(errors, w_cmd,           'r-',   lw=2.5, label='Mamdani w(e)')
    ax5.plot(errors, w_linear(errors), 'g--',  lw=1.5,
             label=f'−k_w·e  (k_w={K_W_LINEAR})')
    ax5.plot(errors, w_tanh(errors),   'm--',  lw=1.5,
             label=f'−K·tanh(a·e³/e₀²)\n(K={K_W_TANH}, a={A_TANH})')
    ax5.axhline( MAX_ANGULAR_Z, color='r', lw=1, ls=':',
                 label=f'±vel_angular_máx ({MAX_ANGULAR_Z})')
    ax5.axhline(-MAX_ANGULAR_Z, color='r', lw=1, ls=':')
    ax5.set_title('Curva de control — w(e)  [cmd_vel.angular.z]')
    ax5.set_xlabel('Error (px)')
    ax5.set_ylabel('w  (rad/s)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.axvline(0, color='k', lw=0.8, ls='--')
    ax5.axhline(0, color='k', lw=0.8, ls='--')

    plt.tight_layout()
    plt.savefig('control_curves.png', dpi=150, bbox_inches='tight')
    print('Saved: control_curves.png')
    plt.show()


# ── Key values for report ─────────────────────────────────────────────────────
def print_key_values(errors, v_cmd, w_cmd, label='Mamdani v1'):
    idx0 = np.argmin(np.abs(errors))  # index closest to e=0

    print(f'\n── Key values — {label} ──────────────────────────────────────')
    print(f'v(0)    = {v_cmd[idx0]:.4f} m/s   ← v_max for math fit')
    print(f'w(0)    = {w_cmd[idx0]:.4f} rad/s  ← should be ~0')

    for e_check in [50, 100, 150, 200, 250, 320]:
        idx_p = np.argmin(np.abs(errors - e_check))
        idx_n = np.argmin(np.abs(errors + e_check))
        print(f'v(±{e_check:3d}) = {v_cmd[idx_p]:.4f} m/s  |  '
              f'w(+{e_check:3d}) = {w_cmd[idx_p]:+.4f}  '
              f'w(-{e_check:3d}) = {w_cmd[idx_n]:+.4f} rad/s')

    # Saturation threshold for w
    sat_idx = np.argmax(np.abs(w_cmd) >= MAX_ANGULAR_Z - 0.001)
    print(f'\nw saturates at |e| ≈ {abs(errors[sat_idx]):.0f} px')

    # Error where v starts dropping noticeably (>2% drop from v_max)
    v_max = v_cmd[idx0]
    drop_threshold = v_max * 0.98
    for i, e in enumerate(errors):
        if e >= 0 and v_cmd[i] < drop_threshold:
            print(f'v drops >2% from v_max at |e| ≈ {abs(e):.0f} px')
            break

    print('────────────────────────────────────────────────────\n')


# ── v2 — 2-input fuzzy system (E + DE) ───────────────────────────────────────
def build_fuzzy_system_v2():
    E  = control.Antecedent(np.arange(-320, 321, 1),    "E")
    DE = control.Antecedent(np.arange(-300, 301, 1),    "DE")
    V  = control.Consequent(np.arange(0.0,  1.51, 0.01), "Velocidad_lineal")
    W  = control.Consequent(np.arange(-3.0, 3.01, 0.01), "Velocidad_angular")

    E["mucho_error_negativo"] = fuzz.zmf(E.universe, -250, -70)
    E["poco_error_negativo"]  = fuzz.trimf(E.universe, [-100, -80, -8])
    E["sin_error"]            = fuzz.trimf(E.universe, [-15, 0, 15])
    E["poco_error_positivo"]  = fuzz.trimf(E.universe, [8, 80, 100])
    E["mucho_error_positivo"] = fuzz.smf(E.universe, 70, 250)

    DE["decreciendo"] = fuzz.zmf(DE.universe, -300, -20)
    DE["estable"]     = fuzz.trimf(DE.universe, [-40, 0, 40])
    DE["creciendo"]   = fuzz.smf(DE.universe,  20,  300)

    V["nada"]   = fuzz.trimf(V.universe, [0.0, 0.0, 0.4])
    V["poca"]   = fuzz.trimf(V.universe, [0.4, 0.65, 0.9])
    V["normal"] = fuzz.trapmf(V.universe, [0.9, 1.5, 1.5, 1.5])

    W["mucho_negativo"] = fuzz.trimf(W.universe, [-3.0, -3.0, -1.5])
    W["poco_negativo"]  = fuzz.trimf(W.universe, [-1.5, -0.8, -0.3])
    W["nada"]           = fuzz.trimf(W.universe, [-0.3,  0.0,  0.3])
    W["poco_positivo"]  = fuzz.trimf(W.universe, [ 0.3,  0.8,  1.5])
    W["mucho_positivo"] = fuzz.trimf(W.universe, [ 1.5,  3.0,  3.0])

    rules = [
        control.Rule(E["mucho_error_negativo"] & DE["decreciendo"], (V["poca"],   W["poco_positivo"])),
        control.Rule(E["mucho_error_negativo"] & DE["estable"],     (V["poca"],   W["mucho_positivo"])),
        control.Rule(E["mucho_error_negativo"] & DE["creciendo"],   (V["nada"],   W["mucho_positivo"])),
        control.Rule(E["poco_error_negativo"]  & DE["decreciendo"], (V["normal"], W["nada"])),
        control.Rule(E["poco_error_negativo"]  & DE["estable"],     (V["normal"], W["poco_positivo"])),
        control.Rule(E["poco_error_negativo"]  & DE["creciendo"],   (V["poca"],   W["mucho_positivo"])),
        control.Rule(E["sin_error"]            & DE["decreciendo"], (V["normal"], W["poco_positivo"])),
        control.Rule(E["sin_error"]            & DE["estable"],     (V["normal"], W["nada"])),
        control.Rule(E["sin_error"]            & DE["creciendo"],   (V["normal"], W["poco_negativo"])),
        control.Rule(E["poco_error_positivo"]  & DE["decreciendo"], (V["normal"], W["nada"])),
        control.Rule(E["poco_error_positivo"]  & DE["estable"],     (V["normal"], W["poco_negativo"])),
        control.Rule(E["poco_error_positivo"]  & DE["creciendo"],   (V["poca"],   W["mucho_negativo"])),
        control.Rule(E["mucho_error_positivo"] & DE["decreciendo"], (V["poca"],   W["poco_negativo"])),
        control.Rule(E["mucho_error_positivo"] & DE["estable"],     (V["poca"],   W["mucho_negativo"])),
        control.Rule(E["mucho_error_positivo"] & DE["creciendo"],   (V["nada"],   W["mucho_negativo"])),
    ]

    return control.ControlSystemSimulation(control.ControlSystem(rules))


def sweep_2d(sim_v2, e_step=10, de_step=20):
    """Grid sweep over (E, DE) space. Returns meshgrid arrays and output surfaces."""
    e_vals  = np.arange(-320, 321, e_step,  dtype=float)
    de_vals = np.arange(-300, 301, de_step, dtype=float)
    EE, DEE = np.meshgrid(e_vals, de_vals)

    V_surf = np.zeros_like(EE)
    W_surf = np.zeros_like(EE)

    for i in range(EE.shape[0]):
        for j in range(EE.shape[1]):
            e_in  = float(np.clip(EE[i, j] * ERROR_GAIN, -320, 320))
            de_in = float(np.clip(DEE[i, j], -300, 300))
            sim_v2.input["E"]  = e_in
            sim_v2.input["DE"] = de_in
            sim_v2.compute()
            V_surf[i, j] = float(sim_v2.output["Velocidad_lineal"])  * FUZZY_V_SCALE
            W_surf[i, j] = float(np.clip(
                sim_v2.output["Velocidad_angular"] * FUZZY_W_SCALE,
                -MAX_ANGULAR_Z, MAX_ANGULAR_Z
            ))

    return EE, DEE, V_surf, W_surf


def plot_surfaces(EE, DEE, V_surf, W_surf, sim_v2):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Mamdani v2 (E + ΔE) — Superficies de Control', fontsize=14, fontweight='bold')

    # ── 3D surface: v(E, DE) ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot_surface(EE, DEE, V_surf, cmap='Blues', alpha=0.85)
    ax1.set_xlabel('E  (px)')
    ax1.set_ylabel('ΔE  (px/s)')
    ax1.set_zlabel('v  (m/s)')
    ax1.set_title('Superficie v(E, ΔE)')

    # ── 3D surface: w(E, DE) ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot_surface(EE, DEE, W_surf, cmap='RdBu', alpha=0.85)
    ax2.set_xlabel('E  (px)')
    ax2.set_ylabel('ΔE  (px/s)')
    ax2.set_zlabel('w  (rad/s)')
    ax2.set_title('Superficie w(E, ΔE)')

    # ── DE MFs ────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    deu = np.arange(-300, 301, 1)
    ax3.plot(deu, fuzz.zmf(deu, -300, -20),    label='decreciendo')
    ax3.plot(deu, fuzz.trimf(deu, [-40, 0, 40]), label='estable')
    ax3.plot(deu, fuzz.smf(deu,   20,  300),    label='creciendo')
    ax3.set_title('FP — Entrada ΔE (px/s)')
    ax3.set_xlabel('ΔE (px/s)')
    ax3.set_ylabel('μ')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axvline(0, color='k', lw=0.8, ls='--')

    # ── Slice curves v(E) at three DE values ─────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    e_slice = np.arange(-320, 321, 4, dtype=float)
    colors  = ['blue', 'green', 'red']
    labels  = ['ΔE = −100 (corrigiendo)', 'ΔE =   0 (estable)', 'ΔE = +100 (divergiendo)']
    for de_val, c, lbl in zip([-100, 0, 100], colors, labels):
        vs = []
        for e in e_slice:
            e_in = float(np.clip(e * ERROR_GAIN, -320, 320))
            sim_v2.input["E"]  = e_in
            sim_v2.input["DE"] = float(de_val)
            sim_v2.compute()
            vs.append(float(sim_v2.output["Velocidad_lineal"]) * FUZZY_V_SCALE)
        ax4.plot(e_slice, vs, color=c, lw=2, label=lbl)
    ax4.set_title('Cortes de v(E) a ΔE fijo')
    ax4.set_xlabel('E (px)')
    ax4.set_ylabel('v (m/s)')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)
    ax4.axvline(0, color='k', lw=0.8, ls='--')

    # ── Slice curves w(E) at three DE values ─────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    for de_val, c, lbl in zip([-100, 0, 100], colors, labels):
        ws = []
        for e in e_slice:
            e_in = float(np.clip(e * ERROR_GAIN, -320, 320))
            sim_v2.input["E"]  = e_in
            sim_v2.input["DE"] = float(de_val)
            sim_v2.compute()
            ws.append(float(np.clip(
                sim_v2.output["Velocidad_angular"] * FUZZY_W_SCALE,
                -MAX_ANGULAR_Z, MAX_ANGULAR_Z
            )))
        ax5.plot(e_slice, ws, color=c, lw=2, label=lbl)
    ax5.set_title('Cortes de w(E) a ΔE fijo')
    ax5.set_xlabel('E (px)')
    ax5.set_ylabel('w (rad/s)')
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)
    ax5.axvline(0, color='k', lw=0.8, ls='--')
    ax5.axhline(0, color='k', lw=0.8, ls='--')

    # ── Top-down heatmap: w(E, DE) ────────────────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    im = ax6.contourf(EE, DEE, W_surf, levels=20, cmap='RdBu')
    fig.colorbar(im, ax=ax6, label='w (rad/s)')
    ax6.set_title('Mapa de calor w(E, ΔE)')
    ax6.set_xlabel('E (px)')
    ax6.set_ylabel('ΔE (px/s)')
    ax6.axvline(0, color='k', lw=0.8, ls='--')
    ax6.axhline(0, color='k', lw=0.8, ls='--')

    plt.tight_layout()
    plt.savefig('control_surfaces_v2.png', dpi=150, bbox_inches='tight')
    print('Saved: control_surfaces_v2.png')
    plt.show()


# ── Sugeno zero-order controller (same MFs as Mamdani v1) ────────────────────
class SugenoController:
    """Zero-order Sugeno: same 5 input MFs, constant consequents, weighted-average defuzz."""

    _eu = np.arange(-320, 321, 1)
    _mf_list = None  # computed once on first call

    # Constant consequents — Sugeno equivalents of Mamdani output centroids
    #   mucho_neg → V=0.65 (poca),   W=+2.30 (mucho_positivo)
    #   poco_neg  → V=1.20 (normal), W=+0.80 (poco_positivo)
    #   sin_error → V=1.50 (normal), W= 0.00 (nada)
    #   poco_pos  → V=1.20 (normal), W=-0.80 (poco_negativo)
    #   mucho_pos → V=0.65 (poca),   W=-2.30 (mucho_negativo)
    V_SINGLETONS = np.array([0.65, 1.20, 1.50, 1.20, 0.65])
    W_SINGLETONS = np.array([2.30, 0.80, 0.00, -0.80, -2.30])
    RULE_LABELS  = ['mucho_neg', 'poco_neg', 'sin_error', 'poco_pos', 'mucho_pos']

    @classmethod
    def _mfs(cls):
        if cls._mf_list is None:
            eu = cls._eu
            cls._mf_list = [
                fuzz.zmf(eu, -250, -70),
                fuzz.trimf(eu, [-100, -80, -8]),
                fuzz.trimf(eu, [-15, 0, 15]),
                fuzz.trimf(eu, [8, 80, 100]),
                fuzz.smf(eu, 70, 250),
            ]
        return cls._mf_list

    @classmethod
    def compute(cls, e_px: float):
        e = float(np.clip(e_px, -320.0, 320.0))
        weights = np.array([
            fuzz.interp_membership(cls._eu, mf, e) for mf in cls._mfs()
        ])
        w_sum = weights.sum()
        if w_sum <= 1e-9:
            return 0.0, 0.0
        return (float(np.dot(weights, cls.V_SINGLETONS) / w_sum),
                float(np.dot(weights, cls.W_SINGLETONS) / w_sum))


def sweep_sugeno():
    errors = np.arange(-320, 321, 2, dtype=float)
    v_raw  = np.zeros_like(errors)
    w_raw  = np.zeros_like(errors)
    v_cmd  = np.zeros_like(errors)
    w_cmd  = np.zeros_like(errors)
    for i, e_px in enumerate(errors):
        e_in = float(np.clip(e_px * ERROR_GAIN, -320, 320))
        vf, wf = SugenoController.compute(e_in)
        v_raw[i] = vf
        w_raw[i] = wf
        v_cmd[i] = vf * SUGENO_V_SCALE
        w_cmd[i] = wf * SUGENO_W_SCALE   # no MAX_ANGULAR_Z clamp — Sugeno node doesn't apply it
    return errors, v_raw, w_raw, v_cmd, w_cmd


def plot_sugeno(errors, v_raw, w_raw, v_cmd, w_cmd):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Controlador Sugeno de Orden Cero — Curvas de Control', fontsize=14, fontweight='bold')

    S_V_MAX = float(v_cmd.max())
    S_W_AMP = float(np.abs(w_cmd).max())
    x_bars  = np.arange(len(SugenoController.RULE_LABELS))

    # 1 ── Error MFs (identical to Mamdani v1)
    ax1 = fig.add_subplot(2, 3, 1)
    eu = np.arange(-320, 321, 1)
    for mf_arr, lbl in zip(SugenoController._mfs(), SugenoController.RULE_LABELS):
        ax1.plot(eu, mf_arr, label=lbl)
    ax1.set_title('FP — Entrada de error (px)')
    ax1.set_xlabel('Error (px)')
    ax1.set_ylabel('μ')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='k', lw=0.8, ls='--')

    # 2 ── V singletons bar chart
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar(x_bars, SugenoController.V_SINGLETONS, color='steelblue', alpha=0.85)
    ax2.set_xticks(x_bars)
    ax2.set_xticklabels([l.replace('_', '\n') for l in SugenoController.RULE_LABELS], fontsize=8)
    ax2.set_title('Singletons V (brutos)\npor consecuente de regla')
    ax2.set_ylabel('Singleton V [0 – 1.5]')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3 ── W singletons bar chart
    ax3 = fig.add_subplot(2, 3, 3)
    bar_colors = ['tab:green' if w > 0 else ('tab:red' if w < 0 else 'tab:gray')
                  for w in SugenoController.W_SINGLETONS]
    ax3.bar(x_bars, SugenoController.W_SINGLETONS, color=bar_colors, alpha=0.85)
    ax3.set_xticks(x_bars)
    ax3.set_xticklabels([l.replace('_', '\n') for l in SugenoController.RULE_LABELS], fontsize=8)
    ax3.set_title('Singletons W (brutos)\npor consecuente de regla')
    ax3.set_ylabel('Singleton W [−2.3 – 2.3]')
    ax3.axhline(0, color='k', lw=0.8)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4 ── Raw output curves (twin axis: V left, W right)
    ax4  = fig.add_subplot(2, 3, 4)
    ax4b = ax4.twinx()
    ax4.plot(errors, v_raw,  'b-',  lw=2, label='V bruto')
    ax4b.plot(errors, w_raw, 'r--', lw=2, label='W bruto')
    ax4.set_title('Salidas Sugeno brutas (antes de ×escala)')
    ax4.set_xlabel('Error (px)')
    ax4.set_ylabel('V bruto [0 – 1.5]',     color='b')
    ax4b.set_ylabel('W bruto [−2.3 – 2.3]', color='r')
    ax4.axvline(0, color='k', lw=0.8, ls='--')
    ax4.grid(True, alpha=0.3)

    # 5 ── v(e) with math fit
    v_fit_s = V_MIN + (S_V_MAX - V_MIN) / (1.0 + np.exp(K_SIG * (np.abs(errors) - E_THRESH)))
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(errors, v_cmd,   'b-',  lw=2.5, label='Sugeno v(e)')
    ax5.plot(errors, v_fit_s, 'b--', lw=1.5,
             label=f'ajuste sigmoide\n(v_max={S_V_MAX:.3f}, v_min={V_MIN:.3f})')
    ax5.set_title('Curva de control — v(e)  [cmd_vel.linear.x]')
    ax5.set_xlabel('Error (px)')
    ax5.set_ylabel('v  (m/s)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.axvline(0, color='k', lw=0.8, ls='--')

    # 6 ── w(e) with math fit
    E_SCALE  = 150.0
    w_fit_s  = -S_W_AMP * np.tanh(A_TANH * errors**3 / E_SCALE**2)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(errors, w_cmd,  'r-',  lw=2.5, label='Sugeno w(e)')
    ax6.plot(errors, w_fit_s,'m--', lw=1.5,
             label=f'−K·tanh(a·e³/e₀²)\n(K={S_W_AMP:.3f}, a={A_TANH})')
    ax6.set_title('Curva de control — w(e)  [cmd_vel.angular.z]')
    ax6.set_xlabel('Error (px)')
    ax6.set_ylabel('w  (rad/s)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.axvline(0, color='k', lw=0.8, ls='--')
    ax6.axhline(0, color='k', lw=0.8, ls='--')

    plt.tight_layout()
    plt.savefig('sugeno_curves.png', dpi=150, bbox_inches='tight')
    print('Saved: sugeno_curves.png')
    plt.show()


def plot_comparison(errors, v_cmd_m, w_cmd_m, v_cmd_s, w_cmd_s):
    """Side-by-side comparison of Mamdani v1 and Sugeno zero-order control curves."""
    fig, (ax_v, ax_w) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Mamdani v1 vs Sugeno Orden Cero — Comparación de Curvas de Control',
                 fontsize=13, fontweight='bold')

    ax_v.plot(errors, v_cmd_m, 'b-',  lw=2.5, label='Mamdani v1')
    ax_v.plot(errors, v_cmd_s, 'b--', lw=2.5, label='Sugeno orden cero')
    ax_v.set_title('Velocidad lineal v(e)')
    ax_v.set_xlabel('Error (px)')
    ax_v.set_ylabel('v  (m/s)')
    ax_v.legend(fontsize=10)
    ax_v.grid(True, alpha=0.3)
    ax_v.axvline(0, color='k', lw=0.8, ls='--')

    ax_w.plot(errors, w_cmd_m, 'r-',  lw=2.5, label='Mamdani v1')
    ax_w.plot(errors, w_cmd_s, 'r--', lw=2.5, label='Sugeno orden cero')
    ax_w.axhline( MAX_ANGULAR_Z, color='gray', lw=1, ls=':',
                  label=f'Límite Mamdani ±{MAX_ANGULAR_Z}')
    ax_w.axhline(-MAX_ANGULAR_Z, color='gray', lw=1, ls=':')
    ax_w.set_title('Velocidad angular w(e)')
    ax_w.set_xlabel('Error (px)')
    ax_w.set_ylabel('w  (rad/s)')
    ax_w.legend(fontsize=10)
    ax_w.grid(True, alpha=0.3)
    ax_w.axvline(0, color='k', lw=0.8, ls='--')
    ax_w.axhline(0, color='k', lw=0.8, ls='--')

    plt.tight_layout()
    plt.savefig('comparison_curves.png', dpi=150, bbox_inches='tight')
    print('Saved: comparison_curves.png')
    plt.show()


if __name__ == '__main__':
    # ── Part 1a: Mamdani v1 single-input curves ──────────────────────────────
    print('=== Part 1a: Mamdani v1 single-input FIS ===')
    print('Building fuzzy system v1...')
    sim_v1 = build_fuzzy_system()

    print('Sweeping error range [-320, 320]...')
    errors, v_raw_m, w_raw_m, v_cmd_m, w_cmd_m = sweep(sim_v1)

    print_key_values(errors, v_cmd_m, w_cmd_m, label='Mamdani v1')
    plot(errors, v_raw_m, w_raw_m, v_cmd_m, w_cmd_m)

    # ── Part 1b: Sugeno zero-order curves ────────────────────────────────────
    print('\n=== Part 1b: Sugeno zero-order FIS ===')
    print('Sweeping Sugeno...')
    errors_s, v_raw_s, w_raw_s, v_cmd_s, w_cmd_s = sweep_sugeno()

    print_key_values(errors_s, v_cmd_s, w_cmd_s, label='Sugeno zero-order')
    plot_sugeno(errors_s, v_raw_s, w_raw_s, v_cmd_s, w_cmd_s)

    # ── Comparison: Mamdani v1 vs Sugeno ─────────────────────────────────────
    print('\n=== Comparison: Mamdani v1 vs Sugeno ===')
    plot_comparison(errors, v_cmd_m, w_cmd_m, v_cmd_s, w_cmd_s)

    # ── Part 2: v2 two-input surfaces ───────────────────────────────────────
    print('\n=== Part 2: v2 two-input FIS (E + ΔE) ===')
    print('Building fuzzy system v2...')
    sim_v2 = build_fuzzy_system_v2()

    print('Sweeping (E, DE) grid — this takes ~30 s...')
    EE, DEE, V_surf, W_surf = sweep_2d(sim_v2)

    plot_surfaces(EE, DEE, V_surf, W_surf, sim_v2)
