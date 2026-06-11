#!/usr/bin/env python3
"""
Fuzzy controller — control curve analysis.
No ROS required. Run with:  python3 curve_analysis.py

Produces:
  - Membership function plots for the error input
  - Control curves v(e) and w(e) as actually commanded to the robot
  - Candidate mathematical approximations overlaid for comparison
  - Printed key values useful for fitting the math functions
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
    fig.suptitle('Mamdani Fuzzy Controller — Control Curves', fontsize=14, fontweight='bold')

    # 1 ── Membership functions (error input)
    ax1 = fig.add_subplot(2, 3, 1)
    eu = np.arange(-320, 321, 1)
    ax1.plot(eu, fuzz.zmf(eu, -250, -70),         label='mucho_neg')
    ax1.plot(eu, fuzz.trimf(eu, [-100, -80, -8]), label='poco_neg')
    ax1.plot(eu, fuzz.trimf(eu, [-15, 0, 15]),    label='sin_error')
    ax1.plot(eu, fuzz.trimf(eu, [8, 80, 100]),    label='poco_pos')
    ax1.plot(eu, fuzz.smf(eu, 70, 250),           label='mucho_pos')
    ax1.set_title('MF — Error input (px)')
    ax1.set_xlabel('Error (px)')
    ax1.set_ylabel('μ')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='k', lw=0.8, ls='--')

    # 2 ── Raw fuzzy v output (before scale)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(errors, v_raw, 'b-', lw=2)
    ax2.set_title('Raw fuzzy output — V\n(before ×fuzzy_v_scale)')
    ax2.set_xlabel('Error (px)')
    ax2.set_ylabel('V fuzzy [0 – 1.5]')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color='k', lw=0.8, ls='--')

    # 3 ── Raw fuzzy w output (before scale)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(errors, w_raw, 'r-', lw=2)
    ax3.set_title('Raw fuzzy output — W\n(before ×fuzzy_w_scale)')
    ax3.set_xlabel('Error (px)')
    ax3.set_ylabel('W fuzzy [−3 – 3]')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(0, color='k', lw=0.8, ls='--')
    ax3.axhline(0, color='k', lw=0.8, ls='--')

    # 4 ── Control curve v(e) with math approximation
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.plot(errors, v_cmd,        'b-',  lw=2.5, label='Mamdani v(e)')
    ax4.plot(errors, v_math(errors), 'b--', lw=1.5,
             label=f'sigmoid\n(v_min={V_MIN}, v_max={V_MAX}, thresh={E_THRESH:.0f}px)')
    ax4.set_title('Control curve — v(e)  [cmd_vel.linear.x]')
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
                 label=f'±max_angular_z ({MAX_ANGULAR_Z})')
    ax5.axhline(-MAX_ANGULAR_Z, color='r', lw=1, ls=':')
    ax5.set_title('Control curve — w(e)  [cmd_vel.angular.z]')
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
def print_key_values(errors, v_cmd, w_cmd):
    idx0 = np.argmin(np.abs(errors))  # index closest to e=0

    print('\n── Key values ──────────────────────────────────────')
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


if __name__ == '__main__':
    print('Building fuzzy system...')
    sim = build_fuzzy_system()

    print('Sweeping error range [-320, 320]...')
    errors, v_raw, w_raw, v_cmd, w_cmd = sweep(sim)

    print_key_values(errors, v_cmd, w_cmd)
    plot(errors, v_raw, w_raw, v_cmd, w_cmd)
