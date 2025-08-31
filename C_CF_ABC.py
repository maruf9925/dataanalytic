# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import gamma

# # =============================
# # Parameters (weekly time unit)
# # =============================
# params = {
#     'Lambda': 707.0302,                # recruitment into S
#     'mu': 1.0 / (74.7 * 52.0),         # natural mortality
#     'beta_c': 0.1543,                  # contact rate (active criminals)
#     'beta_p': 0.0010,                  # contact rate (prison)
#     'eps': 0.43,                       # efficacy of education/intervention
#     'eta': 0.52,                       # proportion observing programs
#     'nu': 0.003,                       # reformed -> out (fully recovered)
#     'theta': 0.7,                      # fraction of reformed relapsing
#     'tau': 0.0100,                     # desistance by criminals
#     'omega': 0.1067,                   # recidivism rate (prison -> criminal)
#     'alpha_param': 0.005,              # incarceration rate
#     'gamma_r': 0.0159                  # prison -> reformed rate
# }

# # ================
# # Initial state y0
# # ================
# S0 = 2_742_386
# C0 = 3_950
# P0 = 2
# R0 = 50
# y0 = np.array([S0, C0, P0, R0], dtype=float)

# # =================
# # Model RHS helpers
# # =================
# def force_lambda(S, C, P, R, par):
#     N = S + C + P + R
#     if N <= 0:
#         return 0.0
#     eff = (1.0 - par['eps'] * par['eta'])
#     return eff * (par['beta_c'] * C + par['beta_p'] * P) / N

# def f_vec(y, par):
#     S, C, P, R = y
#     lam = force_lambda(S, C, P, R, par)
#     dS = par['Lambda'] - lam*S + (1 - par['theta'])*par['nu']*R - par['mu']*S
#     dC = lam*S + par['omega']*P + par['theta']*par['nu']*R - (par['mu'] + par['alpha_param'] + par['tau'])*C
#     dP = par['alpha_param']*C - (par['mu'] + par['gamma_r'] + par['omega'])*P
#     dR = par['gamma_r']*P + par['tau']*C - (par['mu'] + par['nu'])*R
#     return np.array([dS, dC, dP, dR], dtype=float)

# # ==========================================
# # Caputo solver (two-step Lagrange weights)
# # ==========================================
# def caputo_solver(alpha, T=52, h=0.1, y0=y0, par=params):
#     steps = int(T / h)
#     ys = np.zeros((steps + 1, 4), dtype=float)
#     ys[0] = y0
#     fs = np.zeros_like(ys)
#     fs[0] = f_vec(ys[0], par)

#     coef = (h ** alpha) / gamma(alpha)

#     for n in range(steps):
#         sum1 = np.zeros(4)
#         for j in range(0, n + 1):
#             w1 = ((n + 1 - j) ** alpha * (n - j + 2 + alpha)
#                   - (n - j) ** alpha * (n - j + 2 + 2 * alpha)) / (alpha * (alpha + 1))
#             sum1 += fs[j] * w1

#         sum2 = np.zeros(4)
#         for j in range(1, n + 1):
#             w2 = ((n + 1 - j) ** (alpha + 1)
#                   - (n - j) ** alpha * (n - j + 1 + alpha)) / (alpha * (alpha + 1))
#             sum2 += fs[j - 1] * w2

#         ys[n + 1] = y0 + coef * (sum1 - sum2)
#         ys[n + 1] = np.maximum(ys[n + 1], 0.0)
#         fs[n + 1] = f_vec(ys[n + 1], par)

#     t = np.linspace(0.0, T, steps + 1)
#     return t, ys

# # ==========================================================
# # Caputo–Fabrizio solver (NO Euler; exact A/B two-step form)
# # ==========================================================
# def caputo_fabrizio_solver(alpha, T=52, h=0.1, y0=y0, par=params):
#     steps = int(T / h)
#     ys = np.zeros((steps + 1, 4), dtype=float)
#     ys[0] = y0

#     # coefficients from the paper's scheme
#     A = (2.0 - alpha) * (1.0 - alpha) / 2.0 + (3.0 * h / 4.0) * alpha * (2.0 - alpha)
#     B = (2.0 - alpha) * (1.0 - alpha) / 2.0 + (h / 4.0) * alpha * (2.0 - alpha)

#     f0 = f_vec(y0, par)

#     # consistent initialization: set f_{-1} = f_0  -> y1 = y0 + (A - B) f0
#     ys[1] = ys[0] + (A - B) * f0
#     ys[1] = np.maximum(ys[1], 0.0)

#     f_prev = f0
#     f_curr = f_vec(ys[1], par)

#     for n in range(1, steps):
#         ys[n + 1] = ys[n] + A * f_curr - B * f_prev
#         ys[n + 1] = np.maximum(ys[n + 1], 0.0)
#         f_prev, f_curr = f_curr, f_vec(ys[n + 1], par)

#     t = np.linspace(0.0, T, steps + 1)
#     return t, ys

# # ==========================================================
# # ABC (Atangana–Baleanu–Caputo) solver, Caputo sense
# # ==========================================================
# def abc_solver(alpha, T=52, h=0.1, y0=y0, par=params):
#     steps = int(T / h)
#     ys = np.zeros((steps + 1, 4), dtype=float)
#     ys[0] = y0
#     fs = np.zeros_like(ys)
#     fs[0] = f_vec(y0, par)

#     g = gamma(alpha)
#     # B(α) = 1 - α + α/Γ(α)
#     # Using Q,U without dividing by α(α+1) -> absorb into coeff2
#     coeff1 = (g * (1.0 - alpha)) / (g * (1.0 - alpha) + alpha)                 # (1-α)/B(α)
#     coeff2 = 1.0 / ((alpha + 1.0) * ((1.0 - alpha) * g + alpha))               # α/[B(α)Γ(α)] * 1/[α(α+1)]
#     h_alpha = h ** alpha

#     for n in range(steps):
#         sum1 = np.zeros(4)
#         for j in range(0, n + 1):
#             Q = (n + 1 - j) ** alpha * (n - j + 2 + alpha) - (n - j) ** alpha * (n - j + 2 + 2.0 * alpha)
#             sum1 += fs[j] * Q

#         sum2 = np.zeros(4)
#         for j in range(1, n + 1):
#             U = (n + 1 - j) ** (alpha + 1.0) - (n - j) ** alpha * (n - j + 1.0 + alpha)
#             sum2 += fs[j - 1] * U

#         ys[n + 1] = y0 + coeff1 * fs[n] + coeff2 * h_alpha * (sum1 - sum2)
#         ys[n + 1] = np.maximum(ys[n + 1], 0.0)
#         fs[n + 1] = f_vec(ys[n + 1], par)

#     t = np.linspace(0.0, T, steps + 1)
#     return t, ys

# # =================================
# # Integer-order (α = 1) via RK4
# # =================================
# def rk4_rhs(t, y, par):
#     return f_vec(y, par)

# def rk4(f, y0, T=52, h=0.1, par=params):
#     steps = int(T / h)
#     ys = np.zeros((steps + 1, len(y0)), dtype=float)
#     ys[0] = y0
#     t = np.linspace(0.0, T, steps + 1)
#     for n in range(steps):
#         k1 = f(t[n], ys[n], par)
#         k2 = f(t[n] + h / 2.0, ys[n] + h * k1 / 2.0, par)
#         k3 = f(t[n] + h / 2.0, ys[n] + h * k2 / 2.0, par)
#         k4 = f(t[n] + h, ys[n] + h * k3, par)
#         ys[n + 1] = ys[n] + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#         ys[n + 1] = np.maximum(ys[n + 1], 0.0)
#     return t, ys

# # ============================
# # Run & Visualization helpers
# # ============================
# def run_fractional(alphas=(0.50,0.65,0.75,0.85,0.90,0.95,0.99), T=50, h=0.1):
#     methods = {
#         'Caputo Sence': caputo_solver,
#         'Caputo–Fabrizio ': caputo_fabrizio_solver,
#         'ABC (Caputo sense)': abc_solver
#     }
#     for name, solver in methods.items():
#         fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#         axs = axs.ravel()
#         labels = ['S(Susceptible)','C(Criminals)','P(Prison)','R(Reformed)']
#         for alpha in alphas:
#             t, Y = solver(alpha=alpha, T=T, h=h, y0=y0, par=params)
#             for k in range(4):
#                 axs[k].plot(t, Y[:, k], label=f'α={alpha:.2f}',linewidth=2.1)
#                 axs[k].set_title(labels[k])
#                 axs[k].set_xlabel('Time (weeks)')
#                 axs[k].set_ylabel('Population')
#                 axs[k].grid(True, alpha=0.3)
#         for ax in axs:
#             ax.legend()
#         # fig.suptitle(f'Fractional Crime Model({name}) Approach',fontsize=10)
#         plt.tight_layout()
#         plt.show()

# def run_integer(T=52, h=0.1):
#     t, Y = rk4(rk4_rhs, y0, T=T, h=h, par=params)
#     labels = ['S (Susceptible)', 'C (Criminals)', 'P (Prison)', 'R (Reformed)']
#     plt.figure(figsize=(10, 6))
#     for i in range(4):
#         plt.plot(t, Y[:, i], label=labels[i])
#     plt.xlabel("Time (weeks)")
#     plt.ylabel("Population")
#     plt.title("Integer-order Crime Model (RK4 solution, α=1)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()

# # ============================
# # Example run
# # ============================
# if __name__ == "__main__":
#     T = 52
#     h = 0.1
#     run_fractional(alphas=(0.50,0.65,0.75,0.85,0.90,0.95,0.99), T=T, h=h)
#     run_integer(T=T, h=h)





import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# =============================
# Parameters (weekly time unit)
# =============================
params = {
    'Lambda': 707.0302,                # recruitment into S
    'mu': 1.0 / (74.7 * 52.0),         # natural mortality
    'beta_c': 0.1543,                  # contact rate (active criminals)
    'beta_p': 0.0010,                  # contact rate (prison)
    'eps': 0.43,                       # efficacy of education/intervention
    'eta': 0.52,                       # proportion observing programs
    'nu': 0.003,                       # reformed -> out (fully recovered)
    'theta': 0.7,                      # fraction of reformed relapsing
    'tau': 0.0100,                     # desistance by criminals
    'omega': 0.1067,                   # recidivism rate (prison -> criminal)
    'alpha_param': 0.005,              # incarceration rate
    'gamma_r': 0.0159                  # prison -> reformed rate
}

# ================
# Initial state y0
# ================
S0 = 2_742_386
C0 = 3_950
P0 = 2
R0 = 50
y0 = np.array([S0, C0, P0, R0], dtype=float)

# =================
# Model RHS helpers
# =================
def force_lambda(S, C, P, R, par, alpha=1.0):
    N = S + C + P + R
    if N <= 0:
        return 0.0
    eff = (1.0 - par['eps'] * par['eta'])
    # Parameters beta_c and beta_p raised to alpha
    return eff * (par['beta_c']**alpha * C + par['beta_p']**alpha * P) / N

def f_vec(y, par, alpha=1.0):
    S, C, P, R = y
    lam = force_lambda(S, C, P, R, par, alpha=alpha)  # alpha power applied inside force_lambda

    dS = par['Lambda']**alpha - lam*S + (1 - par['theta'])*par['nu']**alpha*R - par['mu']**alpha*S
    dC = lam*S + par['omega']**alpha*P + par['theta']*par['nu']**alpha*R - (par['mu']**alpha + par['alpha_param']**alpha + par['tau']**alpha)*C
    dP = par['alpha_param']**alpha*C - (par['mu']**alpha + par['gamma_r']**alpha + par['omega']**alpha)*P
    dR = par['gamma_r']**alpha*P + par['tau']**alpha*C - (par['mu']**alpha + par['nu']**alpha)*R

    return np.array([dS, dC, dP, dR], dtype=float)


# ==========================================
# Caputo solver (two-step Lagrange weights)
# ==========================================
def caputo_solver(alpha, T=52, h=0.1, y0=y0, par=params):
    steps = int(T / h)
    ys = np.zeros((steps + 1, 4), dtype=float)
    ys[0] = y0
    fs = np.zeros_like(ys)
    fs[0] = f_vec(ys[0], par)

    coef = (h ** alpha) / gamma(alpha)

    for n in range(steps):
        sum1 = np.zeros(4)
        for j in range(0, n + 1):
            w1 = ((n + 1 - j) ** alpha * (n - j + 2 + alpha)
                  - (n - j) ** alpha * (n - j + 2 + 2 * alpha)) / (alpha * (alpha + 1))
            sum1 += fs[j] * w1

        sum2 = np.zeros(4)
        for j in range(1, n + 1):
            w2 = ((n + 1 - j) ** (alpha + 1)
                  - (n - j) ** alpha * (n - j + 1 + alpha)) / (alpha * (alpha + 1))
            sum2 += fs[j - 1] * w2

        ys[n + 1] = y0 + coef * (sum1 - sum2)
        ys[n + 1] = np.maximum(ys[n + 1], 0.0)
        fs[n + 1] = f_vec(ys[n + 1], par)

    t = np.linspace(0.0, T, steps + 1)
    return t, ys

# ==========================================================
# Caputo–Fabrizio solver (NO Euler; exact A/B two-step form)
# ==========================================================
def caputo_fabrizio_solver(alpha, T=52, h=0.1, y0=y0, par=params):
    steps = int(T / h)
    ys = np.zeros((steps + 1, 4), dtype=float)
    ys[0] = y0

    # coefficients from the paper's scheme
    A = (2.0 - alpha) * (1.0 - alpha) / 2.0 + (3.0 * h / 4.0) * alpha * (2.0 - alpha)
    B = (2.0 - alpha) * (1.0 - alpha) / 2.0 + (h / 4.0) * alpha * (2.0 - alpha)

    f0 = f_vec(y0, par)

    # consistent initialization: set f_{-1} = f_0  -> y1 = y0 + (A - B) f0
    ys[1] = ys[0] + (A - B) * f0
    ys[1] = np.maximum(ys[1], 0.0)

    f_prev = f0
    f_curr = f_vec(ys[1], par)

    for n in range(1, steps):
        ys[n + 1] = ys[n] + A * f_curr - B * f_prev
        ys[n + 1] = np.maximum(ys[n + 1], 0.0)
        f_prev, f_curr = f_curr, f_vec(ys[n + 1], par)

    t = np.linspace(0.0, T, steps + 1)
    return t, ys

# ==========================================================
# ABC (Atangana–Baleanu–Caputo) solver, Caputo sense
# ==========================================================
def abc_solver(alpha, T=52, h=0.1, y0=y0, par=params):
    steps = int(T / h)
    ys = np.zeros((steps + 1, 4), dtype=float)
    ys[0] = y0
    fs = np.zeros_like(ys)
    fs[0] = f_vec(y0, par)

    g = gamma(alpha)
    # B(α) = 1 - α + α/Γ(α)
    # Using Q,U without dividing by α(α+1) -> absorb into coeff2
    coeff1 = (g * (1.0 - alpha)) / (g * (1.0 - alpha) + alpha)                 # (1-α)/B(α)
    coeff2 = 1.0 / ((alpha + 1.0) * ((1.0 - alpha) * g + alpha))               # α/[B(α)Γ(α)] * 1/[α(α+1)]
    h_alpha = h ** alpha

    for n in range(steps):
        sum1 = np.zeros(4)
        for j in range(0, n + 1):
            Q = (n + 1 - j) ** alpha * (n - j + 2 + alpha) - (n - j) ** alpha * (n - j + 2 + 2.0 * alpha)
            sum1 += fs[j] * Q

        sum2 = np.zeros(4)
        for j in range(1, n + 1):
            U = (n + 1 - j) ** (alpha + 1.0) - (n - j) ** alpha * (n - j + 1.0 + alpha)
            sum2 += fs[j - 1] * U

        ys[n + 1] = y0 + coeff1 * fs[n] + coeff2 * h_alpha * (sum1 - sum2)
        ys[n + 1] = np.maximum(ys[n + 1], 0.0)
        fs[n + 1] = f_vec(ys[n + 1], par)

    t = np.linspace(0.0, T, steps + 1)
    return t, ys

# =================================
# Integer-order (α = 1) via RK4
# =================================
def rk4_rhs(t, y, par):
    return f_vec(y, par)

def rk4(f, y0, T=50, h=0.1, par=params):
    steps = int(T / h)
    ys = np.zeros((steps + 1, len(y0)), dtype=float)
    ys[0] = y0
    t = np.linspace(0.0, T, steps + 1)
    for n in range(steps):
        k1 = f(t[n], ys[n], par)
        k2 = f(t[n] + h / 2.0, ys[n] + h * k1 / 2.0, par)
        k3 = f(t[n] + h / 2.0, ys[n] + h * k2 / 2.0, par)
        k4 = f(t[n] + h, ys[n] + h * k3, par)
        ys[n + 1] = ys[n] + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        ys[n + 1] = np.maximum(ys[n + 1], 0.0)
    return t, ys

# ============================
# Run & Visualization helpers
# ============================
def run_fractional(alphas=(0.50,0.65,0.75,0.85,0.90,0.95,0.99,1), T=52, h=0.1):
    methods = {
        'Caputo': caputo_solver,
        'Caputo–Fabrizio': caputo_fabrizio_solver,
        'ABC (Caputo sense)': abc_solver
    }
    for name, solver in methods.items():
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.ravel()
        labels = ['S(Susceptible)', 'C(Criminals)', 'P(Prison)', 'R(Reformed)']
        for alpha in alphas:
            t, Y = solver(alpha=alpha, T=T, h=h, y0=y0, par=params)
            for k in range(4):
                axs[k].plot(t, Y[:, k], label=f'α={alpha:.2f}',linewidth=2.1)
                axs[k].set_title(labels[k])
                axs[k].set_xlabel('Time (weeks)')
                axs[k].set_ylabel('Population')
                axs[k].grid(True, alpha=0.3)
        for ax in axs:
            ax.legend()
        plt.tight_layout()
        plt.show()

def run_integer(T=50, h=1.0):
    t, Y = rk4(rk4_rhs, y0, T=T, h=h, par=params)
    labels = ['S (Susceptible)', 'C (Criminals)', 'P (Prison)', 'R (Reformed)']
    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(t, Y[:, i], label=labels[i])
    plt.xlabel("Time (weeks)")
    plt.ylabel("Population")
    plt.title("Integer-order Crime Model (RK4 solution, α=1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ============================
# Example run
# ============================
if __name__ == "__main__":
    T = 52
    h = 0.1
    run_fractional(alphas=(0.50,0.65,0.75,0.85,0.90,0.95,0.99,1), T=T, h=h)
    run_integer(T=T, h=h)


