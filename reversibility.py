import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
L = 120.0
N = 1200
dx = L / N
dt = 0.003
steps_per_frame = 10
hbar = 1.0
m = 1.0
x = np.linspace(-L / 2, L / 2, N)
def gaussian_wavepacket(x_vals, x0, sigma, k0):
    psi = np.exp(-(x_vals - x0) ** 2 / (4 * sigma ** 2)) * np.exp(1j * k0 * x_vals)
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi / norm
def crank_nicolson_solver(V_vals):
    k_const = hbar ** 2 / (2 * m * dx ** 2)
    gamma = 1j * dt / (2 * hbar)
    main_diag_A = 1.0 + gamma * (2 * k_const + V_vals)
    off_diag_A = -gamma * k_const * np.ones(N - 1)
    main_diag_B = 1.0 - gamma * (2 * k_const + V_vals)
    off_diag_B = gamma * k_const * np.ones(N - 1)
    A = sparse.diags([off_diag_A, main_diag_A, off_diag_A], offsets=[-1, 0, 1], format='csc')
    B = sparse.diags([off_diag_B, main_diag_B, off_diag_B], offsets=[-1, 0, 1], format='csc')
    return linalg.factorized(A), B
free_V = np.zeros(N)
omega = 0.5
harmonic_V = 0.5 * m * omega ** 2 * x ** 2
wall_width = 30.0
wall_height = 300.0
box_V = np.zeros(N)
box_V[x < (-L / 2 + wall_width)] = wall_height
box_V[x > (L / 2 - wall_width)] = wall_height
barrier_left = 5.0
barrier_width = 10.0
barrier_right = barrier_left + barrier_width
barrier_height = 40 
barrier_V = np.zeros(N)
barrier_V[(x > barrier_left) & (x < barrier_right)] = barrier_height
sim_configs = [
{
        "title": "Free Particle",
        "V": free_V,
        "x0": -40.0,
        "k0": 6.0,       
        "psi0": gaussian_wavepacket(x, -40.0, 3.5, 6.0),
        "markers": [],
    },
    {
        "title": "Harmonic Oscillator",
        "V": harmonic_V,
        "x0": -25.0,
        "k0": 0.0,
        "psi0": gaussian_wavepacket(x, -25.0, 1.0, 0.0),
        "markers": [
            {"type": "vline", "x": 0.0, "color": "#555555", "label": "Equilibrium"}
        ],
    },
    {
        "title": "Particle in a Box",
        "V": box_V,
        "x0": -10.0,
        "k0": 7.0,         
        "psi0": gaussian_wavepacket(x, -20.0, 2.5, 7.0),
        "markers": [
            {"type": "span", "x0": -L / 2, "x1": -L / 2 + wall_width, "color": "#b5b5b5"},
            {"type": "span", "x0": L / 2 - wall_width, "x1": L / 2, "color": "#b5b5b5"},
        ],
    },
    {
        "title": "Potential Barrier",
        "V": barrier_V,
        "x0": -35.0,
        "k0": 10,
        "psi0": gaussian_wavepacket(x, -35.0, 3.0, 10),
        "markers": [
            {"type": "span", "x0": barrier_left, "x1": barrier_right, "color": "#c9c9c9"},
            {"type": "vline", "x": barrier_left, "color": "#666666", "label": "Barrier"},
            {"type": "vline", "x": barrier_right, "color": "#666666"},
        ],
    },
]
plot_margin = 5.0
T0=1.7
time_steps = int(T0 / dt)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()
sim_states = []
for ax, config in zip(axes, sim_configs):
    V_vals = config["V"]
    psi0 = config["psi0"]
    solver, B = crank_nicolson_solver(V_vals)
    max_V = np.max(V_vals)
    scale = 0.12 / max_V if max_V > 0 else 0.0
    if max_V > 0:
        ax.plot(x, V_vals * scale, color="#777777", linestyle="--")
    for marker in config["markers"]:
        if marker["type"] == "span":
            ax.axvspan(marker["x0"], marker["x1"], color=marker["color"], alpha=0.35)
        elif marker["type"] == "vline":
            ax.axvline(marker["x"], color=marker["color"], linestyle=":", alpha=0.9)
    line, = ax.plot(x, np.abs(psi0) ** 2, color="#1f77b4")
    ax.set_title(config["title"])
    ax.set_xlim(-L / 2, L / 2)
    ax.set_ylim(0, 0.45)
    ax.grid(alpha=0.2)
    sim_states.append({
        "psi": psi0.copy(),
        "psi_init": psi0.copy(),  
        "solver": solver,
        "B": B,
        "line": line,
    })
fig.tight_layout()
def update(frame):
    updated = []
    if frame == 0:
        for ax in axes:
            for txt in ax.texts:
                txt.remove()
        for state in sim_states:
            state["psi"] = state["psi_init"].copy()
    if frame == 320:
        for state in sim_states:
            for ax in axes:
                ax.text(0.5, 0.5, "Reversing Time", transform=ax.transAxes, fontsize=16, color="#ff5555", ha="center", va="center", alpha=0.8)
            state["psi"] = np.conj(state["psi"])
    for state in sim_states:
        for _ in range(steps_per_frame):
            d = state["B"].dot(state["psi"])
            state["psi"] = state["solver"](d)
        state["line"].set_ydata(np.abs(state["psi"]) ** 2)
        updated.append(state["line"])
    return updated
ani = animation.FuncAnimation(
    fig,
    update,
    frames=time_steps,
    interval=10,
    blit=False,
    repeat=True
)
# writer = FFMpegWriter(fps=60)
# ani.save('simulation_cycle.mp4', writer=writer, dpi=150)
plt.show()