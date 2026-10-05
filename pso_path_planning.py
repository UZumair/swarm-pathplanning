"""
Swarm Intelligence Lab - Assignment 1
PSO-based path planning on a 2D grid with obstacles.

Every problem instance (obstacles, start, goal) is generated programmatically
from ROLL_NUMBER used as the random seed. Nothing is hardcoded.

Usage:
    python pso_path_planning.py                 # uses ROLL_NUMBER below
    python pso_path_planning.py --roll 12345    # override seed from CLI
"""
import argparse
import json
import os
import random
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------------------
# STUDENT DETAILS  (CHANGE THESE)
# ----------------------------------------------------------------------------
STUDENT_NAME = "Your Name"
ROLL_NUMBER = 12345          # <-- used as the random SEED
# ----------------------------------------------------------------------------

# Problem parameters
GRID_SIZE = 20               # N x N grid
OBSTACLE_DENSITY = 0.20      # fraction of cells blocked

# PSO parameters
N_WAYPOINTS = 8              # intermediate points per path (dimension = 2*N)
N_PARTICLES = 120
N_ITERATIONS = 300
W_START, W_END = 0.9, 0.4    # linearly decreasing inertia weight
C1, C2 = 1.6, 1.6            # cognitive / social coefficients
PENALTY = 50.0               # cost per colliding sample point
SAMPLE_STEP = 0.1            # collision-check resolution along segments


# ----------------------------------------------------------------------------
# 1. PROBLEM GENERATION
# ----------------------------------------------------------------------------
def bfs_reachable(grid, start, goal):
    """4-connected BFS, used only to guarantee the generated instance is solvable."""
    n = grid.shape[0]
    seen, q = {start}, deque([start])
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and grid[ny, nx] == 0 and (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    return False


def generate_problem(seed, n=GRID_SIZE, density=OBSTACLE_DENSITY):
    """Generate obstacles + start + goal from the seed. grid[y, x] = 1 means blocked."""
    random.seed(seed)
    np.random.seed(seed)
    while True:  # deterministic retry loop (same seed -> same final instance)
        grid = np.zeros((n, n), dtype=int)
        for _ in range(int(density * n * n)):
            grid[random.randrange(n), random.randrange(n)] = 1
        free = [(x, y) for y in range(n) for x in range(n) if grid[y, x] == 0]
        start, goal = random.sample(free, 2)
        # keep start and goal reasonably far apart so the task is non-trivial
        if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < n // 2:
            continue
        if bfs_reachable(grid, start, goal):
            return grid, start, goal


# ----------------------------------------------------------------------------
# 2. PATH REPRESENTATION & FITNESS
# ----------------------------------------------------------------------------
def build_path(particle, start, goal):
    """Particle = flat [x1,y1,...,xk,yk] waypoints -> full polyline incl. start/goal."""
    wp = particle.reshape(-1, 2)
    s = np.array(start, float) + 0.5      # cell centres
    g = np.array(goal, float) + 0.5
    return np.vstack([s, wp, g])


def collision_count(path, grid):
    """Number of sampled points along the polyline lying in blocked cells / outside grid."""
    n = grid.shape[0]
    count = 0
    for a, b in zip(path[:-1], path[1:]):
        L = np.linalg.norm(b - a)
        k = max(2, int(L / SAMPLE_STEP))
        t = np.linspace(0, 1, k)[:, None]
        pts = a + t * (b - a)
        out = (pts < 0).any(1) | (pts >= n).any(1)
        ix = np.clip(pts[:, 0].astype(int), 0, n - 1)
        iy = np.clip(pts[:, 1].astype(int), 0, n - 1)
        count += int(np.sum(out | (grid[iy, ix] == 1)))
    return count


def path_length(path):
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def fitness(particle, grid, start, goal):
    path = build_path(particle, start, goal)
    return path_length(path) + PENALTY * collision_count(path, grid)


# ----------------------------------------------------------------------------
# 3. PSO CORE
# ----------------------------------------------------------------------------
def pso(grid, start, goal, seed):
    rng = np.random.default_rng(seed)
    n = grid.shape[0]
    dim = 2 * N_WAYPOINTS
    s = np.array(start, float) + 0.5
    g = np.array(goal, float) + 0.5

    # Initialise: half the swarm near the straight line, half uniformly random
    pos = rng.uniform(0, n, (N_PARTICLES, dim))
    for i in range(N_PARTICLES // 2):
        t = np.sort(rng.uniform(0, 1, N_WAYPOINTS))[:, None]
        line = s + t * (g - s)
        pos[i] = np.clip(line + rng.normal(0, 2.5, line.shape), 0, n - 1e-6).ravel()
    vmax = 0.2 * n
    vel = rng.uniform(-vmax, vmax, (N_PARTICLES, dim))

    pbest = pos.copy()
    pbest_f = np.array([fitness(p, grid, start, goal) for p in pos])
    gi = int(np.argmin(pbest_f))
    gbest, gbest_f = pbest[gi].copy(), pbest_f[gi]
    history = [gbest_f]

    for it in range(N_ITERATIONS):
        w = W_START - (W_START - W_END) * it / (N_ITERATIONS - 1)
        r1, r2 = rng.random((2, N_PARTICLES, dim))
        vel = w * vel + C1 * r1 * (pbest - pos) + C2 * r2 * (gbest - pos)
        vel = np.clip(vel, -vmax, vmax)
        pos = np.clip(pos + vel, 0, n - 1e-6)

        f = np.array([fitness(p, grid, start, goal) for p in pos])
        better = f < pbest_f
        pbest[better], pbest_f[better] = pos[better], f[better]
        gi = int(np.argmin(pbest_f))
        if pbest_f[gi] < gbest_f:
            gbest, gbest_f = pbest[gi].copy(), pbest_f[gi]
        history.append(gbest_f)
        if (it + 1) % 25 == 0 or it == 0:
            print(f"  iter {it+1:3d}/{N_ITERATIONS}  best fitness = {gbest_f:8.3f}")
    return build_path(gbest, start, goal), gbest_f, history


# ----------------------------------------------------------------------------
# 4. BASELINE (for comparison only) & VISUALISATION
# ----------------------------------------------------------------------------
def bfs_shortest(grid, start, goal):
    n = grid.shape[0]
    prev, q = {start: None}, deque([start])
    while q:
        c = q.popleft()
        if c == goal:
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (c[0] + dx, c[1] + dy)
            if 0 <= nb[0] < n and 0 <= nb[1] < n and grid[nb[1], nb[0]] == 0 and nb not in prev:
                prev[nb] = c
                q.append(nb)
    p, c = [], goal
    while c is not None:
        p.append(c)
        c = prev[c]
    return len(p) - 1


def plot_result(grid, start, goal, path, history, seed, name, length, collisions, out_dir):
    n = grid.shape[0]
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(grid, cmap=ListedColormap(["white", "#333333"]), origin="lower",
              extent=(0, n, 0, n))
    ax.set_xticks(range(n + 1)); ax.set_yticks(range(n + 1))
    ax.grid(True, color="#cccccc", linewidth=0.5)
    ax.tick_params(labelsize=7)
    ax.plot(path[:, 0], path[:, 1], "-o", color="crimson", lw=2, ms=4, label="PSO path")
    ax.scatter(*(np.array(start) + .5), s=220, c="limegreen", marker="s",
               edgecolors="k", zorder=5, label=f"Start {start}")
    ax.scatter(*(np.array(goal) + .5), s=260, c="gold", marker="*",
               edgecolors="k", zorder=5, label=f"Goal {goal}")
    ax.set_title(f"PSO Path Planning | {name} | seed={seed}\n"
                 f"length={length:.2f}, collisions={collisions}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "final_path.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, color="navy")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best fitness (length + penalty)")
    ax.set_title("PSO convergence"); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "convergence.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", type=int, default=ROLL_NUMBER, help="roll number = seed")
    ap.add_argument("--name", type=str, default=STUDENT_NAME)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seed = args.roll
    grid, start, goal = generate_problem(seed)
    print("=" * 60)
    print(f"Student : {args.name}\nSeed    : {seed}")
    print(f"Grid    : {GRID_SIZE}x{GRID_SIZE}   Obstacles: {int(grid.sum())}")
    print(f"Start   : {start}   Goal: {goal}")
    print("=" * 60)

    # PSO is stochastic: a few independent restarts (seeded from roll number)
    best = None
    for r in range(5):
        print(f"\n--- PSO run {r+1} ---")
        path, f, hist = pso(grid, start, goal, seed + r)
        coll = collision_count(path, grid)
        print(f"  run {r+1}: length={path_length(path):.3f} collisions={coll} fitness={f:.3f}")
        if best is None or f < best[1]:
            best = (path, f, hist)
        if coll == 0 and r >= 1:      # good enough: feasible after >=2 runs
            break
    path, f, hist = best
    L, coll = path_length(path), collision_count(path, grid)

    print("\n" + "=" * 60)
    print("RESULT")
    print(f"  Best path length : {L:.3f}")
    print(f"  Collisions       : {coll}  ({'OBSTACLE-FREE' if coll == 0 else 'NOT feasible'})")
    print(f"  BFS grid-optimal : {bfs_shortest(grid, start, goal)} steps (4-connected baseline)")
    print("  Waypoints (x, y):")
    for i, p in enumerate(path):
        print(f"    {i:2d}: ({p[0]:6.2f}, {p[1]:6.2f})")
    print("=" * 60)

    plot_result(grid, start, goal, path, hist, seed, args.name, L, coll, args.out)
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump({"name": args.name, "seed": seed, "start": start, "goal": goal,
                   "length": L, "collisions": coll, "path": path.tolist()}, fh, indent=2)
    print(f"Saved plots/results to ./{args.out}/")


if __name__ == "__main__":
    main()
