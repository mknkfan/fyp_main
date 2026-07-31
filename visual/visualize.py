from func.datastruct import Point, Machine
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
from typing import List, Tuple, Optional, Dict

def visualize_layout(machines: List[Machine], robot_position: Point, 
                    sequence: List[int], workspace_bounds: Tuple[float, float, float, float]):
    """Visualize the final layout"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Draw workspace bounds
    min_x, max_x, min_y, max_y = workspace_bounds
    workspace_rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                              linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
    ax.add_patch(workspace_rect)
    
    # Color map for machines
    colors = plt.cm.Set3(np.linspace(0, 1, len(machines)))
    
    # Draw machines
    for i, machine in enumerate(machines):
        corners = machine.get_corners()
        polygon_coords = [(p.x, p.y) for p in corners]
        
        polygon = Polygon(polygon_coords, closed=True, 
                         facecolor=colors[i], edgecolor='black', alpha=0.7)
        ax.add_patch(polygon)
        
        # Label machine
        ax.text(machine.position.x, machine.position.y, f'M{machine.id}',
               ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw access point
        access_point = machine.get_access_point_world()
        ax.plot(access_point.x, access_point.y, 'ro', markersize=8)
        '''ax.text(access_point.x + 0.5, access_point.y + 0.5, f'A{machine.id}',
               fontsize=8, color='red')'''
    
    # Draw robot position
    ax.plot(robot_position.x, robot_position.y, 'bs', markersize=12, label='Robot')
    
    # Draw sequence path
    path_points = [robot_position]
    for machine_id in sequence:
        machine = next(m for m in machines if m.id == machine_id)
        path_points.append(machine.get_access_point_world())
    path_points.append(robot_position)
    
    path_x = [p.x for p in path_points]
    path_y = [p.y for p in path_points]
    ax.plot(path_x, path_y, 'r--', linewidth=2, alpha=0.7, label='Robot Path')
    
    # Add arrows to show direction
    for i in range(len(path_points) - 1):
        dx = path_points[i+1].x - path_points[i].x
        dy = path_points[i+1].y - path_points[i].y
        arrow = FancyArrowPatch(
            (path_points[i].x, path_points[i].y),   # start
            (path_points[i+1].x, path_points[i+1].y), # end
            arrowstyle='->',      # clean arrowhead style
            mutation_scale=15,     # size of the arrowhead
            color='red',
            lw=2,
            alpha=0.7
        )
        ax.add_patch(arrow)

        """ax.arrow(path_points[i].x, path_points[i].y, dx*0.8, dy*0.8,
                head_width=0.5, head_length=0.3, fc='red', ec='red', alpha=0.6)"""
    
    ax.set_xlim(min_x - 2, max_x + 2)
    ax.set_ylim(min_y - 2, max_y + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Optimized Robotic Workcell Layout', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    plt.tight_layout()
    plt.show()

def plot_optimization_progress(results: Dict):
    """Plot optimization progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Fitness evolution
    ax1.plot(results['best_fitness_history'], 'b-', linewidth=2, label='Best Fitness')
    ax1.plot(results['avg_fitness_history'], 'r--', linewidth=2, label='Average Fitness')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness (Distance)')
    ax1.set_title('Fitness Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Convergence rate
    best_fitness = np.array(results['best_fitness_history'])
    improvement = np.diff(best_fitness)
    ax2.plot(improvement, 'g-', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Improvement')
    ax2.set_title('Convergence Rate')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def visualize_layout_generation(decoded_machine_list: List[List[Machine]], 
                            robot_position: Point,
                            sequence: List[int], 
                            workspace_bounds: Tuple[float, float, float, float],
                            snapshot_interval: int = 10):

    num_snapshots = len(decoded_machine_list)

    for i, machines in enumerate(decoded_machine_list):
        generation_num = i * snapshot_interval
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Draw workspace bounds
        min_x, max_x, min_y, max_y = workspace_bounds
        workspace_rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                    linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(workspace_rect)
        
        # Color map for machines
        colors = plt.cm.Set3(np.linspace(0, 1, len(machines)))
        
        # Draw machines
        for j, machine in enumerate(machines):
            corners = machine.get_corners()
            polygon_coords = [(p.x, p.y) for p in corners]
            
            polygon = Polygon(polygon_coords, closed=True, 
                            facecolor=colors[j], edgecolor='black', alpha=0.7, linewidth=2)
            ax.add_patch(polygon)
            
            # Label machine
            ax.text(machine.position.x, machine.position.y, f'M{machine.id}',
                    ha='center', va='center', fontweight='bold', fontsize=10)
            
            # Draw access point
            access_point = machine.get_access_point_world()
            ax.plot(access_point.x, access_point.y, 'ro', markersize=8)
        
        # Draw robot position
        ax.plot(robot_position.x, robot_position.y, 'bs', markersize=12, label='Robot Start')
        
        # Draw sequence path
        path_points = [robot_position]
        for machine_id in sequence:
            machine = next(m for m in machines if m.id == machine_id)
            path_points.append(machine.get_access_point_world())
        path_points.append(robot_position)
        
        path_x = [p.x for p in path_points]
        path_y = [p.y for p in path_points]
        ax.plot(path_x, path_y, 'r--', linewidth=2, alpha=0.7, label='Robot Path')
        
        # Add arrows to show direction
        for k in range(len(path_points) - 1):
            arrow = FancyArrowPatch(
                (path_points[k].x, path_points[k].y),
                (path_points[k+1].x, path_points[k+1].y),
                arrowstyle='->',
                mutation_scale=15,
                color='red',
                lw=2,
                alpha=0.7
            )
            ax.add_patch(arrow)
        
        ax.set_xlim(min_x - 2, max_x + 2)
        ax.set_ylim(min_y - 2, max_y + 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Robotic Workcell Layout Generation {generation_num}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        plt.tight_layout()
        plt.savefig(f'layout_generation_{generation_num:03d}.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    print(f"Generated {num_snapshots} layout visualizations")

def create_layout_animation(decoded_evolution: List[List[Machine]], 
                           robot_position: Point,
                           sequence: List[int], 
                           workspace_bounds: Tuple[float, float, float, float],
                           snapshot_interval: int = 10,
                           output_filename: str = 'layout_evolution.gif',
                           duration: float = 1.0):

    import matplotlib.animation as animation
    from matplotlib.animation import PillowWriter
    
    num_snapshots = len(decoded_evolution)
    min_x, max_x, min_y, max_y = workspace_bounds
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    def update(frame_num):
        ax.clear()
        
        machines = decoded_evolution[frame_num]
        generation_num = frame_num * snapshot_interval
        
        # Draw workspace bounds
        workspace_rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                  linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(workspace_rect)
        
        # Color map for machines
        colors = plt.cm.Set3(np.linspace(0, 1, len(machines)))
        
        # Draw machines
        for j, machine in enumerate(machines):
            corners = machine.get_corners()
            polygon_coords = [(p.x, p.y) for p in corners]
            
            polygon = Polygon(polygon_coords, closed=True, 
                            facecolor=colors[j], edgecolor='black', alpha=0.7, linewidth=2)
            ax.add_patch(polygon)
            
            # Label machine
            ax.text(machine.position.x, machine.position.y, f'M{machine.id}',
                   ha='center', va='center', fontweight='bold', fontsize=10)
            
            # Draw access point
            access_point = machine.get_access_point_world()
            ax.plot(access_point.x, access_point.y, 'ro', markersize=8)
        
        # Draw robot position
        ax.plot(robot_position.x, robot_position.y, 'bs', markersize=12, label='Robot Start')
        
        # Draw sequence path
        path_points = [robot_position]
        for machine_id in sequence:
            machine = next(m for m in machines if m.id == machine_id)
            path_points.append(machine.get_access_point_world())
        path_points.append(robot_position)
        
        path_x = [p.x for p in path_points]
        path_y = [p.y for p in path_points]
        ax.plot(path_x, path_y, 'r--', linewidth=2, alpha=0.7, label='Robot Path')
        
        # Add arrows
        for k in range(len(path_points) - 1):
            arrow = FancyArrowPatch(
                (path_points[k].x, path_points[k].y),
                (path_points[k+1].x, path_points[k+1].y),
                arrowstyle='->',
                mutation_scale=15,
                color='red',
                lw=2,
                alpha=0.7
            )
            ax.add_patch(arrow)
        
        ax.set_xlim(min_x - 2, max_x + 2)
        ax.set_ylim(min_y - 2, max_y + 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Robotic Workcell Layout Generation {generation_num}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=num_snapshots, 
                                  interval=duration*1000, repeat=True)
    
    # Save as GIF
    writer = PillowWriter(fps=1/duration)
    anim.save(output_filename, writer=writer)
    
    plt.close()
    print(f"Animation saved as {output_filename}")

import matplotlib.pyplot as plt

def plot_rl_vs_ga_parameters(results: dict, save_path: str = None):
    """
    Plot RL-GA parameter evolution vs constant Pure GA parameters.
    """

    rl_log = results["rl_log"]

    generations = [entry["generation"] for entry in rl_log]

    mutation_rates = [entry["mutation_rate"] for entry in rl_log]
    crossover_rates = [entry["crossover_rate"] for entry in rl_log]
    ls_probs = [entry["local_search_prob"] for entry in rl_log]

    # Pure GA constants
    GA_MUT = 0.1
    GA_CROSS = 0.8
    GA_LS = 0.1

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # -------- Mutation Rate --------
    axes[0].plot(generations, mutation_rates, linewidth=2, label="RL-GA")
    axes[0].axhline(GA_MUT, color = "red", linestyle="--", linewidth=2, label="Pure GA")
    axes[0].set_ylabel("Mutation Rate")
    axes[0].set_title("GA Parameter Adaptation (RL-GA vs Pure GA)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # -------- Crossover Rate --------
    axes[1].plot(generations, crossover_rates, linewidth=2, label="RL-GA")
    axes[1].axhline(GA_CROSS, color="red",linestyle="--", linewidth=2, label="Pure GA")
    axes[1].set_ylabel("Crossover Rate")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # -------- Local Search Probability --------
    axes[2].plot(generations, ls_probs, linewidth=2, label="RL-GA")
    axes[2].axhline(GA_LS, color="red",linestyle="--", linewidth=2, label="Pure GA")
    axes[2].set_ylabel("Local Search Probability")
    axes[2].set_xlabel("Generation")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved parameter comparison plot to {save_path}")

    plt.show()


# ============================================================
# RL agent training curves
# ============================================================

# Categorical slots 1-3 of the reference palette (validated for all-pairs
# separation under colour-vision deficiency in both light and dark modes).
_SERIES_1 = "#2a78d6"  # blue
_SERIES_2 = "#eb6834"  # orange
_SERIES_3 = "#1baf7a"  # aqua
_INK_MUTED = "#8a8a85"


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean, shrinking the window at the edges."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return values
    window = max(1, min(window, n))
    out = np.empty(n, dtype=float)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.mean(values[lo:hi])
    return out


def plot_rl_training_progress(
    history: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot how the RL agent evolves over a full training session.

    `history` is the dict returned by algo.train_rl_agent.train_agent()
    (also saved as an .npz), containing per-run and per-generation records.

    Four panels, each on a single y-axis:
      1. Reward per training run (raw + rolling mean over one full pass)
      2. Reward per credited generation across the whole session (smoothed)
      3. Normalised best fitness vs pass number - did solutions get better?
      4. Q-table update magnitude per run - did the agent converge?
    """
    run_reward_sum = np.asarray(history["run_reward_sum"], dtype=float)
    run_problem = np.asarray(history["run_problem"], dtype=int)
    run_best_fitness = np.asarray(history["run_best_fitness"], dtype=float)
    run_q_delta = np.asarray(history["run_q_delta"], dtype=float)
    gen_rewards = np.asarray(history["gen_rewards"], dtype=float)

    num_problems = int(history["num_problems"])
    runs_per_problem = int(history["runs_per_problem"])
    num_runs = len(run_reward_sum)
    runs = np.arange(1, num_runs + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # -------- 1. Reward per run --------
    ax = axes[0, 0]
    ax.plot(runs, run_reward_sum, linewidth=1, color=_INK_MUTED, alpha=0.7,
            label="Per run")
    ax.plot(runs, _rolling_mean(run_reward_sum, num_problems), linewidth=2,
            color=_SERIES_1, label=f"Rolling mean ({num_problems} runs)")
    ax.axhline(0, color="black", linewidth=1, alpha=0.4)
    ax.set_xlabel("Training run")
    ax.set_ylabel("Total reward")
    ax.set_title("Reward per training run")
    ax.legend()
    ax.grid(alpha=0.3)

    # -------- 2. Reward per credited generation --------
    ax = axes[0, 1]
    if len(gen_rewards):
        gens = np.arange(1, len(gen_rewards) + 1)
        # Keep the smoothed line meaningfully different from the raw series,
        # otherwise it just paints over it and the legend lies.
        window = max(5, len(gen_rewards) // 100)
        ax.plot(gens, gen_rewards, linewidth=0.5, color=_INK_MUTED, alpha=0.35,
                label="Per decision")
        ax.plot(gens, _rolling_mean(gen_rewards, window), linewidth=2,
                color=_SERIES_2, label=f"Rolling mean ({window} decisions)")
        ax.axhline(0, color="black", linewidth=1, alpha=0.4)
        ax.legend()
    ax.set_xlabel("RL decision (cumulative over training)")
    ax.set_ylabel("Reward")
    ax.set_title("Reward signal over the whole session")
    ax.grid(alpha=0.3)

    # -------- 3. Normalised best fitness vs pass number --------
    # Each problem's runs are divided by that problem's own mean, so the 15
    # layouts (which have very different absolute distances) are comparable.
    ax = axes[1, 0]
    passes = np.arange(1, runs_per_problem + 1)
    normalised = np.full((num_problems, runs_per_problem), np.nan)
    for p in range(num_problems):
        vals = run_best_fitness[run_problem == p]
        if len(vals) and np.mean(vals) != 0:
            normalised[p, : len(vals)] = vals[:runs_per_problem] / np.mean(vals)

    mean_by_pass = np.nanmean(normalised, axis=0)
    std_by_pass = np.nanstd(normalised, axis=0)
    ax.fill_between(passes, mean_by_pass - std_by_pass, mean_by_pass + std_by_pass,
                    color=_SERIES_3, alpha=0.15, label="+/- 1 SD across problems")
    ax.plot(passes, mean_by_pass, linewidth=2, color=_SERIES_3, marker="o",
            markersize=6, label="Mean across problems")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_xlabel("Pass number (run index within each problem)")
    ax.set_ylabel("Best fitness / problem mean")
    ax.set_title("Solution quality as training progresses")
    ax.set_xticks(passes)
    ax.legend()
    ax.grid(alpha=0.3)

    # -------- 4. Q-table convergence --------
    ax = axes[1, 1]
    ax.plot(runs, run_q_delta, linewidth=1, color=_INK_MUTED, alpha=0.7,
            label="Per run")
    ax.plot(runs, _rolling_mean(run_q_delta, num_problems), linewidth=2,
            color=_SERIES_1, label=f"Rolling mean ({num_problems} runs)")
    ax.set_xlabel("Training run")
    ax.set_ylabel("||Q_after - Q_before||")
    ax.set_title("Q-table update magnitude (convergence)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"RL agent training: {num_problems} problems x {runs_per_problem} runs "
        f"= {num_runs} GA runs",
        fontsize=13,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved RL training progress plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_q_table(q_table: np.ndarray, save_path: Optional[str] = None, show: bool = True):
    """
    Heat map of the learned Q-table (states x actions).

    Sequential single-hue ramp for magnitude; values are printed in each cell so
    the reading never depends on colour alone.
    """
    action_labels = [
        "mut+", "mut-", "cross+", "cross-",
        "LS+", "LS-", "cycle mut", "soft reset",
    ]
    state_labels = [
        f"div{d}/imp{i}" for d in ("lo", "md", "hi") for i in ("neg", "sml", "big")
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(q_table, cmap="Blues", aspect="auto")

    ax.set_xticks(range(q_table.shape[1]))
    ax.set_xticklabels(action_labels[: q_table.shape[1]], rotation=30, ha="right")
    ax.set_yticks(range(q_table.shape[0]))
    ax.set_yticklabels(state_labels[: q_table.shape[0]])
    ax.set_xlabel("Action")
    ax.set_ylabel("State (diversity / improvement bin)")
    ax.set_title("Learned Q-table")

    # Direct value labels - identity never rests on colour alone.
    # Q values are small, so pick a precision that does not collapse to "0.000",
    # and threshold the label colour on imshow's own min..max normalisation.
    lo, hi = float(np.min(q_table)), float(np.max(q_table))
    span = (hi - lo) or 1.0
    scale = np.max(np.abs(q_table))
    decimals = 3 if scale >= 0.01 else 5
    for r in range(q_table.shape[0]):
        for c in range(q_table.shape[1]):
            val = float(q_table[r, c])
            label = "0" if val == 0 else f"{val:.{decimals}f}"
            ax.text(c, r, label, ha="center", va="center", fontsize=8,
                    color="#ffffff" if (val - lo) / span > 0.6 else "#0b0b0b")

    fig.colorbar(im, ax=ax, label="Q value")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Q-table heat map to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)