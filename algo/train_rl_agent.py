"""
train_rl_agent.py

Train a single Q-learning agent that controls GA parameters across
multiple machine-layout problems, then save its Q-table to disk.

Usage (from project root):
    python -m algo.train_rl_agent

Or:
    python algo/train_rl_agent.py
"""

from typing import List, Tuple
import numpy as np

from func.datastruct import Point, Machine
from algo.agent import QLearningAgent
from algo.gen_algo_rl_test import RLGeneticAlgorithm


# ============================================================
# 1. Define training problems 
# ============================================================

def build_training_problems() -> List[Tuple[List[Machine], List[int], Point, Tuple[float, float, float, float]]]:
    problems: List[Tuple[List[Machine], List[int], Point, Tuple[float, float, float, float]]] = []

    # ============================================================
    # PROBLEM SET A
    # ============================================================
    machines_A = [
        Machine(id=1, shape='rectangle', width=4.0, height=3.0, 
                access_point=Point(1.5, 0)),
        Machine(id=2, shape='l_shape', width=5.0, height=4.0, 
                access_point=Point(-1.0, 1.0),
                l_cutout_width=2.0, l_cutout_height=2.0),
        Machine(id=3, shape='rectangle', width=3.5, height=2.5, 
                access_point=Point(0, -1.0)),
        Machine(id=4, shape='l_shape', width=4.5, height=3.5, 
                access_point=Point(1.0, -0.5),
                l_cutout_width=1.5, l_cutout_height=1.5),
        Machine(id=5, shape='l_shape', width=3.5, height=5.5, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=1, l_cutout_height=2),
        Machine(id=6, shape='l_shape', width=3.5, height=5.5, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=1, l_cutout_height=2),
        Machine(id=7, shape='l_shape', width=3.5, height=5.5, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=1, l_cutout_height=2),
        Machine(id=8, shape='rectangle', width=3.5, height=2.5, 
                access_point=Point(0, -1.0)),
    ]

    seq_A = [1,2,3,4,5,6,7,8]
    problems.append((machines_A, seq_A, Point(0,0), (-15,15,-15,15)))


    # ============================================================
    # PROBLEM SET B
    # ============================================================
    machines_B = [
        Machine(id=1, shape='rectangle', width=4.0, height=3.0, 
                access_point=Point(1.5, 0)),
        Machine(id=2, shape='l_shape', width=5.0, height=4.0, 
                access_point=Point(-1.0, 1.0),
                l_cutout_width=2.0, l_cutout_height=2.0),
        Machine(id=3, shape='rectangle', width=3.5, height=2.5, 
                access_point=Point(0, -1.0)),
        Machine(id=4, shape='l_shape', width=4.5, height=3.5, 
                access_point=Point(1.0, -0.5),
                l_cutout_width=1.5, l_cutout_height=1.5),
        Machine(id=5, shape='l_shape', width=3, height=5.5, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=1, l_cutout_height=2),
        Machine(id=6, shape='l_shape', width=5, height=3, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=2, l_cutout_height=2),
        Machine(id=7, shape='l_shape', width=3.5, height=2.5, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=1, l_cutout_height=2),
        Machine(id=8, shape='rectangle', width=3.5, height=2.5, 
                access_point=Point(0, 0))
    ]

    seq_B = [1,2,3,4,5,6,7,8]
    problems.append((machines_B, seq_B, Point(0,0), (-15,15,-15,15)))


    # ============================================================
    # PROBLEM SET C — More rectangles, wider machines
    # ============================================================
    machines_C = [
        Machine(id=1, shape="rectangle", width=6.0, height=3.0,
                access_point=Point(2.0, 0)),
        Machine(id=2, shape="rectangle", width=5.5, height=2.5,
                access_point=Point(-1.0, 0)),
        Machine(id=3, shape="rectangle", width=4.5, height=3.0,
                access_point=Point(0, -1.0)),
        Machine(id=4, shape="rectangle", width=3.5, height=4.0,
                access_point=Point(0, 1.5)),
        Machine(id=5, shape="rectangle", width=4.0, height=4.0,
                access_point=Point(1.0, -1.0)),
        Machine(id=6, shape="rectangle", width=5.0, height=3.5,
                access_point=Point(-1.5, 0)),
        Machine(id=7, shape="rectangle", width=6.0, height=2.5,
                access_point=Point(0, 1.0)),
        Machine(id=8, shape="rectangle", width=4.0, height=2.0,
                access_point=Point(0.5, 0)),
    ]

    seq_C = [1,2,3,4,5,6,7,8]
    problems.append((machines_C, seq_C, Point(0,0), (-15,15,-15,15)))


    # ============================================================
    # PROBLEM SET D — Small machines, mixed shapes
    # ============================================================
    machines_D = [
        Machine(id=1, shape="rectangle", width=2.5, height=2.5,
                access_point=Point(0.5, 0)),
        Machine(id=2, shape="l_shape", width=3.0, height=3.0,
                access_point=Point(1.0, -0.5),
                l_cutout_width=1.0, l_cutout_height=1.0),
        Machine(id=3, shape="rectangle", width=2.0, height=2.0,
                access_point=Point(0, 1.0)),
        Machine(id=4, shape="l_shape", width=3.5, height=3.5,
                access_point=Point(-0.5, 1.0),
                l_cutout_width=1.0, l_cutout_height=1.2),
        Machine(id=5, shape="rectangle", width=2.0, height=3.0,
                access_point=Point(0.5, -1.0)),
        Machine(id=6, shape="rectangle", width=3.0, height=2.0,
                access_point=Point(1.0, 0)),
        Machine(id=7, shape="l_shape", width=3.0, height=3.5,
                access_point=Point(1.0, 1.0),
                l_cutout_width=1.0, l_cutout_height=1.0),
        Machine(id=8, shape="rectangle", width=2.5, height=2.0,
                access_point=Point(0, 0)),
    ]
    
    seq_D = [1,2,3,4,5,6,7,8]
    problems.append((machines_D, seq_D, Point(0,0), (-15,15,-15,15)))


    # ============================================================
    # PROBLEM SET E — Big machines, heavy L-shapes
    # ============================================================
    machines_E = [
        Machine(id=1, shape="l_shape", width=6.0, height=6.0,
                access_point=Point(1.0, -1.0),
                l_cutout_width=2.0, l_cutout_height=2.0),
        Machine(id=2, shape="l_shape", width=5.5, height=5.0,
                access_point=Point(-1.0, 1.0),
                l_cutout_width=2.0, l_cutout_height=2.0),
        Machine(id=3, shape="rectangle", width=6.0, height=4.0,
                access_point=Point(0, -1.5)),
        Machine(id=4, shape="l_shape", width=6.5, height=5.5,
                access_point=Point(1.0, 1.0),
                l_cutout_width=2.5, l_cutout_height=2.0),
        Machine(id=5, shape="rectangle", width=5.0, height=5.0,
                access_point=Point(1.5, 0)),
        Machine(id=6, shape="l_shape", width=5.0, height=4.5,
                access_point=Point(0, -1.0),
                l_cutout_width=1.5, l_cutout_height=1.5),
        Machine(id=7, shape="rectangle", width=5.5, height=4.0,
                access_point=Point(-1, 0)),
        Machine(id=8, shape="rectangle", width=4.5, height=3.5,
                access_point=Point(1.0, 0)),
    ]

    seq_E = [1,2,3,4,5,6,7,8]
    problems.append((machines_E, seq_E, Point(0,0), (-15,15,-15,15)))


    # ============================================================
    # PROBLEM SET F — Vertical and slim machines
    # ============================================================
    machines_F = [
        Machine(id=1, shape="rectangle", width=2.0, height=6.0,
                access_point=Point(0.8, 0)),
        Machine(id=2, shape="rectangle", width=2.5, height=5.5,
                access_point=Point(-0.5, 1.0)),
        Machine(id=3, shape="l_shape", width=3.0, height=6.0,
                access_point=Point(0, -1.0),
                l_cutout_width=1.0, l_cutout_height=2.0),
        Machine(id=4, shape="rectangle", width=2.2, height=6.5,
                access_point=Point(1.0, 0)),
        Machine(id=5, shape="rectangle", width=1.8, height=5.0,
                access_point=Point(1.0, -1.0)),
        Machine(id=6, shape="l_shape", width=3.0, height=5.5,
                access_point=Point(-1.0, 1.0),
                l_cutout_width=1.0, l_cutout_height=1.5),
        Machine(id=7, shape="rectangle", width=2.5, height=6.0,
                access_point=Point(0, 1.0)),
        Machine(id=8, shape="l_shape", width=3.5, height=5.0,
                access_point=Point(0.5, 0),
                l_cutout_width=1.0, l_cutout_height=2.0),
    ]

    seq_F = [1,2,3,4,5,6,7,8]
    problems.append((machines_F, seq_F, Point(0,0), (-15,15,-15,15)))


    # ============================================================
    # PROBLEM SET G — Random mixed medium machines
    # ============================================================
    machines_G = [
        Machine(id=1, shape="rectangle", width=4.0, height=2.5,
                access_point=Point(1, 0)),
        Machine(id=2, shape="rectangle", width=5.0, height=3.5,
                access_point=Point(-1, 0)),
        Machine(id=3, shape="l_shape", width=4.0, height=4.0,
                access_point=Point(1, 1),
                l_cutout_width=1.5, l_cutout_height=1.0),
        Machine(id=4, shape="rectangle", width=3.0, height=3.0,
                access_point=Point(0, -1)),
        Machine(id=5, shape="l_shape", width=4.5, height=3.0,
                access_point=Point(1, -1),
                l_cutout_width=1.0, l_cutout_height=1.5),
        Machine(id=6, shape="rectangle", width=4.0, height=3.0,
                access_point=Point(-1, 1)),
        Machine(id=7, shape="rectangle", width=3.0, height=2.0,
                access_point=Point(0.5, 0)),
        Machine(id=8, shape="l_shape", width=4.0, height=3.0,
                access_point=Point(0, -1),
                l_cutout_width=1.0, l_cutout_height=1.0),
    ]

    seq_G = [1,2,3,4,5,6,7,8]
    problems.append((machines_G, seq_G, Point(0,0), (-15,15,-15,15)))


    # ============================================================
    # PROBLEM SET H — Very mixed shapes + rotating access point directions
    # ============================================================
    machines_H = [
        Machine(id=1, shape="rectangle", width=3.5, height=4.0,
                access_point=Point(1, 0)),
        Machine(id=2, shape="l_shape", width=4.5, height=4.0,
                access_point=Point(0, 1),
                l_cutout_width=1.5, l_cutout_height=1.5),
        Machine(id=3, shape="rectangle", width=5.0, height=3.0,
                access_point=Point(-1, 0)),
        Machine(id=4, shape="l_shape", width=5.0, height=5.0,
                access_point=Point(0, -1),
                l_cutout_width=2.0, l_cutout_height=1.5),
        Machine(id=5, shape="rectangle", width=3.0, height=3.5,
                access_point=Point(1, 1)),
        Machine(id=6, shape="rectangle", width=4.0, height=3.0,
                access_point=Point(-1, -1)),
        Machine(id=7, shape="l_shape", width=4.5, height=3.5,
                access_point=Point(1, 0),
                l_cutout_width=1.5, l_cutout_height=1.0),
        Machine(id=8, shape="rectangle", width=3.5, height=2.0,
                access_point=Point(0, -0.5)),
    ]
    
    # ============================================================
    # PROBLEM SET I — Symmetric rectangles (tests symmetry breaking)
    # ============================================================

    seq_H = [1,2,3,4,5,6,7,8]
    problems.append((machines_H, seq_H, Point(0,0), (-15,15,-15,15)))

    machines_I = [
        Machine(1, "rectangle", 4.0, 3.0, Point(1, 0)),
        Machine(2, "rectangle", 4.0, 3.0, Point(-1, 0)),
        Machine(3, "rectangle", 4.0, 3.0, Point(0, 1)),
        Machine(4, "rectangle", 4.0, 3.0, Point(0, -1)),
        Machine(5, "rectangle", 3.5, 2.5, Point(1, 1)),
        Machine(6, "rectangle", 3.5, 2.5, Point(-1, 1)),
        Machine(7, "rectangle", 3.5, 2.5, Point(1, -1)),
        Machine(8, "rectangle", 3.5, 2.5, Point(-1, -1)),
    ]

    seq_I = [1,2,3,4,5,6,7,8]
    problems.append((machines_I, seq_I, Point(0,0), (-15,15,-15,15)))
    
    # ============================================================
    # PROBLEM SET J — Clustered access points (path-dominated)
    # ============================================================
    
    machines_J = [
        Machine(1, "rectangle", 4.5, 3.0, Point(1.2, 0.8)),
        Machine(2, "rectangle", 4.0, 3.5, Point(1.0, 0.9)),
        Machine(3, "l_shape", 5.0, 4.0, Point(1.1, 1.0)),
        Machine(4, "rectangle", 3.5, 3.0, Point(0.9, 1.1)),
        Machine(5, "l_shape", 4.5, 3.5, Point(1.0, 0.7)),
        Machine(6, "rectangle", 3.0, 2.5, Point(1.2, 1.2)),
        Machine(7, "rectangle", 3.0, 3.0, Point(0.8, 0.9)),
        Machine(8, "rectangle", 3.5, 2.5, Point(1.1, 0.6)),
    ]

    seq_J = [1,2,3,4,5,6,7,8]
    problems.append((machines_J, seq_J, Point(0,0), (-15,15,-15,15)))
    
    # ============================================================
    # PROBLEM SET K  — Narrow corridor packing
    # ============================================================

    machines_K = [
        Machine(1, "rectangle", 2.0, 6.0, Point(0, 1)),
        Machine(2, "rectangle", 2.2, 5.8, Point(0, -1)),
        Machine(3, "rectangle", 2.0, 6.2, Point(0.5, 0)),
        Machine(4, "rectangle", 2.3, 5.5, Point(-0.5, 0)),
        Machine(5, "l_shape", 3.0, 6.0, Point(0, 1)),
        Machine(6, "rectangle", 2.0, 5.0, Point(0, -1)),
        Machine(7, "rectangle", 2.1, 6.1, Point(0.3, 0)),
        Machine(8, "rectangle", 2.4, 5.6, Point(-0.3, 0)),
    ]

    seq_K = [1,2,3,4,5,6,7,8]
    problems.append((machines_K, seq_K, Point(0,0), (-15,15,-15,15)))

    # ============================================================
    # PROBLEM SET L — Large machines near workspace limits
    # ============================================================

    machines_L = [
        Machine(1, "rectangle", 6.5, 4.0, Point(1.5, 0)),
        Machine(2, "rectangle", 6.0, 4.5, Point(-1.5, 0)),
        Machine(3, "l_shape", 6.0, 5.0, Point(0, 1)),
        Machine(4, "rectangle", 5.5, 4.0, Point(0, -1)),
        Machine(5, "l_shape", 5.5, 5.5, Point(1, 1)),
        Machine(6, "rectangle", 5.0, 4.5, Point(-1, 1)),
        Machine(7, "rectangle", 5.8, 3.8, Point(1, -1)),
        Machine(8, "rectangle", 5.2, 4.2, Point(-1, -1)),
    ]
    seq_L = [1,2,3,4,5,6,7,8]
    problems.append((machines_L, seq_L, Point(0,0), (-15,15,-15,15)))

    # ============================================================
    # PROBLEM SET M — Alternating small / large machines
    # ============================================================

    machines_M = [
        Machine(1, "rectangle", 2.0, 2.0, Point(0.5, 0)),
        Machine(2, "rectangle", 6.0, 4.0, Point(-1.5, 0)),
        Machine(3, "rectangle", 2.2, 2.5, Point(0, 0.5)),
        Machine(4, "l_shape", 6.0, 5.0, Point(1, 1)),
        Machine(5, "rectangle", 2.0, 3.0, Point(0, -0.5)),
        Machine(6, "rectangle", 5.5, 4.5, Point(-1, 1)),
        Machine(7, "rectangle", 2.5, 2.0, Point(1, 0)),
        Machine(8, "l_shape", 5.5, 4.0, Point(0, -1)),
    ]
    seq_M = [1,2,3,4,5,6,7,8]
    problems.append((machines_M, seq_M, Point(0,0), (-15,15,-15,15)))

    # ============================================================
    # PROBLEM SET N — L-shape dominant, irregular cutouts
    # ============================================================

    machines_N = [
        Machine(1, "l_shape", 4.5, 5.5, Point(1, 0)),
        Machine(2, "l_shape", 5.0, 4.5, Point(-1, 1)),
        Machine(3, "l_shape", 4.0, 5.0, Point(0, -1)),
        Machine(4, "l_shape", 5.5, 4.0, Point(1, -1)),
        Machine(5, "rectangle", 3.0, 3.0, Point(0.5, 0)),
        Machine(6, "rectangle", 3.5, 2.5, Point(-0.5, 0)),
        Machine(7, "rectangle", 3.0, 2.0, Point(0, 1)),
        Machine(8, "rectangle", 2.5, 3.0, Point(0, -1)),
    ]
    seq_N = [1,2,3,4,5,6,7,8]
    problems.append((machines_N, seq_N, Point(0,0), (-15,15,-15,15)))
    
    # ============================================================
    # PROBLEM SET O - Asymmetric layout + skewed access points
    # ============================================================

    machines_O = [
        Machine(1, "rectangle", 4.5, 3.0, Point(1.8, 0.2)),
        Machine(2, "rectangle", 4.0, 3.5, Point(1.6, -0.4)),
        Machine(3, "l_shape", 5.0, 4.0, Point(1.4, 1.0)),
        Machine(4, "rectangle", 3.5, 3.0, Point(1.2, -1.0)),
        Machine(5, "l_shape", 4.0, 3.5, Point(1.0, 0.6)),
        Machine(6, "rectangle", 3.0, 2.5, Point(0.8, -0.8)),
        Machine(7, "rectangle", 3.5, 2.8, Point(1.3, 0.4)),
        Machine(8, "rectangle", 3.0, 3.0, Point(0.9, 1.2)),
    ]
    seq_O = [1,2,3,4,5,6,7,8]
    problems.append((machines_O, seq_O, Point(0,0), (-15,15,-15,15)))

    return problems


# ============================================================
# 2. Training loop: run GA on each problem for several epochs
# ============================================================

def train_agent(
    num_epochs: int = 3,
    q_table_path: str = "rl_ga_q_table_3.npy",
    state_size: int = 9,
    action_size: int = 8,
) -> QLearningAgent:
    """
    Train a single QLearningAgent across multiple problems and save its Q-table.

    - num_epochs: how many passes over the whole problem set
    - q_table_path: output .npy file path for the Q-table
    """
    problems = build_training_problems()
    if not problems:
        raise ValueError(
            "No training problems defined. "
            "Fill in build_training_problems() with your machine layouts."
        )

    # Shared RL agent across all problems
    agent = QLearningAgent(state_size=state_size, action_size=action_size)

    print(f"Starting training on {len(problems)} problems for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        for idx, (machines, sequence, robot_position, workspace_bounds) in enumerate(problems):
            print(f"\n  -> Problem {idx + 1}/{len(problems)}")

            # IMPORTANT:
            # RLGeneticAlgorithm should accept rl_agent as an optional argument:
            # def __init__(..., rl_agent: Optional[QLearningAgent] = None)
            ga = RLGeneticAlgorithm(
                machines=machines,
                sequence=sequence,
                robot_position=robot_position,
                workspace_bounds=workspace_bounds,
                rl_agent=agent,    # reuse same agent
            )

            # If your RLGeneticAlgorithm.optimize() takes extra args (e.g. num_generations),
            # pass them here.
            final_layout, best_fitness, results = ga.optimize()

            print(
                f"     Finished GA run. Best fitness: {best_fitness:.4f}, "
                f"total_distance: {results.get('total_distance', float('nan')):.4f}"
            )

    # Save the learned Q-table
    np.save(q_table_path, agent.q_table)
    print(f"\nTraining complete. Saved Q-table to: {q_table_path}")

    return agent


# ============================================================
# 3. Helper: load a trained agent
# ============================================================

def load_trained_agent(
    q_table_path: str = "rl_ga_q_table_3.npy",
    state_size: int = 9,
    action_size: int = 8,
    epsilon: float = 0.0,
) -> QLearningAgent:
    """
    Load a trained QLearningAgent from a saved Q-table.

    - epsilon: set to 0.0 for fully greedy behaviour (no exploration),
               or a small value (e.g. 0.05) if you still want some exploration.
    """
    q_table = np.load(q_table_path)
    agent = QLearningAgent(state_size=state_size, action_size=action_size)
    agent.q_table = q_table
    agent.epsilon = epsilon
    return agent


# ============================================================
# 4. (Optional) Quick demo on a NEW problem using a trained agent
# ============================================================

def solve_new_problem_with_trained_agent(
    machines: List[Machine],
    sequence: List[int],
    robot_position: Point,
    workspace_bounds: Tuple[float, float, float, float],
    q_table_path: str = "rl_ga_q_table_3.npy",
):
    """
    Example of how to use a trained agent on a fresh machine layout.
    """
    agent = load_trained_agent(q_table_path=q_table_path, epsilon=0.0)

    ga = RLGeneticAlgorithm(
        machines=machines,
        sequence=sequence,
        robot_position=robot_position,
        workspace_bounds=workspace_bounds,
        rl_agent=agent,
    )

    final_layout, best_fitness, results = ga.optimize()
    print("Solved new problem with trained agent.")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Total distance: {results.get('total_distance', float('nan')):.4f}")
    return final_layout, best_fitness, results


# ============================================================
# 5. Main
# ============================================================

if __name__ == "__main__":
    # Adjust num_epochs and q_table_path as you wish
    train_agent(num_epochs=3, q_table_path="rl_ga_q_table.npy")