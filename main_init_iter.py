from func.datastruct import Point, Machine
from algo.gen_algo_init import GeneticAlgorithm
from algo.agent import QLearningAgent
from algo.gen_algo_rl_init import RLGeneticAlgorithm
import numpy as np
import random
import os


def generate_problem(problem_idx: int):
    """
    Generate one problem instance with 8 machines.
    We vary machine sizes, shapes and access points per problem.
    """
    # For reproducibility of problem definitions
    base_seed = 1000 + problem_idx
    random.seed(base_seed)
    np.random.seed(base_seed)

    workspace_bounds = (-15, 15, -15, 15)
    robot_position = Point(0, 0)

    machines = []
    for i in range(8):
        m_id = i + 1

        # Alternate shapes for variety
        shape = "l_shape" if i % 2 == 0 else "rectangle"

        # Random-ish but reasonable machine sizes
        width = random.uniform(3.0, 6.0)
        height = random.uniform(2.5, 5.0)

        # Access point somewhere near machine center
        ap_x = random.uniform(-1.5, 1.5)
        ap_y = random.uniform(-1.5, 1.5)

        if shape == "l_shape":
            # L-cutout smaller than total width/height
            l_cutout_width = random.uniform(1.0, width / 2.0)
            l_cutout_height = random.uniform(1.0, height / 2.0)
            machine = Machine(
                id=m_id,
                shape="l_shape",
                width=width,
                height=height,
                access_point=Point(ap_x, ap_y),
                l_cutout_width=l_cutout_width,
                l_cutout_height=l_cutout_height,
            )
        else:
            machine = Machine(
                id=m_id,
                shape="rectangle",
                width=width,
                height=height,
                access_point=Point(ap_x, ap_y),
            )

        machines.append(machine)

    # Simple sequence: visit machines 1..8
    sequence = list(range(1, 9))

    return machines, sequence, robot_position, workspace_bounds


def main():
    # =========================
    # Config
    # =========================
    runs_per_problem = 3
    num_problems = 10

    BASE_RESULTS_DIR = "results_500_300_8"
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

    # =========================
    # Load trained agent (base weights)
    # =========================
    base_agent = QLearningAgent(state_size=9, action_size=8)
    base_agent.q_table = np.load("rl_agent/rl_ga_q_table_500_300_8.npy")
    base_agent.epsilon = 0.05  # low exploration during evaluation

    all_problems_results = {}

    for problem_idx in range(num_problems):
        # -------------------------
        # Per-problem folder
        # -------------------------
        problem_id = problem_idx + 1
        problem_dir = os.path.join(BASE_RESULTS_DIR, f"problem_{problem_id}")
        os.makedirs(problem_dir, exist_ok=True)

        print("\n" + "#" * 80)
        print(f"PROBLEM {problem_id}/{num_problems}")
        print("#" * 80)

        # -------------------------
        # Generate problem instance
        # -------------------------
        machines, sequence, robot_position, workspace_bounds = generate_problem(problem_idx)

        print(f"=== MACHINE DATA : PROBLEM {problem_id} ===")
        print(f"Workspace bounds: {workspace_bounds}")
        print(f"Sequence: {sequence}\n")

        for m in machines:
            if m.shape == "l_shape":
                print(
                    f"Machine {m.id} | Shape={m.shape} | "
                    f"Width={m.width:.3f}, Height={m.height:.3f} | "
                    f"AccessPoint=({m.access_point.x:.3f}, {m.access_point.y:.3f}) | "
                    f"L_cutout=({m.l_cutout_width:.3f}, {m.l_cutout_height:.3f})"
                )
            else:
                print(
                    f"Machine {m.id} | Shape={m.shape} | "
                    f"Width={m.width:.3f}, Height={m.height:.3f} | "
                    f"AccessPoint=({m.access_point.x:.3f}, {m.access_point.y:.3f})"
                )

        print("\n=== Robotic Workcell Layout Optimization ===")
        print(f"Problem ID: {problem_id}")
        print(f"Machines: {len(machines)}")
        print(f"Sequence: {sequence}")
        print(f"Workspace: {workspace_bounds}\n")

        optimizer_list = [GeneticAlgorithm, RLGeneticAlgorithm]

        # -------------------------
        # Per-problem results store
        # -------------------------
        comparison_results = {
            "GeneticAlgorithm": {
                "best_fitness": [],
                "execution_time": [],
                "total_distance": [],
                "final_machines": [],
            },
            "RLGeneticAlgorithm": {
                "best_fitness": [],
                "execution_time": [],
                "total_distance": [],
                "final_machines": [],
            },
        }

        # -------------------------
        # Runs for this problem
        # -------------------------
        for run in range(runs_per_problem):
            run_id = run + 1

            print(f"\n{'=' * 70}")
            print(f"Problem {problem_id} – Run {run_id}/{runs_per_problem}")
            print(f"{'=' * 70}\n")

            # Per-run seeds for reproducibility
            seed = 42 + problem_idx * 100 + run
            random.seed(seed)
            np.random.seed(seed)

            # Base GA used ONLY to create a shared initial population
            base_ga = GeneticAlgorithm(
                machines=machines,
                sequence=sequence,
                robot_position=robot_position,
                workspace_bounds=workspace_bounds,
            )
            initial_population = base_ga.create_initial_population()

            # Evaluate both GA and RL-GA using the same initial population
            for OptimizerClass in optimizer_list:
                optimizer_name = OptimizerClass.__name__
                print(f"\n--- {optimizer_name} ---")

                if optimizer_name == "RLGeneticAlgorithm":
                    # Fresh RL agent for every RL-GA run
                    rl_agent = QLearningAgent(state_size=9, action_size=8)
                    rl_agent.q_table = base_agent.q_table.copy()
                    rl_agent.epsilon = base_agent.epsilon

                    optimizer = OptimizerClass(
                        machines=machines,
                        sequence=sequence,
                        robot_position=robot_position,
                        workspace_bounds=workspace_bounds,
                        rl_agent=rl_agent,
                    )
                else:
                    optimizer = OptimizerClass(
                        machines=machines,
                        sequence=sequence,
                        robot_position=robot_position,
                        workspace_bounds=workspace_bounds,
                    )

                optimized_machines, best_fitness, results = optimizer.optimize(
                    initial_population=initial_population
                )

                # Store results
                comparison_results[optimizer_name]["best_fitness"].append(best_fitness)
                comparison_results[optimizer_name]["execution_time"].append(results["execution_time"])
                comparison_results[optimizer_name]["total_distance"].append(results["total_distance"])
                comparison_results[optimizer_name]["final_machines"].append(optimized_machines)

                # Display individual run results
                print("\n=== RUN RESULTS ===")
                print(f"Best Fitness: {best_fitness:.2f}")
                print(f"Total Distance: {results['total_distance']:.2f} units")
                print(f"Execution Time: {results['execution_time']:.2f} seconds")
                print(f"Generations: {results['generations']}")

        # =========================
        # Per-problem statistics
        # =========================
        print("\n" + "=" * 80)
        print(f"COMPARISON SUMMARY – PROBLEM {problem_id}")
        print("=" * 80)

        summary_path = os.path.join(problem_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Problem {problem_id} summary\n")
            f.write(f"Sequence: {sequence}\n")
            f.write(f"Workspace: {workspace_bounds}\n\n")

            for optimizer_name in comparison_results.keys():
                r = comparison_results[optimizer_name]

                avg_fitness = float(np.mean(r["best_fitness"]))
                std_fitness = float(np.std(r["best_fitness"]))
                min_fitness = float(np.min(r["best_fitness"]))
                max_fitness = float(np.max(r["best_fitness"]))

                avg_time = float(np.mean(r["execution_time"]))
                std_time = float(np.std(r["execution_time"]))

                avg_distance = float(np.mean(r["total_distance"]))
                std_distance = float(np.std(r["total_distance"]))

                print(f"\n{optimizer_name}:")
                print(f"  Best Fitness: {avg_fitness:.2f} ± {std_fitness:.2f} (min={min_fitness:.2f}, max={max_fitness:.2f})")
                print(f"  Execution Time: {avg_time:.2f} ± {std_time:.2f} seconds")
                print(f"  Total Distance: {avg_distance:.2f} ± {std_distance:.2f} units")

                f.write(f"{optimizer_name}:\n")
                f.write(f"  Best Fitness: {avg_fitness:.4f} ± {std_fitness:.4f} (min={min_fitness:.4f}, max={max_fitness:.4f})\n")
                f.write(f"  Execution Time (s): {avg_time:.4f} ± {std_time:.4f}\n")
                f.write(f"  Total Distance: {avg_distance:.4f} ± {std_distance:.4f}\n\n")

        # =========================
        # Performance comparison
        # =========================
        ga_avg_fitness = float(np.mean(comparison_results["GeneticAlgorithm"]["best_fitness"]))
        rl_avg_fitness = float(np.mean(comparison_results["RLGeneticAlgorithm"]["best_fitness"]))

        ga_avg_time = float(np.mean(comparison_results["GeneticAlgorithm"]["execution_time"]))
        rl_avg_time = float(np.mean(comparison_results["RLGeneticAlgorithm"]["execution_time"]))

        fitness_improvement = ((ga_avg_fitness - rl_avg_fitness) / ga_avg_fitness) * 100.0 if ga_avg_fitness != 0 else 0.0
        time_difference = ((rl_avg_time - ga_avg_time) / ga_avg_time) * 100.0 if ga_avg_time != 0 else 0.0

        print("\n" + "=" * 80)
        print(f"PERFORMANCE COMPARISON – PROBLEM {problem_id}")
        print("=" * 80)

        print(f"\nFitness Improvement: {fitness_improvement:+.2f}%")
        print(f"Time Difference: {time_difference:+.2f}%")

        # Best solutions
        best_ga_idx = int(np.argmin(comparison_results["GeneticAlgorithm"]["best_fitness"]))
        best_rl_idx = int(np.argmin(comparison_results["RLGeneticAlgorithm"]["best_fitness"]))

        best_ga_fitness = comparison_results["GeneticAlgorithm"]["best_fitness"][best_ga_idx]
        best_rl_fitness = comparison_results["RLGeneticAlgorithm"]["best_fitness"][best_rl_idx]

        print(f"\nBest Solution Found in Problem {problem_id}:")
        if best_rl_fitness < best_ga_fitness:
            print(f"  → RL-GA (Run {best_rl_idx + 1}): Fitness = {best_rl_fitness:.2f}")
            best_machines = comparison_results["RLGeneticAlgorithm"]["final_machines"][best_rl_idx]
        else:
            print(f"  → Standard GA (Run {best_ga_idx + 1}): Fitness = {best_ga_fitness:.2f}")
            best_machines = comparison_results["GeneticAlgorithm"]["final_machines"][best_ga_idx]

        print("\nBest Machine Positions:")
        for machine in best_machines:
            print(
                f"Machine {machine.id}: "
                f"Position=({machine.position.x:.2f}, {machine.position.y:.2f}), "
                f"Rotation={machine.rotation:.1f}°"
            )

        # =========================
        # Save NPZ (inside problem_dir)
        # =========================
        np.savez(
            os.path.join(problem_dir, "comparison_results.npz"),
            ga_best_fitness=np.array(comparison_results["GeneticAlgorithm"]["best_fitness"]),
            rl_best_fitness=np.array(comparison_results["RLGeneticAlgorithm"]["best_fitness"]),
            ga_execution_time=np.array(comparison_results["GeneticAlgorithm"]["execution_time"]),
            rl_execution_time=np.array(comparison_results["RLGeneticAlgorithm"]["execution_time"]),
            ga_total_distance=np.array(comparison_results["GeneticAlgorithm"]["total_distance"]),
            rl_total_distance=np.array(comparison_results["RLGeneticAlgorithm"]["total_distance"]),
        )

        # =========================
        # Plot comparison (inside problem_dir)
        # =========================
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].bar(
                ["GA", "RL-GA"],
                [ga_avg_fitness, rl_avg_fitness],
                yerr=[
                    np.std(comparison_results["GeneticAlgorithm"]["best_fitness"]),
                    np.std(comparison_results["RLGeneticAlgorithm"]["best_fitness"]),
                ],
                capsize=5,
                alpha=0.7,
            )
            axes[0].set_ylabel("Average Best Fitness")
            axes[0].set_title(f"Fitness Comparison (Problem {problem_id}) – Lower is Better")
            axes[0].grid(axis="y", alpha=0.3)

            axes[1].bar(
                ["GA", "RL-GA"],
                [ga_avg_time, rl_avg_time],
                yerr=[
                    np.std(comparison_results["GeneticAlgorithm"]["execution_time"]),
                    np.std(comparison_results["RLGeneticAlgorithm"]["execution_time"]),
                ],
                capsize=5,
                alpha=0.7,
            )
            axes[1].set_ylabel("Average Execution Time (seconds)")
            axes[1].set_title(f"Execution Time (Problem {problem_id})")
            axes[1].grid(axis="y", alpha=0.3)

            plt.tight_layout()
            fig_path = os.path.join(problem_dir, "comparison.png")
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"\nComparison chart saved as '{fig_path}'")
            plt.close(fig)

        except ImportError:
            print("\nMatplotlib not available for visualization")

        # Store per-problem results in global dict
        all_problems_results[f"problem_{problem_id}"] = comparison_results

    print(f"\nAll results saved under: {os.path.abspath(BASE_RESULTS_DIR)}")
    return all_problems_results



if __name__ == "__main__":
    all_results = main()