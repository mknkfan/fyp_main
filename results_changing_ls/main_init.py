from func.datastruct import Point, Machine
from algo.gen_algo_init import GeneticAlgorithm
from algo.agent import QLearningAgent
from algo.gen_algo_rl_init import RLGeneticAlgorithm
import numpy as np
import random
from visual.visualize import visualize_layout, visualize_layout_generation, create_layout_animation

def plot_fitness_convergence(comparison_results, filename="fitness_convergence.png"):
    import matplotlib.pyplot as plt

    # Helper: stack histories and pad to same length (just in case)
    def stack_histories(hist_list):
        max_len = max(len(h) for h in hist_list)
        arr = np.full((len(hist_list), max_len), np.nan, dtype=float)
        for i, h in enumerate(hist_list):
            arr[i, :len(h)] = np.array(h, dtype=float)
        return arr

    ga_best_runs = comparison_results["GeneticAlgorithm"]["best_fitness_history_per_run"]
    ga_avg_runs  = comparison_results["GeneticAlgorithm"]["avg_fitness_history_per_run"]
    rl_best_runs = comparison_results["RLGeneticAlgorithm"]["best_fitness_history_per_run"]
    rl_avg_runs  = comparison_results["RLGeneticAlgorithm"]["avg_fitness_history_per_run"]

    if len(ga_best_runs) == 0 or len(rl_best_runs) == 0:
        print("No fitness history stored. Did optimize() return best_fitness_history/avg_fitness_history?")
        return

    # Stack
    ga_best = stack_histories(ga_best_runs)
    ga_avg  = stack_histories(ga_avg_runs)
    rl_best = stack_histories(rl_best_runs)
    rl_avg  = stack_histories(rl_avg_runs)

    # Mean/std ignoring NaNs
    ga_best_mean = np.nanmean(ga_best, axis=0)
    ga_best_std  = np.nanstd(ga_best, axis=0)
    ga_avg_mean  = np.nanmean(ga_avg, axis=0)
    ga_avg_std   = np.nanstd(ga_avg, axis=0)

    rl_best_mean = np.nanmean(rl_best, axis=0)
    rl_best_std  = np.nanstd(rl_best, axis=0)
    rl_avg_mean  = np.nanmean(rl_avg, axis=0)
    rl_avg_std   = np.nanstd(rl_avg, axis=0)

    x = np.arange(len(ga_best_mean))

    plt.figure(figsize=(12, 6))

    # Best fitness curves
    plt.plot(x, ga_best_mean, label="GA Best", linewidth=2)
    plt.plot(x, rl_best_mean, label="RL-GA Best", linewidth=2)

    # Shaded std bands (useful when runs > 1)
    plt.fill_between(x, ga_best_mean - ga_best_std, ga_best_mean + ga_best_std, alpha=0.15)
    plt.fill_between(x, rl_best_mean - rl_best_std, rl_best_mean + rl_best_std, alpha=0.15)

    # Avg fitness curves (dashed)
    plt.plot(x, ga_avg_mean, label="GA Avg", linestyle="--", linewidth=1.8)
    plt.plot(x, rl_avg_mean, label="RL-GA Avg", linestyle="--", linewidth=1.8)

    plt.xlabel("Generation")
    plt.ylabel("Fitness (lower is better)")
    plt.title("Fitness Convergence: GA vs RL-GA")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\nFitness convergence plot saved as '{filename}'")
    plt.show()


def main():    
    # machines = [
    #     Machine(id=1, shape='l_shape', width=5.21, height=4.33,
    #             access_point=Point(1.48, -1.12),
    #             l_cutout_width=2.01, l_cutout_height=1.72),

    #     Machine(id=2, shape='rectangle', width=4.87, height=3.12,
    #             access_point=Point(0.92, -0.44)),

    #     Machine(id=3, shape='l_shape', width=4.56, height=5.46,
    #             access_point=Point(-0.72, 1.89),
    #             l_cutout_width=1.58, l_cutout_height=2.74),

    #     Machine(id=4, shape='rectangle', width=3.44, height=2.61,
    #             access_point=Point(0.13, 0.89)),

    #     Machine(id=5, shape='l_shape', width=5.94, height=3.77,
    #             access_point=Point(1.04, -0.88),
    #             l_cutout_width=2.23, l_cutout_height=1.32),

    #     Machine(id=6, shape='rectangle', width=4.32, height=2.48,
    #             access_point=Point(-0.63, -0.41)),

    #     Machine(id=7, shape='l_shape', width=3.88, height=4.91,
    #             access_point=Point(0.55, 1.22),
    #             l_cutout_width=1.25, l_cutout_height=2.01),

    #     Machine(id=8, shape='rectangle', width=3.27, height=2.74,
    #             access_point=Point(0.15, -0.52)),
    # ]
    machines = [
        Machine(id=1, shape='l_shape', width=4.492, height=3.400,
                access_point=Point(0.666, -0.160),
                l_cutout_width=1.565, l_cutout_height=1.018),

        Machine(id=2, shape='rectangle', width=4.005, height=3.689,
                access_point=Point(-0.146, -1.236)),

        Machine(id=3, shape='l_shape', width=3.885, height=3.347,
                access_point=Point(1.206, 0.652),
                l_cutout_width=1.847, l_cutout_height=1.022),

        Machine(id=4, shape='rectangle', width=3.685, height=3.845,
                access_point=Point(-1.316, -0.812)),

        Machine(id=5, shape='l_shape', width=4.679, height=2.502,
                access_point=Point(-0.324, -1.406),
                l_cutout_width=1.322, l_cutout_height=1.225),

        Machine(id=6, shape='rectangle', width=4.160, height=3.170,
                access_point=Point(1.234, 1.069)),

        Machine(id=7, shape='l_shape', width=3.765, height=3.568,
                access_point=Point(-0.908, 0.793),
                l_cutout_width=1.401, l_cutout_height=1.604),

        Machine(id=8, shape='rectangle', width=5.263, height=2.832,
                access_point=Point(-1.092, 0.727)),
    ]

    sequence = [1, 2, 3, 4, 5, 6, 7, 8]
    robot_position = Point(0, 0)
    workspace_bounds = (-15, 15, -15, 15)
    
    # LOAD TRAINED AGENT
    trained_agent = QLearningAgent(state_size=9, action_size=8)
    trained_agent.q_table = np.load("rl_ga_q_table_200_300_15.npy") 
    trained_agent.epsilon = 0.05

    print("=== Robotic Workcell Layout Optimization ===")
    print(f"Machines: {len(machines)}")
    print(f"Sequence: {sequence}")
    print(f"Workspace: {workspace_bounds}")
    print()
    
    runs = 1
    optimizer_list = [GeneticAlgorithm, RLGeneticAlgorithm]
    
    comparison_results = {
        'GeneticAlgorithm': {
            'best_fitness': [],
            'execution_time': [],
            'total_distance': [],
            'final_machines': [],
            'results_per_run': [],
            # ✅ add histories
            'best_fitness_history_per_run': [],
            'avg_fitness_history_per_run': [],
        },
        'RLGeneticAlgorithm': {
            'best_fitness': [],
            'execution_time': [],
            'total_distance': [],
            'final_machines': [],
            'results_per_run': [],
            # ✅ add histories
            'best_fitness_history_per_run': [],
            'avg_fitness_history_per_run': [],
        }
    }
    
    for run in range(runs):
        print(f"\n{'='*70}")
        print(f"Run {run + 1}/{runs}")
        print(f"{'='*70}\n")

        problem_idx = 3
        # base_seed = 1000 + problem_idx
        # random.seed(base_seed)
        # np.random.seed(base_seed)

        seed = 42 + problem_idx * 100 + run
        random.seed(seed)
        np.random.seed(seed)
        # Base GA used only to create a shared initial population (fairness)
        base_ga = GeneticAlgorithm(
            machines=machines,
            sequence=sequence,
            robot_position=robot_position,
            workspace_bounds=workspace_bounds,
        )
        initial_population = base_ga.create_initial_population()

        for OptimizerClass in optimizer_list:
            optimizer_name = OptimizerClass.__name__
            print(f"\n--- {optimizer_name} ---")

            if optimizer_name == "RLGeneticAlgorithm":
                optimizer = OptimizerClass(
                    machines=machines,
                    sequence=sequence,
                    robot_position=robot_position,
                    workspace_bounds=workspace_bounds,
                    rl_agent=trained_agent
                )
            else:
                optimizer = OptimizerClass(
                    machines=machines,
                    sequence=sequence,
                    robot_position=robot_position,
                    workspace_bounds=workspace_bounds
                )
            
            optimized_machines, best_fitness, results = optimizer.optimize(
                initial_population=initial_population
            )
            
            # Store scalar results
            comparison_results[optimizer_name]['best_fitness'].append(best_fitness)
            comparison_results[optimizer_name]['execution_time'].append(results['execution_time'])
            comparison_results[optimizer_name]['total_distance'].append(results['total_distance'])
            comparison_results[optimizer_name]['final_machines'].append(optimized_machines)
            comparison_results[optimizer_name]['results_per_run'].append(results)

            # ✅ Store fitness histories (for convergence plot)
            comparison_results[optimizer_name]['best_fitness_history_per_run'].append(
                results['best_fitness_history']
            )
            comparison_results[optimizer_name]['avg_fitness_history_per_run'].append(
                results['avg_fitness_history']
            )
            
            print("\n=== RUN RESULTS ===")
            print(f"Best Fitness: {best_fitness:.2f}")
            print(f"Total Distance: {results['total_distance']:.2f} units")
            print(f"Execution Time: {results['execution_time']:.2f} seconds")
            print(f"Generations: {results['generations']}")

                    
            # Create animation snapshots every 20 generations (per optimizer)
            decoded_evolution = results.get("decoded_evolution", [])
            if decoded_evolution:
                create_layout_animation(
                    decoded_evolution=decoded_evolution,
                    robot_position=robot_position,
                    sequence=sequence,
                    workspace_bounds=workspace_bounds,
                    snapshot_interval=results.get("snapshot_interval", 20),
                    output_filename=f"layout_evolution_{optimizer_name.lower()}.gif",
                    duration=1.0,
                )
            else:
                print(f"No decoded evolution data for {optimizer_name}; skipping GIF.")

            visualize_layout(
                machines=optimized_machines,
                robot_position=robot_position,
                sequence=sequence,
                workspace_bounds=workspace_bounds
            )
    
    # Summary stats
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    for optimizer_name in comparison_results.keys():
        results = comparison_results[optimizer_name]
        
        avg_fitness = np.mean(results['best_fitness'])
        std_fitness = np.std(results['best_fitness'])
        min_fitness = np.min(results['best_fitness'])
        max_fitness = np.max(results['best_fitness'])
        
        avg_time = np.mean(results['execution_time'])
        std_time = np.std(results['execution_time'])
        
        avg_distance = np.mean(results['total_distance'])
        std_distance = np.std(results['total_distance'])
        
        print(f"\n{optimizer_name}:")
        print(f"  Best Fitness:")
        print(f"    Average: {avg_fitness:.2f} ± {std_fitness:.2f}")
        print(f"    Min: {min_fitness:.2f}, Max: {max_fitness:.2f}")
        print(f"  Execution Time:")
        print(f"    Average: {avg_time:.2f} ± {std_time:.2f} seconds")
        print(f"  Total Distance:")
        print(f"    Average: {avg_distance:.2f} ± {std_distance:.2f} units")
    
    # Performance comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    ga_avg_fitness = np.mean(comparison_results['GeneticAlgorithm']['best_fitness'])
    rl_avg_fitness = np.mean(comparison_results['RLGeneticAlgorithm']['best_fitness'])
    
    ga_avg_time = np.mean(comparison_results['GeneticAlgorithm']['execution_time'])
    rl_avg_time = np.mean(comparison_results['RLGeneticAlgorithm']['execution_time'])
    
    fitness_improvement = ((ga_avg_fitness - rl_avg_fitness) / ga_avg_fitness) * 100
    time_difference = ((rl_avg_time - ga_avg_time) / ga_avg_time) * 100
    
    print(f"\nFitness Improvement: {fitness_improvement:+.2f}%")
    if fitness_improvement > 0:
        print(f"  → RL-GA achieved {fitness_improvement:.2f}% better fitness")
    else:
        print(f"  → Standard GA achieved {abs(fitness_improvement):.2f}% better fitness")
    
    print(f"\nTime Difference: {time_difference:+.2f}%")
    if time_difference > 0:
        print(f"  → RL-GA took {time_difference:.2f}% longer")
    else:
        print(f"  → RL-GA was {abs(time_difference):.2f}% faster")
    
    # Best overall solution
    best_ga_idx = np.argmin(comparison_results['GeneticAlgorithm']['best_fitness'])
    best_rl_idx = np.argmin(comparison_results['RLGeneticAlgorithm']['best_fitness'])
    
    best_ga_fitness = comparison_results['GeneticAlgorithm']['best_fitness'][best_ga_idx]
    best_rl_fitness = comparison_results['RLGeneticAlgorithm']['best_fitness'][best_rl_idx]
    
    print(f"\nBest Solution Found:")
    if best_rl_fitness < best_ga_fitness:
        print(f"  → RL-GA (Run {best_rl_idx + 1}): Fitness = {best_rl_fitness:.2f}")
        best_machines = comparison_results['RLGeneticAlgorithm']['final_machines'][best_rl_idx]
        best_name = "RLGeneticAlgorithm"
        best_idx = best_rl_idx
    else:
        print(f"  → Standard GA (Run {best_ga_idx + 1}): Fitness = {best_ga_fitness:.2f}")
        best_name = "GeneticAlgorithm"
        best_machines = comparison_results['GeneticAlgorithm']['final_machines'][best_ga_idx]
        best_idx = best_ga_idx

        best_machines = comparison_results[best_name]['final_machines'][best_idx]
        best_results  = comparison_results[best_name]['results_per_run'][best_idx]

    for machine in best_machines:
        print(f"Machine {machine.id}: "
              f"Position=({machine.position.x:.2f}, {machine.position.y:.2f}), "
              f"Rotation={machine.rotation:.1f}°")
    
    # ✅ NEW: Fitness convergence plot (best+avg) for both optimizers
    plot_fitness_convergence(comparison_results, filename="fitness_convergence.png")

    # Existing bar plots
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].bar(['GA', 'RL-GA'], 
                    [ga_avg_fitness, rl_avg_fitness],
                    yerr=[np.std(comparison_results['GeneticAlgorithm']['best_fitness']),
                          np.std(comparison_results['RLGeneticAlgorithm']['best_fitness'])],
                    capsize=5, color=['#3498db', '#e74c3c'], alpha=0.7)
        axes[0].set_ylabel('Average Best Fitness')
        axes[0].set_title('Fitness Comparison (Lower is Better)')
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(['GA', 'RL-GA'], 
                    [ga_avg_time, rl_avg_time],
                    yerr=[np.std(comparison_results['GeneticAlgorithm']['execution_time']),
                          np.std(comparison_results['RLGeneticAlgorithm']['execution_time'])],
                    capsize=5, color=['#3498db', '#e74c3c'], alpha=0.7)
        axes[1].set_ylabel('Average Execution Time (seconds)')
        axes[1].set_title('Execution Time Comparison')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimizer_comparison_problem4.png', dpi=300, bbox_inches='tight')
        print(f"\nComparison chart saved as 'optimizer_comparison.png'")
        plt.show()
    except ImportError:
        print("\nMatplotlib not available for visualization")
    
    return comparison_results, best_machines


if __name__ == "__main__":
    comparison_results, best_machines = main()
