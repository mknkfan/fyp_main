import os
import time
import copy
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from func.datastruct import Point, Machine
from gen_algo_params import GeneticAlgorithm   # <- change this import to where your GA class lives


# --------------------------
# Helper: run GA and record diversity history
# (your GA currently doesn't store diversity history in results)
# --------------------------
def run_ga_with_diversity(
    ga: GeneticAlgorithm,
    initial_population: Optional[List[np.ndarray]] = None,
    local_search_prob: float = 0.1,
    local_search_iters: int = 5,
    print_every: int = 0
) -> Tuple[List[Machine], float, Dict]:
    """
    Same logic as ga.optimize(), but also records diversity_history.
    Keeps your original GA class unchanged.
    """

    # Initialize population (shared or new)
    if initial_population is None:
        population = ga.create_initial_population()
    else:
        population = [chrom.copy() for chrom in initial_population]
        if len(population) != ga.population_size:
            raise ValueError(
                f"Initial population size {len(population)} != GA population_size {ga.population_size}"
            )

    best_solution = None
    best_fitness = float("inf")

    ga.best_fitness_history = []
    ga.avg_fitness_history = []
    diversity_history = []

    start_time = time.time()

    for generation in range(ga.generations):
        # Evaluate fitness
        fitness_scores = [ga.fitness_function(ch) for ch in population]

        # Track best solution
        best_idx = int(np.argmin(fitness_scores))
        if fitness_scores[best_idx] < best_fitness:
            best_fitness = float(fitness_scores[best_idx])
            best_solution = population[best_idx].copy()

        # Stats
        finite = [f for f in fitness_scores if f != float("inf")]
        avg_fitness = float(np.mean(finite)) if len(finite) > 0 else float("inf")
        diversity = float(ga.calculate_diversity(population))

        ga.best_fitness_history.append(best_fitness)
        ga.avg_fitness_history.append(avg_fitness)
        diversity_history.append(diversity)

        if print_every and generation % print_every == 0:
            print(
                f"[pop={ga.population_size} gen={ga.generations}] "
                f"Generation {generation}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}, Div={diversity:.3f}"
            )

        # Create new population
        new_population = []

        # Elitism
        elite_count = int(ga.elitism_rate * ga.population_size)
        elite_indices = np.argsort(fitness_scores)[:elite_count]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Offspring
        while len(new_population) < ga.population_size:
            p1 = ga.tournament_selection(population, fitness_scores)
            p2 = ga.tournament_selection(population, fitness_scores)

            c1, c2 = ga.crossover(p1, p2)
            c1 = ga.mutate(c1)
            c2 = ga.mutate(c2)

            # Local search on some solutions (keep same behavior as your GA)
            if random.random() < local_search_prob:
                c1 = ga.local_search(c1, max_iterations=local_search_iters)

            new_population.extend([c1, c2])

        population = new_population[:ga.population_size]

    end_time = time.time()

    final_machines = ga.decode_chromosome(best_solution)

    results = {
        "population_size": ga.population_size,
        "generations": ga.generations,
        "execution_time": end_time - start_time,
        "best_fitness_history": ga.best_fitness_history,
        "avg_fitness_history": ga.avg_fitness_history,
        "diversity_history": diversity_history,
        "final_fitness": best_fitness,
        "total_distance": ga.calculate_total_distance(final_machines),
    }

    return final_machines, best_fitness, results


# --------------------------
# Plotting helpers
# --------------------------
def save_run_plots(outdir: Path, results: Dict):
    pop = results["population_size"]
    gens = results["generations"]

    best_hist = results["best_fitness_history"]
    avg_hist = results["avg_fitness_history"]
    div_hist = results["diversity_history"]

    # 1 figure per variation: Fitness + Diversity (2 subplots)
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(best_hist, label="Best Fitness")
    ax1.plot(avg_hist, label="Avg Fitness")
    ax1.set_title(f"Fitness Convergence (pop={pop}, gen={gens})")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(div_hist, label="Diversity")
    ax2.set_title(f"Diversity (pop={pop}, gen={gens})")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Avg pairwise L2 distance")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig_path = outdir / f"pop_{pop}_gen_{gens}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_population_summary_plots(outdir: Path, all_results_for_pop: List[Dict]):
    """
    For a fixed population size, overlay curves for different generations.
    Useful to compare gen budgets at same pop size.
    """
    if not all_results_for_pop:
        return

    pop = all_results_for_pop[0]["population_size"]

    # Fitness overlay (best fitness)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    for r in sorted(all_results_for_pop, key=lambda x: x["generations"]):
        ax.plot(r["best_fitness_history"], label=f"gen={r['generations']}")
    ax.set_title(f"Best Fitness vs Generation (pop={pop})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"overlay_bestfitness_pop_{pop}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Diversity overlay
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    for r in sorted(all_results_for_pop, key=lambda x: x["generations"]):
        ax.plot(r["diversity_history"], label=f"gen={r['generations']}")
    ax.set_title(f"Diversity vs Generation (pop={pop})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Diversity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"overlay_diversity_pop_{pop}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------
# Main benchmark runner
# --------------------------
def benchmark_ga_grid(
    machines: List[Machine],
    sequence: List[int],
    robot_position: Point,
    workspace_bounds: Tuple[float, float, float, float],
    population_sizes=(100, 200, 500, 1000, 2000),
    generations_list=(100, 200, 500, 1000, 2000),
    out_root="ga_grid_results",
):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Cache initial population per population size (FAIRNESS constraint)
    initial_pop_cache: Dict[int, List[np.ndarray]] = {}

    all_results: List[Dict] = []

    for pop in population_sizes:
        # Create the shared initial population ONCE for this pop size
        # Use a deterministic seed per pop size so it's repeatable.
        seed_for_pop = 12345 + int(pop)
        random.seed(seed_for_pop)
        np.random.seed(seed_for_pop)

        ga_for_init = GeneticAlgorithm(
            machines=machines,
            sequence=sequence,
            robot_position=robot_position,
            workspace_bounds=workspace_bounds,
            population_size=pop,
            generations=1,  # irrelevant here
        )
        initial_pop_cache[pop] = ga_for_init.create_initial_population()

        # Run all generations for this pop size
        pop_results = []

        for gens in generations_list:
            # Reset seeds so pop=100 runs share the same random stream
            # and only differ by how long they run.
            random.seed(seed_for_pop)
            np.random.seed(seed_for_pop)

            ga = GeneticAlgorithm(
                machines=machines,
                sequence=sequence,
                robot_position=robot_position,
                workspace_bounds=workspace_bounds,
                population_size=pop,
                generations=gens,
            )

            # Make sure we pass the same initial population for this pop size
            initial_population = [c.copy() for c in initial_pop_cache[pop]]

            run_dir = out_root / f"pop_{pop}"
            run_dir.mkdir(parents=True, exist_ok=True)

            _, best_fit, res = run_ga_with_diversity(
                ga,
                initial_population=initial_population,
                local_search_prob=0.1,
                local_search_iters=5,
                print_every=0,   # set to 20 if you want logs
            )

            pop_results.append(res)
            all_results.append(res)

            # Save per-variation plot
            save_run_plots(run_dir, res)

            print(
                f"Done: pop={pop}, gen={gens} | best={best_fit:.4f} | time={res['execution_time']:.2f}s"
            )

        # Save overlay plots per population size
        save_population_summary_plots(out_root, pop_results)

    # Write CSV summary
    csv_path = out_root / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        f.write("population_size,generations,final_fitness,total_distance,execution_time\n")
        for r in all_results:
            f.write(
                f"{r['population_size']},{r['generations']},"
                f"{r['final_fitness']},{r['total_distance']},{r['execution_time']}\n"
            )

    # Compute "best generations per population size"
    best_by_pop: Dict[int, Dict] = {}
    for pop in population_sizes:
        candidates = [r for r in all_results if r["population_size"] == pop]
        best_r = min(candidates, key=lambda x: x["final_fitness"])
        best_by_pop[pop] = best_r

    # Print summary
    print("\n=== Best generations per population size ===")
    for pop in population_sizes:
        r = best_by_pop[pop]
        print(
            f"pop={pop}: best_gen={r['generations']} | best_fitness={r['final_fitness']:.4f} | time={r['execution_time']:.2f}s"
        )

    return all_results, best_by_pop


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    machines = [
    Machine(id=1, shape='l_shape', width=5.21, height=4.33,
            access_point=Point(1.48, -1.12),
            l_cutout_width=2.01, l_cutout_height=1.72),

    Machine(id=2, shape='rectangle', width=4.87, height=3.12,
            access_point=Point(0.92, -0.44)),

    Machine(id=3, shape='l_shape', width=4.56, height=5.46,
            access_point=Point(-0.72, 1.89),
            l_cutout_width=1.58, l_cutout_height=2.74),

    Machine(id=4, shape='rectangle', width=3.44, height=2.61,
            access_point=Point(0.13, 0.89)),

    Machine(id=5, shape='l_shape', width=5.94, height=3.77,
            access_point=Point(1.04, -0.88),
            l_cutout_width=2.23, l_cutout_height=1.32),

    Machine(id=6, shape='rectangle', width=4.32, height=2.48,
            access_point=Point(-0.63, -0.41)),

    Machine(id=7, shape='l_shape', width=3.88, height=4.91,
            access_point=Point(0.55, 1.22),
            l_cutout_width=1.25, l_cutout_height=2.01),

    Machine(id=8, shape='rectangle', width=3.27, height=2.74,
            access_point=Point(0.15, -0.52)),
    ]
    sequence = [1, 2, 3, 4, 5, 6, 7, 8]
    robot_position = Point(0, 0)
    workspace_bounds = (-15, 15, -15, 15)

    all_results, best_by_pop = benchmark_ga_grid(
        machines=machines,
        sequence=sequence,
        robot_position=robot_position,
        workspace_bounds=workspace_bounds,
        out_root="ga_grid_results"
    )
