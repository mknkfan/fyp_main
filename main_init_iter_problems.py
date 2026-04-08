import random
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Machine:
    id: int
    shape: str
    width: float
    height: float
    access_point: Point
    l_cutout_width: Optional[float] = None
    l_cutout_height: Optional[float] = None


def generate_problem(problem_idx: int):
    base_seed = 1000 + problem_idx
    random.seed(base_seed)
    np.random.seed(base_seed)

    workspace_bounds = (-15, 15, -15, 15)
    robot_position = Point(0, 0)

    machines = []
    for i in range(8):
        m_id = i + 1
        shape = "l_shape" if i % 2 == 0 else "rectangle"
        width = random.uniform(3.0, 6.0)
        height = random.uniform(2.5, 5.0)
        ap_x = random.uniform(-1.5, 1.5)
        ap_y = random.uniform(-1.5, 1.5)

        if shape == "l_shape":
            l_cutout_width = random.uniform(1.0, width / 2.0)
            l_cutout_height = random.uniform(1.0, height / 2.0)
            machine = Machine(
                id=m_id, shape="l_shape",
                width=width, height=height,
                access_point=Point(ap_x, ap_y),
                l_cutout_width=l_cutout_width,
                l_cutout_height=l_cutout_height,
            )
        else:
            machine = Machine(
                id=m_id, shape="rectangle",
                width=width, height=height,
                access_point=Point(ap_x, ap_y),
            )
        machines.append(machine)

    return machines, list(range(1, 9)), robot_position, workspace_bounds


for problem_idx in range(10):
    problem_id = problem_idx + 1
    machines, sequence, robot_position, workspace_bounds = generate_problem(problem_idx)

    print(f"\n{'='*60}")
    print(f"PROBLEM {problem_id}")
    print(f"{'='*60}")
    print(f"Workspace: {workspace_bounds}")
    print(f"Sequence:  {sequence}")
    print()

    for m in machines:
        if m.shape == "l_shape":
            print(
                f"  Machine {m.id} | l_shape    | "
                f"W={m.width:.3f}  H={m.height:.3f} | "
                f"AP=({m.access_point.x:+.3f}, {m.access_point.y:+.3f}) | "
                f"Cutout W={m.l_cutout_width:.3f}  H={m.l_cutout_height:.3f}"
            )
        else:
            print(
                f"  Machine {m.id} | rectangle  | "
                f"W={m.width:.3f}  H={m.height:.3f} | "
                f"AP=({m.access_point.x:+.3f}, {m.access_point.y:+.3f})"
            )