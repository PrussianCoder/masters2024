from __future__ import annotations

import math
import random
from collections import defaultdict

from ortools.linear_solver import pywraplp

N, T, M, K, L = map(int, input().split())
points = [list(map(int, input().split())) for _ in range(N)]

CLUSTER_FORMATION_TIME = T // 3
INITIAL_CLUSTER_DISTANCE_CRITERIA = 3000
MERGE_THRESHOLD = 3000


class Vector:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Vector(x={self.x}, y={self.y})"

    def __add__(self, other: Vector) -> Vector:
        return Vector((self.x + other.x) % L, (self.y + other.y) % L)

    def __sub__(self, other: Vector) -> Vector:
        return Vector((self.x - other.x) % L, (self.y - other.y) % L)

    def __mul__(self, other: int) -> Vector:
        return Vector((self.x * other) % L, (self.y * other) % L)

    def __truediv__(self, other: int) -> Vector:
        return Vector((self.x / other) % L, (self.y / other) % L)


class Velocity:
    def __init__(self, vx: int, vy: int):
        self.vx = vx
        self.vy = vy

    def __repr__(self) -> str:
        return f"Velocity(vx={self.vx}, vy={self.vy})"

    def abs_velocity(self) -> int:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy)

    def get_vector(self, dt: int) -> Vector:
        return Vector(self.vx * dt, self.vy * dt)

    def __add__(self, other: Velocity) -> Velocity:
        return Velocity((self.vx + other.vx), (self.vy + other.vy))

    def __sub__(self, other: Velocity) -> Velocity:
        return Velocity((self.vx - other.vx), (self.vy - other.vy))

    def __mul__(self, other: int) -> Velocity:
        return Velocity((self.vx * other), (self.vy * other))

    def __truediv__(self, other: int) -> Velocity:
        return Velocity((self.vx / other), (self.vy / other))


def update_velocity(
    velocity_1: Velocity, velocity_2: Velocity, w1: int, w2: int
) -> Velocity:
    return Velocity(
        (w1 * velocity_1.vx + w2 * velocity_2.vx) / (w1 + w2),
        (w1 * velocity_1.vy + w2 * velocity_2.vy) / (w1 + w2),
    )


class TorusGeometry:
    def get_distance(self, vector: Vector) -> int:
        dx = abs((vector.x + L / 2) % L - L / 2)
        dy = abs((vector.y + L / 2) % L - L / 2)
        return int(math.hypot(dx, dy))

    def get_min_distance(
        self, vector: Vector, velocity_vector: Velocity, tmax: int
    ) -> tuple[int, int]:
        if velocity_vector.abs_velocity() == 0:
            return 0, self.get_distance(vector)
        d_min = float("inf")
        t_min = 0
        for lx in range(-1, 2):
            for ly in range(-1, 2):
                x = vector.x + lx * L
                y = vector.y + ly * L
                t_star_real = -(velocity_vector.vx * x + velocity_vector.vy * y) / (
                    velocity_vector.abs_velocity() ** 2
                )
                t_star = int(max(0, min(t_star_real, tmax - 1)))
                d = self.get_distance(Vector(x, y) + velocity_vector.get_vector(t_star))
                if d < d_min:
                    d_min = d
                    t_min = t_star
        return t_min, d_min


class Atom:
    def __init__(self, point_id: int, position: Vector, velocity: Velocity):
        self.point_id = point_id
        self.position = position
        self.velocity = velocity

    def __repr__(self) -> str:
        return f"Atom(point_id={self.point_id})"

    def get_position(self, t: int) -> Vector:
        return self.position + self.velocity.get_vector(t)

    def get_velocity(self) -> Velocity:
        return self.velocity

    @classmethod
    def create_from_input(cls, point_id: int, x: int, y: int, vx: int, vy: int) -> Atom:
        return cls(point_id, Vector(x, y), Velocity(vx, vy))


class AtomsGroup:
    def __init__(
        self,
        d: float,
        t: int,
        positions: dict[Atom, Vector],
        velocity: Velocity,
        connections: list[tuple[int, Atom, Atom]],
    ) -> None:
        self.d = d
        self.t = t
        self.positions = positions
        self.velocity = velocity
        self.connections = connections

    def __repr__(self) -> str:
        return f"AtomsGroup(d={self.d}, t={self.t}, positions={self.positions!r}, velocity={self.velocity!r}, connections={self.connections!r})"

    def get_velocity(self) -> Velocity:
        return self.velocity

    def get_t(self) -> int:
        return self.t

    def get_d(self) -> float:
        return self.d

    def get_positions(self) -> dict[Atom, Vector]:
        return self.positions

    def get_connections(self) -> list[tuple[int, Atom, Atom]]:
        return self.connections

    def get_size(self) -> int:
        return len(self.positions)


class Cluster:
    def __init__(self, start_atom: Atom) -> None:
        self.t = 0
        self.weight = 1
        self.velocity = start_atom.get_velocity()
        self.positions = {start_atom: start_atom.get_position(0)}
        self.connections: list[tuple[int, Atom, Atom]] = []

    def __repr__(self) -> str:
        return f"Cluster(t={self.t}, weight={self.weight}, velocity={self.velocity!r}, position={self.positions!r}, connections={self.connections!r})"

    def add_atom(self, current_atom: Atom, new_atom: Atom, new_t: int) -> None:
        # update position at new_t
        for atom, position in self.positions.items():
            self.positions[atom] = position + self.velocity.get_vector(new_t - self.t)
        # update velocity
        self.velocity = update_velocity(
            self.velocity,
            new_atom.get_velocity(),
            self.weight,
            1,
        )
        # update connections
        self.connections.append((new_t, current_atom, new_atom))
        # update atoms
        self.positions[new_atom] = new_atom.get_position(new_t)
        # update weight
        self.weight += 1
        # update t
        self.t = new_t

    def add_atom_group(
        self, current_atom: Atom, new_atom: Atom, new_atom_group: AtomsGroup, new_t: int
    ) -> None:
        # update position at new_t
        for atom, position in self.positions.items():
            self.positions[atom] = position + self.velocity.get_vector(new_t - self.t)
        # update atoms on new_atom_group
        for atom, position in new_atom_group.positions.items():
            self.positions[atom] = position + self.velocity.get_vector(
                new_t - new_atom_group.t
            )
        # update velocity
        self.velocity = update_velocity(
            self.velocity,
            new_atom_group.velocity,
            self.weight,
            new_atom_group.get_size(),
        )
        # update connections
        self.connections.append((new_t, current_atom, new_atom))
        self.connections.extend(new_atom_group.connections)
        # update weight
        self.weight += new_atom_group.get_size()
        # update t
        self.t = new_t

    def get_nearest_atom(
        self, target_atom: Atom, t_max: int
    ) -> tuple[int, float, Atom]:
        min_distance = float("inf")
        min_atom = None
        min_t = 0
        for atom, position in self.positions.items():
            vec = target_atom.get_position(self.t) - position
            vel = target_atom.get_velocity() - self.velocity
            t, d = torus_geometry.get_min_distance(vec, vel, t_max - self.t)
            if d <= min_distance:
                min_distance = d
                min_atom = atom
                min_t = t + self.t
        return min_t, min_distance, min_atom

    def get_nearest_atoms_group(
        self, atoms_group: AtomsGroup, t_max: int
    ) -> tuple[int, float, Atom, Atom] | None:
        min_distance = float("inf")
        min_pair: tuple[int, float, Atom, Atom] | None = None
        t_start = max(self.t, atoms_group.t)
        for atom, position in self.positions.items():
            pos_start = position + self.velocity.get_vector(t_start - self.t)
            for atom2, position2 in atoms_group.positions.items():
                pos2_start = position2 + atoms_group.velocity.get_vector(
                    t_start - atoms_group.t
                )
                vec = pos_start - pos2_start
                vel = self.velocity - atoms_group.velocity
                t, d = torus_geometry.get_min_distance(vec, vel, t_max - t_start)
                if d <= min_distance:
                    min_distance = d
                    min_pair = (t + t_start, d, atom, atom2)
        return min_pair

    def get_connections(self) -> list[tuple[int, Atom, Atom]]:
        return self.connections

    def get_atoms(self) -> list[Atom]:
        return list(self.positions.keys())

    def get_size(self) -> int:
        return self.weight


torus_geometry = TorusGeometry()
atoms: list[Atom] = []
for i, (x, y, vx, vy) in enumerate(points):
    atoms.append(Atom.create_from_input(i, x, y, vx, vy))


def greedy_connect(start_atom: Atom, atoms: set[Atom], t_max: int) -> Cluster:
    cluster = Cluster(start_atom)
    atoms_used = set([start_atom])
    while cluster.get_size() < 30:
        candidates: list[tuple[int, float, Atom, Atom]] = []
        for atom in atoms:
            if atom in atoms_used:
                continue
            t, d, closest_atom = cluster.get_nearest_atom(atom, t_max)
            if d <= INITIAL_CLUSTER_DISTANCE_CRITERIA:
                candidates.append((t, d, closest_atom, atom))
        if not candidates:
            return cluster
        candidates.sort(key=lambda x: (x[0] + x[1] / 100, x[1]))
        best_t, _, best_current_atom, best_next_atom = candidates[0]
        cluster.add_atom(best_current_atom, best_next_atom, best_t)
        atoms_used.add(best_next_atom)
    return cluster


# 割り当て問題をマッチングの問題として解く
# cost_pairs[i][j] = i番目の余った点とj番目のクラスタとの距離が与えられる
# それぞれのclusterに対して、capacity[j]個の点を割り当てる
# コストが最小になるような割り当てを求め、その時の割り当て結果を返す
class AssignmentProblemSolver:
    def __init__(self):
        self.solver = pywraplp.Solver(
            "assignment_problem", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
        )
        self.torus_geometry = TorusGeometry()

    def solve_assignment_problem(
        self, atoms_groups: list[AtomsGroup], clusters: list[Cluster]
    ) -> list[tuple[int, float, AtomsGroup, Cluster]]:
        cost_pairs = [[0] * len(clusters) for _ in range(len(atoms_groups))]
        connect_info = [[None] * len(clusters) for _ in range(len(atoms_groups))]
        for i, atoms_group in enumerate(atoms_groups):
            for j, cluster in enumerate(clusters):
                t, d, closest_atom1, closest_atom2 = cluster.get_nearest_atoms_group(
                    atoms_group, T
                )
                connect_info[i][j] = (t, d, closest_atom1, closest_atom2)
                cost_pairs[i][j] = d
        capacity = [K - cluster.get_size() for cluster in clusters]
        weights = [atoms_group.get_size() for atoms_group in atoms_groups]
        x = [
            [self.solver.BoolVar(f"x_{i}_{k}") for k in range(len(clusters))]
            for i in range(len(atoms_groups))
        ]
        # x[i][k] = 1 if i番目のグループをk番目のクラスタに割り当てる

        # 目的関数
        objective = self.solver.Sum(
            [
                (cost_pairs[i][k] + atoms_groups[i].get_d()) * x[i][k]
                for i in range(len(atoms_groups))
                for k in range(len(clusters))
            ]
        )
        self.solver.Minimize(objective)

        # 制約条件
        # 全てのatomsが1回だけ使用される
        var_per_atoms: defaultdict[Atom, list[pywraplp.Variable]] = defaultdict(list)
        for i, atoms_group in enumerate(atoms_groups):
            for atom in atoms_group.get_positions().keys():
                for j in range(len(clusters)):
                    var_per_atoms[atom].append(x[i][j])
        for atom in var_per_atoms.keys():
            self.solver.Add(sum(var_per_atoms[atom]) == 1)

        # 各クラスタに対して、capacity[j]個の点を割り当てる
        for j in range(len(clusters)):
            self.solver.Add(
                self.solver.Sum(
                    [weights[i] * x[i][j] for i in range(len(atoms_groups))]
                )
                == capacity[j]
            )

        status = self.solver.Solve()
        if status != self.solver.OPTIMAL:
            raise ValueError("Failed to solve the assignment problem")

        assignments: list[tuple[int, float, AtomsGroup, Cluster]] = []
        for i in range(len(atoms_groups)):
            for k in range(len(clusters)):
                if x[i][k].solution_value() == 1:
                    t, d, closest_atom1, closest_atom2 = connect_info[i][k]
                    assignments.append((t, d, atoms_groups[i], clusters[k]))
        assignments.sort(key=lambda x: (x[0], x[1]))
        return assignments


########################################################
# t < T/2 においては、点を適当に選んで貪欲に繋いでいく
########################################################


atoms_not_used: set[Atom] = set(atoms)
clusters: list[Cluster] = []

while len(clusters) < M:
    start_atom = random.choice(list(atoms_not_used))
    cluster = greedy_connect(start_atom, atoms_not_used, CLUSTER_FORMATION_TIME)
    if cluster.get_size() <= 5:
        continue
    clusters.append(cluster)
    atoms_not_used.difference_update(cluster.get_atoms())

remaining_atoms = list(atoms_not_used)

########################################################
# 余った点について、マージできるものをマージする
########################################################


def single_atoms(atom: Atom) -> AtomsGroup:
    return AtomsGroup(0, 0, {atom: atom.get_position(0)}, atom.get_velocity(), [])


def double_atoms(atom_1: Atom, atom_2: Atom) -> AtomsGroup:
    t, d = torus_geometry.get_min_distance(
        atom_1.position - atom_2.position, atom_1.velocity - atom_2.velocity, T
    )
    if d >= MERGE_THRESHOLD:
        return None
    connections = [(t, atom_1, atom_2)]
    positions = {atom_1: atom_1.get_position(t), atom_2: atom_2.get_position(t)}

    velocity = update_velocity(atom_1.velocity, atom_2.velocity, 1, 1)
    return AtomsGroup(d, t, positions, velocity, connections)


atoms_groups: list[AtomsGroup] = []

for i in range(len(remaining_atoms)):
    for j in range(i + 1, len(remaining_atoms)):
        atoms_group = double_atoms(remaining_atoms[i], remaining_atoms[j])
        if atoms_group is not None:
            atoms_groups.append(atoms_group)

for atom in remaining_atoms:
    atoms_groups.append(single_atoms(atom))

########################################################
# 上で作ったクラスタに対して、atoms_groupsをグループに割り当てていく
########################################################


assignment_solver = AssignmentProblemSolver()
assignments = assignment_solver.solve_assignment_problem(atoms_groups, clusters)

########################################################
# 割り当ての結果を元に、時間順でclusterとatoms_groupを繋げていく
########################################################

ans: list[tuple[int, Atom, Atom]] = []
for t, d, atoms_group, cluster in assignments:
    t2, d, cluster_atom, atoms_group_atom = cluster.get_nearest_atoms_group(
        atoms_group, min(T, t + 10)
    )
    cluster.add_atom_group(cluster_atom, atoms_group_atom, atoms_group, t2)

for cluster in clusters:
    for t, atom1, atom2 in cluster.get_connections():
        print(t, atom1.point_id, atom2.point_id)
