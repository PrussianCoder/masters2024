// 解法概要
// 1) トーラス上の幾何を使って原子間距離・最近接時間を計算。
// 2) 早い時間帯(t≈500)にクラスタの種を貪欲生成: ランダムに始点を選び、近い原子を距離閾値以下で順次追加して最大30個程度の小クラスタを作る。
// 3) 余り原子は「単体」または「2点マージ(一定距離未満なら事前に結合)」の AtomsGroup として束ねる。
// 4) OR-Tools(MPSolver+SCIP)で割り当て最適化: x[i][j]=AtomsGroup i をクラスタ j に貼り付ける0/1変数。
//    - 目的: (最近接距離 + 事前マージ距離d) の総和を最小化。
//    - 制約: 各原子は必ず1回だけ使う、各クラスタの残容量をちょうど満たす。
//    - 時間制限: プログラム開始から約1.7s以内に求解。
// 5) 求解結果を時間順に適用し、クラスタとグループを最も近い時刻で接続。最終的な接続リストを出力する。
// メモ: 貪欲で初期クラスタを作り、OR-Tools で割り当てる二段構えのシンプルな近似方針。


#include <bits/stdc++.h>
#include <chrono>
#include "ortools/linear_solver/linear_solver.h"
#include "absl/time/time.h"

using namespace std;
using namespace operations_research;

long long N, Tlim, M, K, L;
vector<array<long long, 4>> points;

const int CLUSTER_FORMATION_TIME_FACTOR = 500;
const int INITIAL_CLUSTER_DISTANCE_CRITERIA = 2500;
const int MERGE_THRESHOLD = 3300;
const int MERGE_TIME_LIMIT = 700;


double mod_wrap(double v) {
    double m = fmod(v, static_cast<double>(L));
    if (m < 0) m += L;
    return m;
}

struct Vector {
    double x;
    double y;

    Vector() : x(0), y(0) {}
    Vector(double x, double y) : x(x), y(y) {}

    Vector operator+(const Vector& other) const {
        return Vector(mod_wrap(x + other.x), mod_wrap(y + other.y));
    }
    Vector operator-(const Vector& other) const {
        return Vector(mod_wrap(x - other.x), mod_wrap(y - other.y));
    }
    Vector operator*(double k) const { return Vector(mod_wrap(x * k), mod_wrap(y * k)); }
};

struct Velocity {
    double vx;
    double vy;

    Velocity() : vx(0), vy(0) {}
    Velocity(double vx, double vy) : vx(vx), vy(vy) {}

    double abs_velocity() const { return sqrt(vx * vx + vy * vy); }

    Vector get_vector(int dt) const { return Vector(vx * dt, vy * dt); }

    Velocity operator+(const Velocity& other) const { return Velocity(vx + other.vx, vy + other.vy); }
    Velocity operator-(const Velocity& other) const { return Velocity(vx - other.vx, vy - other.vy); }
    Velocity operator*(double k) const { return Velocity(vx * k, vy * k); }
};

Velocity update_velocity(const Velocity& v1, const Velocity& v2, int w1, int w2) {
    double sum = w1 + w2;
    return Velocity((w1 * v1.vx + w2 * v2.vx) / sum, (w1 * v1.vy + w2 * v2.vy) / sum);
}

struct TorusGeometry {
    int get_distance(const Vector& vec) const {
        double dx = fabs(fmod(vec.x + L / 2.0, L) - L / 2.0);
        double dy = fabs(fmod(vec.y + L / 2.0, L) - L / 2.0);
        return static_cast<int>(hypot(dx, dy));
    }

    pair<int, int> get_min_distance(const Vector& vec, const Velocity& vel, int tmax) const {
        if (vel.abs_velocity() == 0) return {0, get_distance(vec)};
        double d_min = 1e30;
        int t_min = 0;
        for (int lx = -1; lx <= 1; ++lx) {
            for (int ly = -1; ly <= 1; ++ly) {
                double x = vec.x + lx * L;
                double y = vec.y + ly * L;
                double denom = vel.abs_velocity();
                double t_star_real = -(vel.vx * x + vel.vy * y) / (denom * denom);
                int t_star = static_cast<int>(max(0.0, min(t_star_real, static_cast<double>(tmax - 1))));
                int d = get_distance(Vector(x, y) + vel.get_vector(t_star));
                if (d < d_min) {
                    d_min = d;
                    t_min = t_star;
                }
            }
        }
        return {t_min, static_cast<int>(d_min)};
    }
};

struct Atom {
    int point_id;
    Vector position;
    Velocity velocity;

    Atom(int id, const Vector& p, const Velocity& v) : point_id(id), position(p), velocity(v) {}

    Vector get_position(int t) const { return position + velocity.get_vector(t); }
    Velocity get_velocity() const { return velocity; }
};

struct AtomsGroup {
    double d;
    int t;
    unordered_map<int, Vector> positions;  // atom_id -> position
    Velocity velocity;
    vector<tuple<int, int, int>> connections;  // (t, atom1_id, atom2_id)

    int get_size() const { return static_cast<int>(positions.size()); }
    double get_d() const { return d; }
};

struct Cluster {
    int t = 0;
    int weight = 1;
    Velocity velocity;
    unordered_map<int, Vector> positions;  // atom_id -> position
    vector<tuple<int, int, int>> connections;

    explicit Cluster(const Atom& start_atom) {
        velocity = start_atom.get_velocity();
        positions[start_atom.point_id] = start_atom.get_position(0);
    }

    void add_atom(int current_atom_id, const Atom& new_atom, int new_t) {
        for (auto& [id, pos] : positions) {
            pos = pos + velocity.get_vector(new_t - t);
        }
        velocity = update_velocity(velocity, new_atom.get_velocity(), weight, 1);
        connections.emplace_back(new_t, current_atom_id, new_atom.point_id);
        positions[new_atom.point_id] = new_atom.get_position(new_t);
        weight += 1;
        t = new_t;
    }

    void add_atom_group(int current_atom_id, int new_atom_id, const AtomsGroup& group, int new_t) {
        for (auto& [id, pos] : positions) {
            pos = pos + velocity.get_vector(new_t - t);
        }
        for (const auto& [id, pos] : group.positions) {
            positions[id] = pos + group.velocity.get_vector(new_t - group.t);
        }
        velocity = update_velocity(velocity, group.velocity, weight, group.get_size());
        connections.emplace_back(new_t, current_atom_id, new_atom_id);
        connections.insert(connections.end(), group.connections.begin(), group.connections.end());
        weight += group.get_size();
        t = new_t;
    }

    tuple<int, double, int> get_nearest_atom(const Atom& target_atom, int t_max, const TorusGeometry& tg) {
        double min_distance = 1e30;
        int min_atom = -1;
        int min_t = 0;
        for (const auto& [atom_id, pos] : positions) {
            Vector vec = target_atom.get_position(t) - pos;
            Velocity vel = target_atom.get_velocity() - velocity;
            auto [t_local, d] = tg.get_min_distance(vec, vel, t_max - t);
            if (d <= min_distance) {
                min_distance = d;
                min_atom = atom_id;
                min_t = t_local + t;
            }
        }
        return {min_t, min_distance, min_atom};
    }

    optional<tuple<int, double, int, int>> get_nearest_atoms_group(const AtomsGroup& group, int t_max,
                                                                   const TorusGeometry& tg) {
        double min_distance = 1e30;
        optional<tuple<int, double, int, int>> min_pair;
        int t_start = max(t, group.t);
        for (const auto& [atom_id, pos] : positions) {
            Vector pos_start = pos + velocity.get_vector(t_start - t);
            for (const auto& [atom2_id, pos2] : group.positions) {
                Vector pos2_start = pos2 + group.velocity.get_vector(t_start - group.t);
                Vector vec = pos_start - pos2_start;
                Velocity vel = velocity - group.velocity;
                auto [t_local, d] = tg.get_min_distance(vec, vel, t_max - t_start);
                if (d <= min_distance) {
                    min_distance = d;
                    min_pair = tuple<int, double, int, int>{t_local + t_start, static_cast<double>(d), atom_id,
                                                            atom2_id};
                }
            }
        }
        return min_pair;
    }

    const vector<tuple<int, int, int>>& get_connections() const { return connections; }
    vector<int> get_atoms() const {
        vector<int> ids;
        ids.reserve(positions.size());
        for (auto& kv : positions) ids.push_back(kv.first);
        return ids;
    }
    int get_size() const { return weight; }
    int get_time() const { return t; }
    Velocity get_velocity() const { return velocity; }
};

TorusGeometry torus_geometry;
vector<Atom> atoms;

int cluster_formation_time() { return static_cast<int>(CLUSTER_FORMATION_TIME_FACTOR); }

Cluster greedy_connect(const Atom& start_atom, const unordered_set<int>& atoms_available, int t_max) {
    Cluster cluster(start_atom);
    unordered_set<int> atoms_used{start_atom.point_id};
    while (cluster.get_size() < 30) {
        vector<tuple<int, double, int, int>> candidates;
        for (int atom_id : atoms_available) {
            if (atoms_used.count(atom_id)) continue;
            const Atom& atom = atoms[atom_id];
            auto [t, d, closest_atom] = cluster.get_nearest_atom(atom, t_max, torus_geometry);
            if (d <= INITIAL_CLUSTER_DISTANCE_CRITERIA) {
                candidates.emplace_back(t, d, closest_atom, atom_id);
            }
        }
        if (candidates.empty()) return cluster;
        sort(candidates.begin(), candidates.end(),
             [](const auto& a, const auto& b) { return tie(get<0>(a), get<1>(a)) < tie(get<0>(b), get<1>(b)); });
        auto [best_t, best_d, best_current_atom, best_next_atom] = candidates.front();
        cluster.add_atom(best_current_atom, atoms[best_next_atom], best_t);
        atoms_used.insert(best_next_atom);
    }
    return cluster;
}

// t < T/2 においては、点を適当に選んで貪欲に繋いでいく
pair<vector<Cluster>, unordered_set<int>> build_initial_clusters(const vector<Atom>& atoms_all) {
    unordered_set<int> atoms_not_used;
    for (const auto& a : atoms_all) atoms_not_used.insert(a.point_id);
    vector<Cluster> clusters;
    vector<int> order(atoms_all.size());
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int i, int j) {
        double si = atoms_all[i].velocity.abs_velocity();
        double sj = atoms_all[j].velocity.abs_velocity();
        if (si != sj) return si < sj;  // 速度が遅い順にスタート候補を使う
        return i < j;
    });

    auto torus_distance = [](double x1, double y1, double x2, double y2) {
        double dx = fmod(x1 - x2, static_cast<double>(L));
        if (dx < 0) dx += static_cast<double>(L);
        if (dx > static_cast<double>(L) / 2.0) dx -= static_cast<double>(L);
        double dy = fmod(y1 - y2, static_cast<double>(L));
        if (dy < 0) dy += static_cast<double>(L);
        if (dy > static_cast<double>(L) / 2.0) dy -= static_cast<double>(L);
        return hypot(dx, dy);
    };

    auto centroid_at_time = [&](const Cluster& c, int t_ref) -> pair<double, double> {
        double cx = 0.0, sx = 0.0, cy = 0.0, sy = 0.0;
        for (const auto& kv : c.positions) {
            Vector pos = kv.second + c.get_velocity().get_vector(t_ref - c.get_time());
            double ang_x = 2.0 * M_PI * pos.x / static_cast<double>(L);
            double ang_y = 2.0 * M_PI * pos.y / static_cast<double>(L);
            cx += cos(ang_x);
            sx += sin(ang_x);
            cy += cos(ang_y);
            sy += sin(ang_y);
        }
        double mean_ang_x = atan2(sx, cx);
        if (mean_ang_x < 0) mean_ang_x += 2.0 * M_PI;
        double mean_ang_y = atan2(sy, cy);
        if (mean_ang_y < 0) mean_ang_y += 2.0 * M_PI;
        double mean_x = mean_ang_x * static_cast<double>(L) / (2.0 * M_PI);
        double mean_y = mean_ang_y * static_cast<double>(L) / (2.0 * M_PI);
        return {mean_x, mean_y};
    };
    int count = 0;
    size_t cursor = 0;
    while (static_cast<int>(clusters.size()) < M && cursor < order.size()) {
        count++;
        int start_id = order[cursor++];
        if (!atoms_not_used.count(start_id)) continue;
        Cluster cluster = greedy_connect(atoms[start_id], atoms_not_used, cluster_formation_time());
        if (cluster.get_size() <= 10 && count < 100) continue;

        // クラスタ間の距離が程々に離れるようにフィルタ
        auto [cx, cy] = centroid_at_time(cluster, cluster_formation_time());
        bool too_close = false;
        double sep_threshold = max(1.0, static_cast<double>(L) / 10.0);
        for (const auto& existing : clusters) {
            auto [ex, ey] = centroid_at_time(existing, cluster_formation_time());
            double d = torus_distance(cx, cy, ex, ey);
            if (d <= sep_threshold && count < 100) {
                too_close = true;
                break;
            }
        }
        if (too_close) continue;

        clusters.push_back(cluster);
        for (int id : cluster.get_atoms()) atoms_not_used.erase(id);
    }
    return {clusters, atoms_not_used};
}

// 割り当て問題をマッチングの問題として解く
// cost_pairs[i][j] = i番目の余った点とj番目のクラスタとの距離が与えられる
// それぞれのclusterに対して、capacity[j]個の点を割り当てる
// コストが最小になるような割り当てを求め、その時の割り当て結果を返す
class AssignmentProblemSolver {
public:
    AssignmentProblemSolver() {
        solver.reset(MPSolver::CreateSolver("SCIP"));
        if (!solver) throw runtime_error("Failed to create solver");
    }

    void set_time_limit_ms(int ms) {
        // OR-Tools の求解を開始からの残り時間以内に打ち切る
        solver->SetTimeLimit(absl::Milliseconds(ms));
    }

    // assignments は AtomsGroup へのポインタを持たせてコピーを避ける
    vector<tuple<int, double, const AtomsGroup*, Cluster*>> solve_assignment_problem(
        const vector<AtomsGroup>& atoms_groups, vector<Cluster>& clusters) {
        const int G = static_cast<int>(atoms_groups.size());
        const int C = static_cast<int>(clusters.size());
        vector<vector<double>> cost_pairs(G, vector<double>(C, 0));
        vector<vector<optional<tuple<int, double, int, int>>>> connect_info(
            G, vector<optional<tuple<int, double, int, int>>>(C));

        for (int i = 0; i < G; ++i) {
            for (int j = 0; j < C; ++j) {
                auto res = clusters[j].get_nearest_atoms_group(atoms_groups[i], static_cast<int>(Tlim), torus_geometry);
                if (!res) continue;
                auto [t, d, a1, a2] = *res;
                connect_info[i][j] = res;
                cost_pairs[i][j] = d;
            }
        }

        vector<int> capacity(C);
        for (int j = 0; j < C; ++j) capacity[j] = static_cast<int>(K - clusters[j].get_size());
        vector<int> weights(G);
        for (int i = 0; i < G; ++i) weights[i] = atoms_groups[i].get_size();

        vector<vector<MPVariable*>> x(G, vector<MPVariable*>(C, nullptr));
        for (int i = 0; i < G; ++i) {
            for (int j = 0; j < C; ++j) {
                if (cost_pairs[i][j] > 20000.0) {
                    // 大きすぎる組み合わせは固定で0
                    x[i][j] = solver->MakeIntVar(0.0, 0.0, "");
                } else {
                    x[i][j] = solver->MakeBoolVar("");
                }
            }
        }

        MPObjective* objective = solver->MutableObjective();
        for (int i = 0; i < G; ++i) {
            double base_cost = atoms_groups[i].get_d();
            for (int j = 0; j < C; ++j) {
                objective->SetCoefficient(x[i][j], cost_pairs[i][j] + base_cost);
            }
        }
        objective->SetMinimization();

        // 各 atom につき「使用は1回」の制約。unordered_map ではなく配列で高速化。
        vector<vector<MPVariable*>> var_per_atom(N);
        for (int i = 0; i < G; ++i) {
            for (const auto& kv : atoms_groups[i].positions) {
                int atom_id = kv.first;
                for (int j = 0; j < C; ++j) var_per_atom[atom_id].push_back(x[i][j]);
            }
        }
        for (const auto& vars : var_per_atom) {
            if (vars.empty()) continue;
            MPConstraint* c = solver->MakeRowConstraint(1, 1);
            for (auto* v : vars) c->SetCoefficient(v, 1);
        }

        for (int j = 0; j < C; ++j) {
            MPConstraint* c = solver->MakeRowConstraint(capacity[j], capacity[j]);
            for (int i = 0; i < G; ++i) c->SetCoefficient(x[i][j], weights[i]);
        }

        // 収束を速めるための SCIP パラメータ
        std::ostringstream param;
        // time limit is already set via SetTimeLimit; reinforce here in seconds
        param << "limits/time=" << (solver->time_limit() / 1000.0) << "\n";
        param << "presolving/maxrounds=1\n";          // 軽い前処理だけ
        param << "heuristics/emphasis=fast\n";        // 発見優先
        param << "parallel/maxnthreads=4\n";          // 適度に並列
        param << "limits/nodes=200000\n";             // ノード数に上限を設定
        param << "lp/solvefreq=0\n";                  // LP を毎回解かない
        solver->SetSolverSpecificParametersAsString(param.str());

        auto status = solver->Solve();
        if (status != MPSolver::OPTIMAL && status != MPSolver::FEASIBLE) {
            throw runtime_error("Failed to solve the assignment problem");
        }

        vector<tuple<int, double, const AtomsGroup*, Cluster*>> assignments;
        assignments.reserve(G);  // 上限
        for (int i = 0; i < G; ++i) {
            for (int j = 0; j < C; ++j) {
                if (x[i][j]->solution_value() > 0.5) {
                    auto info = connect_info[i][j];
                    if (!info) continue;
                    auto [t, d, a1, a2] = *info;
                    assignments.emplace_back(t, d, &atoms_groups[i], &clusters[j]);
                }
            }
        }
        sort(assignments.begin(), assignments.end(),
             [](const auto& a, const auto& b) { return tie(get<0>(a), get<1>(a)) < tie(get<0>(b), get<1>(b)); });
        return assignments;
    }

private:
    unique_ptr<MPSolver> solver;
};

AtomsGroup single_atoms(const Atom& atom) {
    AtomsGroup g;
    g.d = 0;
    g.t = 0;
    g.positions[atom.point_id] = atom.get_position(0);
    g.velocity = atom.get_velocity();
    return g;
}

optional<AtomsGroup> double_atoms(const Atom& a1, const Atom& a2) {
    auto [t, d] = torus_geometry.get_min_distance(a1.position - a2.position, a1.velocity - a2.velocity, static_cast<int>(MERGE_TIME_LIMIT));
    if (d >= MERGE_THRESHOLD) return nullopt;
    AtomsGroup g;
    g.d = d;
    g.t = t;
    g.positions[a1.point_id] = a1.get_position(t);
    g.positions[a2.point_id] = a2.get_position(t);
    g.velocity = update_velocity(a1.velocity, a2.velocity, 1, 1);
    g.connections.emplace_back(t, a1.point_id, a2.point_id);
    return g;
}

// 余った点について、マージできるものをマージする
vector<AtomsGroup> build_atoms_groups(const unordered_set<int>& remaining_atoms) {
    vector<int> atoms_list(remaining_atoms.begin(), remaining_atoms.end());
    vector<AtomsGroup> groups;
    int n = atoms_list.size();
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            auto ag = double_atoms(atoms[atoms_list[i]], atoms[atoms_list[j]]);
            if (ag) groups.push_back(*ag);
        }
    }
    for (int id : atoms_list) groups.push_back(single_atoms(atoms[id]));
    return groups;
}

// 割り当ての結果を元に、時間順でclusterとatoms_groupを繋げていく
vector<tuple<int, int, int>> connect_assignments(const vector<tuple<int, double, const AtomsGroup*, Cluster*>>& assignments,
                                                 vector<Cluster>& clusters) {
    for (const auto& [t, d, group_ptr, cluster_ptr] : assignments) {
        auto res = cluster_ptr->get_nearest_atoms_group(*group_ptr, static_cast<int>(min<long long>(Tlim, t + 10)), torus_geometry);
        if (!res) continue;
        auto [t2, d2, cluster_atom, group_atom] = *res;
        cluster_ptr->add_atom_group(cluster_atom, group_atom, *group_ptr, t2);
    }

    vector<tuple<int, int, int>> connections;
    for (const auto& c : clusters) {
        const auto& conn = c.get_connections();
        connections.insert(connections.end(), conn.begin(), conn.end());
    }
    return connections;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto program_start = chrono::steady_clock::now();

    if (!(cin >> N >> Tlim >> M >> K >> L)) return 0;
    points.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> points[i][0] >> points[i][1] >> points[i][2] >> points[i][3];
    }

    atoms.reserve(N);
    for (int i = 0; i < N; ++i) {
        atoms.emplace_back(i, Vector(points[i][0], points[i][1]), Velocity(points[i][2], points[i][3]));
    }

    // 全体の流れ: クラスタ構築 → 余り点のマージ → 割り当て最適化 → 時間順に接続
    auto [clusters, atoms_not_used] = build_initial_clusters(atoms);
    vector<AtomsGroup> atoms_groups = build_atoms_groups(atoms_not_used);
    AssignmentProblemSolver solver;
    // プログラム開始からの残り時間を渡して求解を1.8sで打ち切る
    int elapsed_ms =
        static_cast<int>(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - program_start).count());
    int remaining_ms = max(1, 1000 - elapsed_ms);
    solver.set_time_limit_ms(remaining_ms);
    auto assignments = solver.solve_assignment_problem(atoms_groups, clusters);
    auto connections = connect_assignments(assignments, clusters);

    for (auto [t, a1, a2] : connections) {
        cout << t << " " << a1 << " " << a2 << "\n";
    }
    return 0;
}
