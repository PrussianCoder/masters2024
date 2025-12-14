// 解法概要
// 1) トーラス幾何: mod_wrap で折り返し、TorusGeometry::get_min_distance が「相対速度で最も近づく時刻と距離」を返す。
// 2) 初期クラスタ: 未使用原子からランダム開始し、距離<=INITIAL_CLUSTER_DISTANCE_CRITERIA でスコア(時刻+距離/100)最小の原子を
//    貪欲追加して最大30個。追加のたび速度を重み平均し、位置を新時刻へ進める。
// 3) 間引き: サイズ<=5 を破棄。CLUSTER_FORMATION_TIME の円平均重心を計算し、既存クラスタと torus_distance<=L/8 なら近すぎとして捨てる。
// 4) 割当: 残り原子→クラスタの最短接近距離をコストに最小費用流で割り当て。容量は K-現クラスタサイズ。
// 5) 接続適用: 割当結果を時刻順に並べ、各原子の接続を(元の最短時刻に+10 まで猶予を見て) add_atom で反映し、全接続を出力。
#include <bits/stdc++.h>
using namespace std;

using ll = long long;

ll L;

inline ll mod_wrap(ll v) {
    v %= L;
    if (v < 0) v += L;
    return v;
}

struct Vector {
    ll x;
    ll y;

    Vector() : x(0), y(0) {}
    Vector(ll x_, ll y_) : x(x_), y(y_) {}

    Vector operator+(const Vector& other) const {
        return Vector(mod_wrap(x + other.x), mod_wrap(y + other.y));
    }
    Vector operator-(const Vector& other) const {
        return Vector(mod_wrap(x - other.x), mod_wrap(y - other.y));
    }
    Vector operator*(ll other) const {
        return Vector(mod_wrap(x * other), mod_wrap(y * other));
    }
    Vector operator/(ll other) const {
        return Vector(mod_wrap(x / other), mod_wrap(y / other));
    }
};

struct Velocity {
    ll vx;
    ll vy;

    Velocity() : vx(0), vy(0) {}
    Velocity(ll vx_, ll vy_) : vx(vx_), vy(vy_) {}

    double abs_velocity() const {
        return hypot(static_cast<double>(vx), static_cast<double>(vy));
    }

    Vector get_vector(int dt) const { return Vector(vx * dt, vy * dt); }

    Velocity operator+(const Velocity& other) const {
        return Velocity(vx + other.vx, vy + other.vy);
    }
    Velocity operator-(const Velocity& other) const {
        return Velocity(vx - other.vx, vy - other.vy);
    }
    Velocity operator*(ll other) const { return Velocity(vx * other, vy * other); }
    Velocity operator/(ll other) const { return Velocity(vx / other, vy / other); }
};

Velocity update_velocity(const Velocity& v1, const Velocity& v2, int w1, int w2) {
    return Velocity(
        static_cast<ll>((w1 * v1.vx + w2 * v2.vx) / (w1 + w2)),
        static_cast<ll>((w1 * v1.vy + w2 * v2.vy) / (w1 + w2)));
}

struct TorusGeometry {
    double get_distance(const Vector& v) const {
        ll dx = mod_wrap(v.x);
        ll dy = mod_wrap(v.y);
        ll dxf = min(dx, L - dx);
        ll dyf = min(dy, L - dy);
        return hypot(static_cast<double>(dxf), static_cast<double>(dyf));
    }

    pair<int, double> get_min_distance(const Vector& vec, const Velocity& vel, int tmax) const {
        if (vel.abs_velocity() == 0.0) return {0, get_distance(vec)};
        double d_min = numeric_limits<double>::infinity();
        int t_min = 0;
        double vv = vel.abs_velocity();
        double vv2 = vv * vv;
        for (int lx = -1; lx <= 1; ++lx) {
            for (int ly = -1; ly <= 1; ++ly) {
                ll x = vec.x + static_cast<ll>(lx) * L;
                ll y = vec.y + static_cast<ll>(ly) * L;
                double t_star_real = -(static_cast<double>(vel.vx) * x + static_cast<double>(vel.vy) * y) / vv2;
                int t_star = static_cast<int>(max(0.0, min(t_star_real, static_cast<double>(tmax - 1))));
                double d = get_distance(Vector(x, y) + vel.get_vector(t_star));
                if (d < d_min) {
                    d_min = d;
                    t_min = t_star;
                }
            }
        }
        return {t_min, d_min};
    }
};

struct Atom {
    int point_id;
    Vector position;
    Velocity velocity;

    Atom() : point_id(0), position(), velocity() {}
    Atom(int pid, const Vector& pos, const Velocity& vel) : point_id(pid), position(pos), velocity(vel) {}

    Vector get_position(int t) const { return position + velocity.get_vector(t); }
};

vector<Atom> g_atoms;
TorusGeometry torus_geometry;

class Cluster {
   public:
    explicit Cluster(int start_atom_idx)
        : atoms_{start_atom_idx},
          t_(0),
          weight_(1),
          velocity_(g_atoms[start_atom_idx].velocity),
          positions_{g_atoms[start_atom_idx].get_position(0)} {}

    void add_atom(int current_atom_idx, int new_atom_idx, int new_t) {
        int dt = new_t - t_;
        for (auto& pos : positions_) {
            pos = pos + velocity_.get_vector(dt);
        }

        velocity_ = update_velocity(velocity_, g_atoms[new_atom_idx].velocity, weight_, 1);
        connections_.push_back({new_t, current_atom_idx, new_atom_idx});
        atoms_.push_back(new_atom_idx);
        positions_.push_back(g_atoms[new_atom_idx].get_position(new_t));
        ++weight_;
        t_ = new_t;
    }

    tuple<int, double, int> get_nearest_atom(int target_atom_idx, int t_max) const {
        double min_distance = numeric_limits<double>::infinity();
        int min_atom = -1;
        int min_t = 0;
        Vector target_pos = g_atoms[target_atom_idx].get_position(t_);
        Velocity target_vel = g_atoms[target_atom_idx].velocity;
        for (size_t i = 0; i < atoms_.size(); ++i) {
            const Vector& pos = positions_[i];
            Vector vec = target_pos - pos;
            Velocity vel = target_vel - velocity_;
            auto [t_candidate, d_candidate] = torus_geometry.get_min_distance(vec, vel, t_max - t_);
            if (d_candidate <= min_distance) {
                min_distance = d_candidate;
                min_atom = atoms_[i];
                min_t = t_candidate + t_;
            }
        }
        return {min_t, min_distance, min_atom};
    }

    const vector<tuple<int, int, int>>& get_connections() const { return connections_; }
    const vector<int>& get_atoms() const { return atoms_; }
    int get_size() const { return weight_; }

   private:
    vector<int> atoms_;
    int t_;
    int weight_;
    Velocity velocity_;
    vector<Vector> positions_;
    vector<tuple<int, int, int>> connections_;
};

pair<double, double> compute_centroid_at_time(const Cluster& cluster, int t_ref) {
    const auto& atoms = cluster.get_atoms();
    if (atoms.empty()) return {0.0, 0.0};

    double cx = 0.0, sx = 0.0, cy = 0.0, sy = 0.0;
    for (int idx : atoms) {
        Vector pos = g_atoms[idx].get_position(t_ref);
        double ang_x = 2.0 * M_PI * static_cast<double>(pos.x) / static_cast<double>(L);
        double ang_y = 2.0 * M_PI * static_cast<double>(pos.y) / static_cast<double>(L);
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
}

double torus_distance(double x1, double y1, double x2, double y2) {
    double dx = fmod(x1 - x2, static_cast<double>(L));
    if (dx < 0) dx += static_cast<double>(L);
    if (dx > static_cast<double>(L) / 2.0) dx -= static_cast<double>(L);
    double dy = fmod(y1 - y2, static_cast<double>(L));
    if (dy < 0) dy += static_cast<double>(L);
    if (dy > static_cast<double>(L) / 2.0) dy -= static_cast<double>(L);
    return hypot(dx, dy);
}

Cluster greedy_connect(int start_atom_idx, const vector<int>& candidate_atoms, int t_max, int distance_criteria) {
    Cluster cluster(start_atom_idx);
    unordered_set<int> atoms_used;
    atoms_used.insert(start_atom_idx);

    while (cluster.get_size() < 30) {
        vector<tuple<double, int, double, int, int>> candidates;  // (score, time, distance, current_atom, next_atom)
        for (int atom_idx : candidate_atoms) {
            if (atoms_used.count(atom_idx)) continue;
            auto [t, d, closest_atom] = cluster.get_nearest_atom(atom_idx, t_max);
            if (d <= distance_criteria) {
                double score = static_cast<double>(t) + d / 100.0;
                candidates.emplace_back(score, t, d, closest_atom, atom_idx);
            }
        }
        if (candidates.empty()) return cluster;
        sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            if (get<0>(a) != get<0>(b)) return get<0>(a) < get<0>(b);
            return get<2>(a) < get<2>(b);
        });
        auto [score, best_time, best_dist, best_current_atom, best_next_atom] = candidates.front();
        cluster.add_atom(best_current_atom, best_next_atom, best_time);
        atoms_used.insert(best_next_atom);
    }
    return cluster;
}

class AssignmentProblemSolver {
   public:
    vector<pair<int, int>> solve_assignment_problem(
        int n, int m, const vector<vector<double>>& cost_pairs, const vector<int>& capacity) {
        if (accumulate(capacity.begin(), capacity.end(), 0) < n) {
            throw runtime_error("capacity is smaller than the number of points to assign");
        }

        struct Edge {
            int to;
            int rev;
            int cap;
            double cost;
        };

        class MinCostFlow {
           public:
            explicit MinCostFlow(int n) : size_(n), graph_(n) {}

            void add_edge(int fr, int to, int cap, double cost) {
                Edge f{to, static_cast<int>(graph_[to].size()), cap, cost};
                Edge b{fr, static_cast<int>(graph_[fr].size()), 0, -cost};
                graph_[fr].push_back(f);
                graph_[to].push_back(b);
            }

            double min_cost_flow(int s, int t, int max_flow) {
                const double INF = numeric_limits<double>::infinity();
                vector<double> h(size_, 0.0);
                vector<int> prev_v(size_), prev_e(size_);
                double res = 0.0;
                int flow = max_flow;

                while (flow > 0) {
                    vector<double> dist(size_, INF);
                    dist[s] = 0.0;
                    using P = pair<double, int>;
                    priority_queue<P, vector<P>, greater<P>> pq;
                    pq.emplace(0.0, s);

                    while (!pq.empty()) {
                        auto [d, v] = pq.top();
                        pq.pop();
                        if (dist[v] < d) continue;
                        for (int i = 0; i < static_cast<int>(graph_[v].size()); ++i) {
                            const Edge& e = graph_[v][i];
                            if (e.cap <= 0) continue;
                            double nd = d + e.cost + h[v] - h[e.to];
                            if (nd < dist[e.to]) {
                                dist[e.to] = nd;
                                prev_v[e.to] = v;
                                prev_e[e.to] = i;
                                pq.emplace(nd, e.to);
                            }
                        }
                    }

                    if (dist[t] == INF) throw runtime_error("could not send enough flow");
                    for (int v = 0; v < size_; ++v) {
                        if (dist[v] < INF) h[v] += dist[v];
                    }

                    int d = flow;
                    for (int v = t; v != s; v = prev_v[v]) {
                        d = min(d, graph_[prev_v[v]][prev_e[v]].cap);
                    }
                    flow -= d;
                    res += static_cast<double>(d) * h[t];
                    for (int v = t; v != s; v = prev_v[v]) {
                        Edge& e = graph_[prev_v[v]][prev_e[v]];
                        e.cap -= d;
                        graph_[v][e.rev].cap += d;
                    }
                }
                return res;
            }

            const vector<vector<Edge>>& graph() const { return graph_; }

           private:
            int size_;
            vector<vector<Edge>> graph_;
        };

        int source = 0;
        int point_offset = 1;
        int cluster_offset = point_offset + n;
        int sink = cluster_offset + m;
        MinCostFlow mcf(sink + 1);

        for (int i = 0; i < n; ++i) {
            mcf.add_edge(source, point_offset + i, 1, 0.0);
            for (int j = 0; j < m; ++j) {
                mcf.add_edge(point_offset + i, cluster_offset + j, 1, cost_pairs[i][j]);
            }
        }
        for (int j = 0; j < m; ++j) {
            if (capacity[j] > 0) mcf.add_edge(cluster_offset + j, sink, capacity[j], 0.0);
        }

        mcf.min_cost_flow(source, sink, n);

        vector<pair<int, int>> assignments;
        const auto& g = mcf.graph();
        for (int i = 0; i < n; ++i) {
            for (const auto& e : g[point_offset + i]) {
                if (e.to >= cluster_offset && e.to < cluster_offset + m && e.cap == 0) {
                    assignments.emplace_back(i, e.to - cluster_offset);
                }
            }
        }
        return assignments;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N, T, M, K;
    if (!(cin >> N >> T >> M >> K >> L)) return 0;

    const int INITIAL_CLUSTER_DISTANCE_CRITERIA = 3000;
    int CLUSTER_FORMATION_TIME = T / 3;

    g_atoms.resize(N);
    for (int i = 0; i < N; ++i) {
        ll x, y, vx, vy;
        cin >> x >> y >> vx >> vy;
        g_atoms[i] = Atom(i, Vector(x, y), Velocity(vx, vy));
    }

    mt19937 rng(random_device{}());
    vector<bool> used(N, false);
    vector<Cluster> clusters;

    auto build_candidate_list = [&]() {
        vector<int> candidates;
        candidates.reserve(N);
        for (int i = 0; i < N; ++i) {
            if (!used[i]) candidates.push_back(i);
        }
        return candidates;
    };

    while (static_cast<int>(clusters.size()) < M) {
        vector<int> candidates = build_candidate_list();
        if (candidates.empty()) break;
        uniform_int_distribution<int> dist(0, static_cast<int>(candidates.size()) - 1);
        int start_atom_idx = candidates[dist(rng)];
        Cluster cluster = greedy_connect(start_atom_idx, candidates, CLUSTER_FORMATION_TIME, INITIAL_CLUSTER_DISTANCE_CRITERIA);
        if (cluster.get_size() <= 5) continue;
        auto cand_centroid = compute_centroid_at_time(cluster, CLUSTER_FORMATION_TIME);
        bool too_close = false;
        for (const auto& existing : clusters) {
            auto existing_centroid = compute_centroid_at_time(existing, CLUSTER_FORMATION_TIME);
            double d = torus_distance(cand_centroid.first, cand_centroid.second,
                                      existing_centroid.first, existing_centroid.second);
            if (d <= static_cast<double>(L) / 8.0) {
                too_close = true;
                break;
            }
        }
        if (too_close) continue;
        for (int idx : cluster.get_atoms()) {
            used[idx] = true;
        }
        clusters.push_back(std::move(cluster));
    }

    vector<int> remaining_atoms;
    remaining_atoms.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (!used[i]) remaining_atoms.push_back(i);
    }

    vector<vector<double>> cost_pairs(remaining_atoms.size(), vector<double>(clusters.size(), 0.0));
    vector<vector<pair<int, int>>> connect_pairs(remaining_atoms.size(), vector<pair<int, int>>(clusters.size(), {0, -1}));
    for (size_t i = 0; i < remaining_atoms.size(); ++i) {
        int atom_idx = remaining_atoms[i];
        for (size_t j = 0; j < clusters.size(); ++j) {
            auto [t, d, closest_atom] = clusters[j].get_nearest_atom(atom_idx, T);
            cost_pairs[i][j] = d;
            connect_pairs[i][j] = {t, closest_atom};
        }
    }

    vector<int> capacity;
    capacity.reserve(clusters.size());
    for (const auto& c : clusters) capacity.push_back(K - c.get_size());

    AssignmentProblemSolver solver;
    vector<pair<int, int>> assignments = solver.solve_assignment_problem(static_cast<int>(remaining_atoms.size()),
                                                                        static_cast<int>(clusters.size()),
                                                                        cost_pairs,
                                                                        capacity);

    vector<tuple<int, int, int>> assigned_connections;
    assigned_connections.reserve(assignments.size());
    for (const auto& [i, j] : assignments) {
        assigned_connections.emplace_back(connect_pairs[i][j].first, remaining_atoms[i], j);
    }

    sort(assigned_connections.begin(), assigned_connections.end(),
         [](const auto& a, const auto& b) { return get<0>(a) < get<0>(b); });

    for (const auto& [t, atom_idx, cluster_idx] : assigned_connections) {
        int t_max = min(T, t + 10);
        auto [t2, d, closest_atom] = clusters[cluster_idx].get_nearest_atom(atom_idx, t_max);
        clusters[cluster_idx].add_atom(closest_atom, atom_idx, t2);
    }

    for (const auto& cluster : clusters) {
        for (const auto& [t, a1, a2] : cluster.get_connections()) {
            cout << t << " " << g_atoms[a1].point_id << " " << g_atoms[a2].point_id << "\n";
        }
    }

    return 0;
}
