// Tesseract decoder core — self-contained, no stim dependency.
//
// Adapted from Google's Tesseract decoder
// (https://github.com/quantumlib/tesseract-decoder).
// See LICENSE-tesseract-decoder for the original Apache-2.0 license.
//
// This header provides the same types and algorithms as the original
// common.h/cc, utils.h/cc, visualization.h/cc, and tesseract.h/cc,
// restructured to accept raw error data directly instead of a
// stim::DetectorErrorModel.

#ifndef TESSERACT_CORE_H_
#define TESSERACT_CORE_H_

#include <boost/dynamic_bitset.hpp>
#include <boost/functional/hash.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ── common.h types ──────────────────────────────────────────────────

namespace common {

// Represents the effect of an error (same as common::Symptom).
struct Symptom {
    std::vector<int> detectors;

    struct hash {
        size_t operator()(const Symptom& s) const {
            size_t h = 0;
            for (int i : s.detectors) h += std::hash<int>{}(i);
            return h;
        }
    };

    bool operator==(const Symptom& other) const {
        return detectors == other.detectors;
    }
};

// Represents a specific subset of errors in the power set of all errors.
// `parent_idx` traces back to the root of the error set in the arena.
struct ErrorChainNode {
    size_t error_index;
    size_t min_detector;
    int64_t parent_idx = -1;
};

// Represents an error / weighted hyperedge (same as common::Error).
struct Error {
    double likelihood_cost;
    Symptom symptom;

    Error() = default;

    // Construct from a probability and sorted detector list.
    Error(double probability, std::vector<int> detectors)
        : symptom{std::move(detectors)} {
        double p = std::clamp(probability, 1e-15, 1.0 - 1e-15);
        likelihood_cost = -std::log(p / (1.0 - p));
    }

    double get_probability() const {
        return 1.0 / (1.0 + std::exp(likelihood_cost));
    }
};

// Merge weight formula (same as common::merge_weights).
inline double merge_weights(double a, double b) {
    auto sgn = std::copysign(1.0, a) * std::copysign(1.0, b);
    auto signed_min = sgn * std::min(std::abs(a), std::abs(b));
    return signed_min + std::log(1 + std::exp(-std::abs(a + b)))
                      - std::log(1 + std::exp(-std::abs(a - b)));
}

}  // namespace common

// ── tesseract.h types ───────────────────────────────────────────────

struct DetectorCostTuple {
    uint32_t error_blocked;
    uint32_t detectors_count;
};

struct ErrorCost {
    double likelihood_cost;
    double min_cost;
};

struct Node {
    double cost;
    size_t num_dets;
    size_t depth;
    int64_t error_chain_idx = -1;

    bool operator>(const Node& other) const {
        return cost > other.cost || (cost == other.cost && num_dets < other.num_dets);
    }
};

struct TesseractConfig {
    int det_beam = 5;
    bool beam_climbing = false;
    bool no_revisit_dets = true;
    bool merge_errors = true;
    size_t pqlimit = 200000;
    std::vector<std::vector<size_t>> det_orders;
    double det_penalty = 0;
};

// ── utils.h: detector ordering (build_det_orders_bfs) ───────────────

namespace tesseract_utils {

inline std::vector<std::vector<size_t>> build_detector_graph(
    size_t num_detectors,
    const std::vector<common::Error>& errors)
{
    std::vector<std::vector<size_t>> neighbors(num_detectors);
    for (const auto& error : errors) {
        const auto& dets = error.symptom.detectors;
        for (size_t i = 0; i < dets.size(); ++i) {
            for (size_t j = i + 1; j < dets.size(); ++j) {
                auto a = static_cast<size_t>(dets[i]);
                auto b = static_cast<size_t>(dets[j]);
                neighbors[a].push_back(b);
                neighbors[b].push_back(a);
            }
        }
    }
    for (auto& neigh : neighbors) {
        std::sort(neigh.begin(), neigh.end());
        neigh.erase(std::unique(neigh.begin(), neigh.end()), neigh.end());
    }
    return neighbors;
}

inline std::vector<std::vector<size_t>> build_det_orders_bfs(
    size_t num_detectors,
    const std::vector<common::Error>& errors,
    size_t num_det_orders,
    uint64_t seed)
{
    std::mt19937_64 rng(seed);
    auto graph = build_detector_graph(num_detectors, errors);
    std::vector<std::vector<size_t>> det_orders(num_det_orders);
    if (num_detectors == 0) return det_orders;

    std::uniform_int_distribution<size_t> dist_det(0, graph.size() - 1);
    for (size_t det_order = 0; det_order < num_det_orders; ++det_order) {
        std::vector<size_t> perm;
        perm.reserve(graph.size());
        std::vector<char> visited(graph.size(), false);
        std::queue<size_t> q;
        size_t start = dist_det(rng);
        while (perm.size() < graph.size()) {
            if (!visited[start]) {
                visited[start] = true;
                q.push(start);
                perm.push_back(start);
            }
            while (!q.empty()) {
                size_t cur = q.front();
                q.pop();
                auto neigh = graph[cur];
                std::shuffle(neigh.begin(), neigh.end(), rng);
                for (size_t n : neigh) {
                    if (!visited[n]) {
                        visited[n] = true;
                        q.push(n);
                        perm.push_back(n);
                    }
                }
            }
            if (perm.size() < graph.size()) {
                do { start = dist_det(rng); } while (visited[start]);
            }
        }
        std::vector<size_t> inv_perm(graph.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            inv_perm[perm[i]] = i;
        }
        det_orders[det_order] = inv_perm;
    }
    return det_orders;
}

}  // namespace tesseract_utils

// ── common.h/cc: merge_indistinguishable_errors ─────────────────────

namespace common {

/// Merge errors with identical detector sets (same as common::merge_indistinguishable_errors).
/// Returns merged errors + mapping from original index to merged index.
inline std::pair<std::vector<Error>, std::vector<size_t>>
merge_indistinguishable_errors(const std::vector<Error>& errors) {
    std::unordered_map<Symptom, size_t, Symptom::hash> merged_index;
    std::vector<Error> merged;
    std::vector<size_t> error_map;
    error_map.reserve(errors.size());

    for (const auto& error : errors) {
        auto it = merged_index.find(error.symptom);
        if (it != merged_index.end()) {
            merged[it->second].likelihood_cost =
                merge_weights(error.likelihood_cost, merged[it->second].likelihood_cost);
            error_map.push_back(it->second);
        } else {
            size_t idx = merged.size();
            merged_index[error.symptom] = idx;
            merged.push_back(error);
            error_map.push_back(idx);
        }
    }
    return {std::move(merged), std::move(error_map)};
}

}  // namespace common

// ── TesseractDecoder ────────────────────────────────────────────────

namespace {
const double INF = std::numeric_limits<double>::infinity();

struct DynBitsetHash {
    size_t operator()(const boost::dynamic_bitset<>& bs) const {
        return boost::hash_value(bs);
    }
};
}  // namespace

class TesseractDecoder {
public:
    TesseractConfig config;

    /// Construct from raw error data.
    TesseractDecoder(
        size_t num_detectors_,
        std::vector<common::Error> errors_,
        TesseractConfig config_)
        : config(std::move(config_))
    {
        if (config.merge_errors) {
            auto [merged, emap] = common::merge_indistinguishable_errors(errors_);
            errors = std::move(merged);
            original_error_map = std::move(emap);
        } else {
            errors = std::move(errors_);
            original_error_map.resize(errors.size());
            std::iota(original_error_map.begin(), original_error_map.end(), 0);
        }

        // Remove zero-probability errors
        {
            std::vector<common::Error> kept;
            std::vector<size_t> remap(errors.size(), std::numeric_limits<size_t>::max());
            size_t kept_idx = 0;
            for (size_t i = 0; i < errors.size(); ++i) {
                if (errors[i].get_probability() > 0) {
                    remap[i] = kept_idx++;
                    kept.push_back(std::move(errors[i]));
                }
            }
            for (size_t& idx : original_error_map) {
                if (idx != std::numeric_limits<size_t>::max()) idx = remap[idx];
            }
            errors = std::move(kept);
        }

        num_detectors = num_detectors_;
        num_errors = errors.size();

        if (config.det_orders.empty()) {
            config.det_orders = tesseract_utils::build_det_orders_bfs(
                num_detectors, errors, 20, 2384753);
        }

        initialize_structures();
    }

    /// Decode, returning predicted error indices in original (pre-merge) numbering.
    std::vector<size_t> decode(const std::vector<uint64_t>& detections) {
        decode_to_errors(detections);
        return predicted_errors_buffer;
    }

private:
    std::vector<common::Error> errors;
    std::vector<size_t> original_error_map;

    size_t num_detectors = 0;
    size_t num_errors = 0;

    std::vector<std::vector<int>> d2e;
    std::vector<std::vector<int>> eneighbors;
    std::vector<std::vector<int>> edets;
    std::vector<ErrorCost> error_costs;

    bool low_confidence_flag = false;
    std::vector<size_t> predicted_errors_buffer;
    std::vector<common::ErrorChainNode> error_chain_arena;

    // Same as TesseractDecoder::initialize_structures in tesseract.cc
    void initialize_structures() {
        d2e.resize(num_detectors);
        edets.resize(num_errors);
        for (size_t ei = 0; ei < num_errors; ++ei) {
            edets[ei] = errors[ei].symptom.detectors;
            for (int d : edets[ei]) {
                d2e[d].push_back(static_cast<int>(ei));
            }
        }
        for (size_t i = 0; i < errors.size(); ++i) {
            error_costs.push_back({
                errors[i].likelihood_cost,
                errors[i].likelihood_cost / static_cast<double>(errors[i].symptom.detectors.size())
            });
        }
        for (size_t d = 0; d < num_detectors; ++d) {
            std::sort(d2e[d].begin(), d2e[d].end(), [this](size_t a, size_t b) {
                return error_costs[a].min_cost < error_costs[b].min_cost;
            });
        }
        eneighbors.resize(num_errors);
        std::vector<boost::dynamic_bitset<>> edets_bs(
            num_errors, boost::dynamic_bitset<>(num_detectors));
        for (size_t ei = 0; ei < num_errors; ++ei) {
            for (int d : edets[ei]) edets_bs[ei][d] = 1;
        }
        for (size_t ei = 0; ei < num_errors; ++ei) {
            boost::dynamic_bitset<> ns(num_detectors, false);
            for (int d : edets[ei]) {
                for (int oei : d2e[d]) ns |= edets_bs[oei];
            }
            ns &= ~edets_bs[ei];
            for (size_t d = ns.find_first(); d != boost::dynamic_bitset<>::npos; d = ns.find_next(d)) {
                eneighbors[ei].push_back(static_cast<int>(d));
            }
        }
    }

    // Same as TesseractDecoder::get_detcost in tesseract.cc
    double get_detcost(size_t d, const std::vector<DetectorCostTuple>& dct) const {
        double min_cost = INF;
        uint32_t min_dc = std::numeric_limits<uint32_t>::max();
        for (int ei : d2e[d]) {
            auto ec = error_costs[ei];
            if (ec.likelihood_cost * min_dc >= min_cost * errors[ei].symptom.detectors.size()) break;
            auto t = dct[ei];
            if (!t.error_blocked) {
                if (ec.likelihood_cost * min_dc < min_cost * t.detectors_count) {
                    min_cost = ec.likelihood_cost;
                    min_dc = t.detectors_count;
                }
            }
        }
        return (min_cost / min_dc) + config.det_penalty;
    }

    // Same as TesseractDecoder::flip_detectors_and_block_errors in tesseract.cc
    void flip_detectors_and_block_errors(
        size_t detector_order, int64_t eci,
        boost::dynamic_bitset<>& detectors,
        std::vector<DetectorCostTuple>& dct) const
    {
        int64_t w = eci;
        while (w != -1) {
            const auto& node = error_chain_arena[w];
            size_t ei = node.error_index;
            for (int oei : d2e[node.min_detector]) {
                dct[oei].error_blocked = 1;
                if (static_cast<size_t>(oei) == ei) break;
            }
            for (int d : edets[ei]) detectors[d] = !detectors[d];
            w = node.parent_idx;
        }
    }

    double cost_from_errors_internal(const std::vector<size_t>& pred) const {
        double total = 0;
        for (size_t ei : pred) total += errors[ei].likelihood_cost;
        return total;
    }

    // Same as TesseractDecoder::decode_to_errors (multi-ordering) in tesseract.cc
    void decode_to_errors(const std::vector<uint64_t>& detections) {
        std::vector<size_t> best;
        double best_cost = std::numeric_limits<double>::max();
        if (config.beam_climbing) {
            int beam = 0, det_ord = 0;
            for (int trial = 0;
                 trial < std::max(config.det_beam + 1, static_cast<int>(config.det_orders.size()));
                 ++trial)
            {
                decode_single(detections, det_ord, beam);
                double c = cost_from_errors_internal(predicted_errors_buffer);
                if (!low_confidence_flag && c < best_cost) { best = predicted_errors_buffer; best_cost = c; }
                beam = (beam + 1) % (config.det_beam + 1);
                det_ord = (det_ord + 1) % static_cast<int>(config.det_orders.size());
            }
        } else {
            for (size_t o = 0; o < config.det_orders.size(); ++o) {
                decode_single(detections, o, config.det_beam);
                double c = cost_from_errors_internal(predicted_errors_buffer);
                if (!low_confidence_flag && c < best_cost) { best = predicted_errors_buffer; best_cost = c; }
            }
        }
        // Map back to original indices
        std::vector<size_t> result;
        result.reserve(best.size());
        for (size_t ei : best) {
            for (size_t orig = 0; orig < original_error_map.size(); ++orig) {
                if (original_error_map[orig] == ei) { result.push_back(orig); break; }
            }
        }
        predicted_errors_buffer = std::move(result);
        low_confidence_flag = best_cost == std::numeric_limits<double>::max();
    }

    // Same as TesseractDecoder::decode_to_errors (single ordering) in tesseract.cc
    void decode_single(
        const std::vector<uint64_t>& detections,
        size_t detector_order, size_t detector_beam)
    {
        predicted_errors_buffer.clear();
        low_confidence_flag = false;
        error_chain_arena.clear();
        error_chain_arena.reserve(config.pqlimit);

        std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
        std::unordered_map<size_t, std::unordered_set<boost::dynamic_bitset<>, DynBitsetHash>> visited;

        boost::dynamic_bitset<> init_det(num_detectors, false);
        std::vector<DetectorCostTuple> init_dct(num_errors);
        for (uint64_t d : detections) {
            if (d >= num_detectors) throw std::runtime_error("Detection >= num_detectors");
            init_det[d] = true;
            for (int ei : d2e[d]) ++init_dct[ei].detectors_count;
        }

        double init_cost = 0;
        for (uint64_t d : detections) init_cost += get_detcost(d, init_dct);
        if (init_cost == INF) { low_confidence_flag = true; return; }

        size_t min_nd = detections.size(), max_nd = min_nd + detector_beam;
        boost::dynamic_bitset<> next_det;
        std::vector<DetectorCostTuple> next_dct;

        pq.push({init_cost, min_nd, 0, -1});
        size_t npush = 1;

        while (!pq.empty()) {
            const Node node = pq.top(); pq.pop();
            if (node.num_dets > max_nd) continue;

            boost::dynamic_bitset<> det = init_det;
            std::vector<DetectorCostTuple> dct(num_errors);
            flip_detectors_and_block_errors(detector_order, node.error_chain_idx, det, dct);

            if (node.num_dets == 0) {
                predicted_errors_buffer.resize(node.depth);
                int64_t w = node.error_chain_idx;
                for (size_t i = 0; i < node.depth; ++i) {
                    predicted_errors_buffer[node.depth - 1 - i] = error_chain_arena[w].error_index;
                    w = error_chain_arena[w].parent_idx;
                }
                return;
            }

            if (config.no_revisit_dets && !visited[node.num_dets].insert(det).second) continue;

            if (node.num_dets < min_nd) {
                min_nd = node.num_dets;
                if (config.no_revisit_dets) {
                    for (size_t i = min_nd + detector_beam + 1; i <= max_nd; ++i) visited[i].clear();
                }
                max_nd = std::min(max_nd, min_nd + detector_beam);
            }

            for (size_t d = 0; d < num_detectors; ++d) {
                if (!det[d]) continue;
                for (int ei : d2e[d]) ++dct[ei].detectors_count;
            }
            next_dct = dct;

            size_t min_detector = std::numeric_limits<size_t>::max();
            for (size_t d = 0; d < num_detectors; ++d) {
                if (det[config.det_orders[detector_order][d]]) {
                    min_detector = config.det_orders[detector_order][d]; break;
                }
            }

            size_t prev_ei = std::numeric_limits<size_t>::max();
            std::vector<double> dc_cache(num_detectors, -1);

            for (int ei : d2e[min_detector]) {
                if (dct[ei].error_blocked) continue;
                if (prev_ei != std::numeric_limits<size_t>::max()) {
                    for (int d : edets[prev_ei]) {
                        int fired = det[d] ? 1 : -1;
                        for (int oei : d2e[d]) next_dct[oei].detectors_count += fired;
                    }
                }
                prev_ei = ei;
                next_det = det;
                next_dct[ei].error_blocked = 1;
                double nc = node.cost + errors[ei].likelihood_cost;
                size_t nnd = node.num_dets;
                for (int d : edets[ei]) {
                    next_det[d] = !next_det[d];
                    int fired = next_det[d] ? 1 : -1;
                    nnd += fired;
                    for (int oei : d2e[d]) next_dct[oei].detectors_count += fired;
                }
                if (nnd > max_nd) continue;
                if (config.no_revisit_dets && visited[nnd].count(next_det)) continue;

                for (int d : edets[ei]) {
                    if (det[d]) {
                        if (dc_cache[d] == -1) dc_cache[d] = get_detcost(d, dct);
                        nc -= dc_cache[d];
                    } else {
                        nc += get_detcost(d, next_dct);
                    }
                }
                for (int od : eneighbors[ei]) {
                    if (!det[od] || !next_det[od]) continue;
                    if (dc_cache[od] == -1) dc_cache[od] = get_detcost(od, dct);
                    nc -= dc_cache[od];
                    nc += get_detcost(od, next_dct);
                }
                if (nc == INF) continue;

                error_chain_arena.push_back({static_cast<size_t>(ei), min_detector, node.error_chain_idx});
                pq.push({nc, nnd, node.depth + 1, static_cast<int64_t>(error_chain_arena.size() - 1)});
                if (++npush > config.pqlimit) { low_confidence_flag = true; return; }
            }
        }
        low_confidence_flag = true;
    }
};

#endif  // TESSERACT_CORE_H_
