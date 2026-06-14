#include "tesseract_bridge.h"
#include "deq-runtime/src/decoder/tesseract_ffi.rs.h"

#include <algorithm>
#include <vector>

namespace tesseract_bridge {

std::unique_ptr<TesseractDecoderHandle> new_tesseract_decoder(
    uint64_t num_detectors,
    rust::Slice<const uint64_t> edge_vertices,
    rust::Slice<const uint64_t> edge_offsets,
    rust::Slice<const double> edge_probabilities,
    int32_t det_beam,
    bool beam_climbing,
    bool no_revisit_dets,
    bool merge_errors,
    uint64_t pqlimit,
    double det_penalty)
{
    size_t num_edges = edge_probabilities.size();
    std::vector<common::Error> errors;
    errors.reserve(num_edges);
    for (size_t i = 0; i < num_edges; ++i) {
        size_t begin = static_cast<size_t>(edge_offsets[i]);
        size_t end   = static_cast<size_t>(edge_offsets[i + 1]);
        std::vector<int> dets;
        dets.reserve(end - begin);
        for (size_t j = begin; j < end; ++j) {
            dets.push_back(static_cast<int>(edge_vertices[j]));
        }
        std::sort(dets.begin(), dets.end());
        errors.emplace_back(edge_probabilities[i], std::move(dets));
    }

    TesseractConfig config;
    config.det_beam = det_beam;
    config.beam_climbing = beam_climbing;
    config.no_revisit_dets = no_revisit_dets;
    config.merge_errors = merge_errors;
    config.pqlimit = static_cast<size_t>(pqlimit);
    config.det_penalty = det_penalty;

    auto decoder = TesseractDecoder(
        static_cast<size_t>(num_detectors),
        std::move(errors),
        std::move(config));

    return std::make_unique<TesseractDecoderHandle>(std::move(decoder));
}

rust::Vec<uint64_t> decode_to_errors(
    TesseractDecoderHandle& handle,
    rust::Slice<const uint64_t> detections)
{
    std::vector<uint64_t> dets(detections.begin(), detections.end());
    auto indices = handle.decoder.decode(dets);

    rust::Vec<uint64_t> result;
    result.reserve(indices.size());
    for (size_t idx : indices) {
        result.push_back(static_cast<uint64_t>(idx));
    }
    return result;
}

}  // namespace tesseract_bridge
