#ifndef TESSERACT_BRIDGE_H_
#define TESSERACT_BRIDGE_H_

#include "rust/cxx.h"
#include "tesseract_core.h"

#include <memory>
#include <vector>

namespace tesseract_bridge {

class TesseractDecoderHandle {
public:
    TesseractDecoder decoder;

    explicit TesseractDecoderHandle(TesseractDecoder dec)
        : decoder(std::move(dec)) {}
};

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
    double det_penalty);

rust::Vec<uint64_t> decode_to_errors(
    TesseractDecoderHandle& handle,
    rust::Slice<const uint64_t> detections);

}  // namespace tesseract_bridge

#endif  // TESSERACT_BRIDGE_H_
