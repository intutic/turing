#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace turing {

/**
 * Native C++20 Zero-Overhead Binary Tensor Serializer for Cross-Device Streaming.
 */
inline std::vector<uint8_t> serialize_tensor_int8_cpp(
    const float* __restrict__ data,
    const std::vector<uint32_t>& shape
) {
    uint32_t total_elements = 1;
    for (uint32_t dim : shape) total_elements *= dim;

    // 1. Compute scale: max(|data|) / 127.0
    float max_abs = 1e-8f;
    for (uint32_t i = 0; i < total_elements; ++i) {
        max_abs = std::max(max_abs, std::abs(data[i]));
    }
    float scale = max_abs / 127.0f;
    float inv_scale = 1.0f / scale;

    // 2. Header layout:
    // payload_len (4B) + dtype_code (1B) + scale (4B) + ndim (4B) + shape (4B * ndim)
    uint32_t ndim = static_cast<uint32_t>(shape.size());
    uint32_t header_len = 1 + 4 + 4 + (4 * ndim);
    uint32_t payload_len = header_len + total_elements;
    uint32_t total_msg_len = 4 + payload_len;

    std::vector<uint8_t> buffer(total_msg_len);
    uint8_t* ptr = buffer.data();

    // Payload len
    std::memcpy(ptr, &payload_len, 4); ptr += 4;
    // Dtype code (1 = INT8)
    uint8_t dtype_code = 1;
    *ptr = dtype_code; ptr += 1;
    // Scale
    std::memcpy(ptr, &scale, 4); ptr += 4;
    // Ndim
    std::memcpy(ptr, &ndim, 4); ptr += 4;
    // Shape
    for (uint32_t s : shape) {
        std::memcpy(ptr, &s, 4); ptr += 4;
    }

    // 3. INT8 Quantized payload
    for (uint32_t i = 0; i < total_elements; ++i) {
        float q_val = std::round(data[i] * inv_scale);
        q_val = std::max(-128.0f, std::min(127.0f, q_val));
        *ptr++ = static_cast<uint8_t>(static_cast<int8_t>(q_val));
    }

    return buffer;
}

inline void deserialize_tensor_int8_cpp(
    const uint8_t* buffer,
    float* __restrict__ output_data,
    float& scale_out,
    std::vector<uint32_t>& shape_out
) {
    const uint8_t* ptr = buffer;
    uint8_t dtype_code = *ptr; ptr += 1;
    (void)dtype_code;

    std::memcpy(&scale_out, ptr, 4); ptr += 4;
    uint32_t ndim = 0;
    std::memcpy(&ndim, ptr, 4); ptr += 4;

    shape_out.resize(ndim);
    uint32_t total_elements = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
        std::memcpy(&shape_out[d], ptr, 4); ptr += 4;
        total_elements *= shape_out[d];
    }

    for (uint32_t i = 0; i < total_elements; ++i) {
        int8_t val = static_cast<int8_t>(*ptr++);
        output_data[i] = static_cast<float>(val) * scale_out;
    }
}

} // namespace turing
