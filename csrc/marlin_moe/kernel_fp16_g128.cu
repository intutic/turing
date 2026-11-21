// Turing Engine: MoE Marlin kernels for fp16 activations, AWQ-INT4 weights, groups of 128: the shapes of
// csrc/marlin/kernel_configs.h (vLLM's generate_kernels.py emits the same list for its MoE kernels).
// One file per (dtype, group) so the build compiles them in parallel.
#include "kernel.h"
#include "marlin_template.h"
#include "kernel_configs.h"

namespace MARLIN_NAMESPACE_NAME {
TURING_MARLIN_SHAPES(TURING_MARLIN_INSTANTIATE, kFloat16, 8)
}  // namespace MARLIN_NAMESPACE_NAME
