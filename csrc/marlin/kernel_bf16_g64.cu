// Turing Engine: Marlin kernels for bf16 activations, AWQ-INT4 weights, groups of 64 (see kernel_configs.h).
// One file per (dtype, group) so the build compiles them in parallel, as vLLM's generated sm80_kernel_*.cu files.
#include "kernel.h"
#include "marlin_template.h"
#include "kernel_configs.h"

namespace MARLIN_NAMESPACE_NAME {
TURING_MARLIN_SHAPES(TURING_MARLIN_INSTANTIATE, kBFloat16, 4)
}  // namespace MARLIN_NAMESPACE_NAME
