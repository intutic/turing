import os
import platform
import sys
import glob
import platform
import re
import sysconfig
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class get_pybind_include:
    def __str__(self):
        import pybind11
        return pybind11.get_include()

# Instruction set by target CPU, not OS: x86_64 (Linux, Intel macOS, Windows) gets AVX2 + FMA + F16C;
# arm64 / aarch64 gets NEON, which every 64-bit ARM core has, so it needs no flag.
IS_X86 = platform.machine().lower() in ("x86_64", "amd64", "i386", "i686")

if sys.platform == "win32":
    # Windows MSVC compiler flags (/GL + /LTCG: whole-program optimisation, MSVC's LTO)
    extra_compile_args = [
        "/std:c++20",
        "/O2",
        "/GL",
        "/EHsc",
        "/bigobj",
        "/utf-8",
        "/DNOMINMAX",
        "/D_USE_MATH_DEFINES",
        "/D__restrict__=__restrict",
    ] + (["/arch:AVX2"] if IS_X86 else [])
    extra_link_args = ["/LTCG"]
elif sys.platform == "darwin":
    # macOS Apple Clang flags
    extra_compile_args = [
        "-std=c++20",
        "-O3",
        "-flto",
        "-fvisibility=hidden",
        "-stdlib=libc++",
        "-mmacosx-version-min=10.15",
    ] + (["-mavx2", "-mfma", "-mf16c"] if IS_X86 else [])
    extra_link_args = [
        "-flto",
    ]
else:
    # Linux GCC / Clang flags (AVX2/FMA on x86_64 only; aarch64 builds the NEON paths)
    extra_compile_args = [
        "-std=c++20",
        "-O3",
        "-flto",
        "-fvisibility=hidden",
    ] + (["-mavx2", "-mfma", "-mf16c"] if IS_X86 else [])
    extra_link_args = [
        "-flto",
        "-Wl,--strip-all",
    ]


def has_liburing() -> bool:
    """liburing's header and library are installed (Linux): build the io_uring reader backend."""
    if not sys.platform.startswith("linux"):
        return False
    import ctypes.util
    include_dirs = [sysconfig.get_config_var("INCLUDEDIR") or "", "/usr/include", "/usr/local/include"]
    header = any(os.path.exists(os.path.join(d, "liburing.h")) for d in include_dirs if d)
    return header and ctypes.util.find_library("uring") is not None


define_macros, libraries = [], []
if has_liburing():
    define_macros.append(("TURING_HAVE_LIBURING", "1"))
    libraries.append("uring")

def marlin_extension():
    """vLLM's Marlin W4A16 GEMMs (csrc/marlin/ and csrc/marlin_moe/, used by turing.kernels.marlin and
    turing.kernels.marlin_moe), as a list of extensions: only with TURING_BUILD_MARLIN=1 and nvcc,
    so the default install needs no CUDA toolkit. TORCH_CUDA_ARCH_LIST (e.g. "8.9") replaces the default sm_80/86/89."""
    if os.environ.get("TURING_BUILD_MARLIN") != "1":
        return None
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension
    if CUDA_HOME is None:
        print("[!] TURING_BUILD_MARLIN=1 but no CUDA toolkit (nvcc) was found: not building the Marlin extension")
        return None
    out = subprocess.run([os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"], capture_output=True, text=True).stdout
    release = re.search(r"release (\d+)\.(\d+)", out)
    nvcc = ["-O3", "-std=c++17", "--threads=3", "-Xcompiler=-fvisibility=hidden",
            # vLLM drops torch's no-half-operator defines for Marlin (cmake/utils.cmake)
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"]
    if release and tuple(map(int, release.groups())) >= (12, 8):
        nvcc.append("-static-global-template-stub=false")  # the selector (marlin.cu) uses kernels other files instantiate
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        nvcc += [f"-gencode=arch=compute_{a},code=sm_{a}" for a in (80, 86, 89)]
    here = os.path.dirname(os.path.abspath(__file__))
    shared = sorted(glob.glob("csrc/marlin/*.h") + glob.glob("csrc/marlin/*.cuh") + glob.glob("csrc/marlin/core/*"))
    args = {"cxx": ["-O3", "-std=c++17", "-fvisibility=hidden"], "nvcc": nvcc}
    return [
        CUDAExtension("turing.turing_marlin", sources=sorted(glob.glob("csrc/marlin/*.cu")), depends=shared,
                      include_dirs=[os.path.join(here, "csrc", "marlin")], extra_compile_args=args),
        # the MoE kernels (csrc/marlin_moe/, used by turing.kernels.marlin_moe): their own template over the same
        # shared headers, which csrc/marlin/ holds
        CUDAExtension("turing.turing_marlin_moe", sources=sorted(glob.glob("csrc/marlin_moe/*.cu")),
                      depends=shared + sorted(glob.glob("csrc/marlin_moe/*.h")),
                      include_dirs=[os.path.join(here, "csrc", "marlin_moe"), os.path.join(here, "csrc", "marlin")],
                      extra_compile_args=args),
    ]


MARLIN = marlin_extension()
if MARLIN is not None:
    from torch.utils.cpp_extension import BuildExtension as build_ext  # compiles the .cu sources with nvcc


class StripBuildExt(build_ext):
    """
    Custom build_ext command that strips binary debug and function symbol tables
    to protect mathematical proprietary algorithms from reverse engineering.
    """
    def run(self):
        super().run()
        for ext in self.extensions:
            ext_path = self.get_ext_fullpath(ext.name)
            if os.path.exists(ext_path):
                try:
                    if sys.platform == "darwin":
                        subprocess.run(["strip", "-x", ext_path], check=False)
                    elif sys.platform == "win32":
                        pass  # MSVC /O2 release builds omit debug symbols by default
                    else:
                        subprocess.run(["strip", "--strip-all", ext_path], check=False)
                    print(f"[+] Successfully prepared binary artifact: {ext_path}")
                except Exception as e:
                    print(f"[!] Note: Symbol stripping: {e}")

ext_modules = [
    Extension(
        "turing.turing_csrc",
        sources=["csrc/pybind_bindings.cpp"],
        depends=sorted(glob.glob("csrc/*.hpp")),  # header edits rebuild the extension
        define_macros=define_macros,
        libraries=libraries,
        include_dirs=[
            get_pybind_include(),
            "csrc",
            sysconfig.get_path("include"),
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
] + (MARLIN or [])

setup(
    name="turing-engine",
    version="2.1.0",

    author="Ishan Gupta",
    author_email="support@intutic.ai",
    description="LLM inference server for KV-cache and quantization research: int8/fp8/int4 KV caches that keep the layers low-bit storage would break in full precision, 4-bit GPTQ, EAGLE-3 speculative decoding, HuggingFace parity tests on 16 model types, and OpenAI, Anthropic and Ollama APIs on one port",
    packages=find_packages(include=["turing", "turing.*"]),  # not the repo's scripts/ or integrations/
    ext_modules=ext_modules,
    cmdclass={"build_ext": StripBuildExt},
    entry_points={
        "console_scripts": [
            "turing=turing.cli:main",
        ],
    },
    python_requires=">=3.10",
)

