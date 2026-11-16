import os
import platform
import sys
import glob
import platform
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
]

setup(
    name="turing-engine",
    version="1.7.0",

    author="Ishan Gupta",
    author_email="support@intutic.ai",
    description="Turing Engine: Subspace-Compressed High-Performance LLM Inference & Serving Runtime",
    packages=find_packages(include=["turing", "turing.*"]),  # not the repo's scripts/ or integrations/
    ext_modules=ext_modules,
    cmdclass={"build_ext": StripBuildExt},
    entry_points={
        "console_scripts": [
            "turing=turing.cli:main",
        ],
    },
    python_requires=">=3.9",
)

