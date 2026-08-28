import os
import sys
import sysconfig
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class get_pybind_include:
    def __str__(self):
        import pybind11
        return pybind11.get_include()

if sys.platform == "win32":
    # Windows MSVC compiler flags
    extra_compile_args = [
        "/std:c++20",
        "/O2",
        "/arch:AVX2",
        "/EHsc",
        "/bigobj",
        "/utf-8",
        "/DNOMINMAX",
        "/D_USE_MATH_DEFINES",
        "/D__restrict__=__restrict",
    ]
    extra_link_args = []
elif sys.platform == "darwin":
    # macOS Apple Clang flags
    extra_compile_args = [
        "-std=c++20",
        "-O3",
        "-flto",
        "-fvisibility=hidden",
        "-stdlib=libc++",
        "-mmacosx-version-min=10.15",
    ]
    extra_link_args = [
        "-flto",
    ]
else:
    # Linux GCC / Clang flags
    extra_compile_args = [
        "-std=c++20",
        "-O3",
        "-flto",
        "-fvisibility=hidden",
        "-mavx2",
        "-mfma",
    ]
    extra_link_args = [
        "-flto",
        "-Wl,--strip-all",
    ]

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
    version="0.3.0",
    author="Ishan Gupta",


    author_email="support@intutic.ai",
    description="Turing Engine: Subspace-Compressed High-Performance LLM Inference & Serving Runtime",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": StripBuildExt},
    entry_points={
        "console_scripts": [
            "turing=turing.cli:main",
        ],
    },
    python_requires=">=3.9",
)

