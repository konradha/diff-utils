from __future__ import annotations

import os
from pathlib import Path
import subprocess

_CPU_EXT = None
_CPU_EXT_FAILED = False

def _cpu_ext():
    global _CPU_EXT, _CPU_EXT_FAILED
    if _CPU_EXT is None and not _CPU_EXT_FAILED:
        backup = dict(os.environ)
        for k in list(os.environ.keys()):
            if k.startswith("CONDA_") or k in {
                "CC",
                "CXX",
                "CPP",
                "CFLAGS",
                "CXXFLAGS",
                "CPPFLAGS",
                "LDFLAGS",
                "LDFLAGS_LD",
                "CMAKE_ARGS",
                "CMAKE_PREFIX_PATH",
                "SDKROOT",
                "MACOSX_DEPLOYMENT_TARGET",
            }:
                os.environ.pop(k, None)
        os.environ["CC"] = "/usr/bin/clang"
        os.environ["CXX"] = "/usr/bin/clang++"
        try:
            os.environ["SDKROOT"] = subprocess.check_output(
                ["xcrun", "--show-sdk-path"], text=True
            ).strip()
        except Exception:
            pass
        src = Path(__file__).with_name("_cpu_kernels.cpp")
        include_paths = []
        try:
            import pybind11

            include_paths = [pybind11.get_include()]
        except Exception:
            include_paths = []

        sdk_flag = []
        sdkroot = os.environ.get("SDKROOT", "")
        if sdkroot:
            sdk_flag = [f"-isysroot{sdkroot}"]

        try:
            from torch.utils.cpp_extension import load

            _CPU_EXT = load(
                name="diff_utils_cpu_kernels_v1",
                sources=[str(src)],
                extra_cflags=[
                    "-O3",
                    "-DNDEBUG",
                    "-march=native",
                    "-mtune=native",
                    "-fstrict-aliasing",
                    "-fno-math-errno",
                    "-fno-trapping-math",
                    "-funroll-loops",
                ]
                + sdk_flag,
                extra_include_paths=include_paths,
                verbose=False,
            )
        except Exception:
            _CPU_EXT = None
            _CPU_EXT_FAILED = True
        finally:
            os.environ.clear()
            os.environ.update(backup)
    return _CPU_EXT

def _tensor_has_storage(x) -> bool:
    try:
        _ = x.data_ptr()
        return True
    except Exception:
        return False
