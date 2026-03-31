import hashlib
import importlib
import os
from pathlib import Path
import subprocess

_CPU_EXT = None
_CPU_EXT_FAILED = False

_SRC_PATH = Path(__file__).with_name("_cpu_kernels.cpp")
_BUILD_DIR = Path(__file__).parent / "_build"


def _source_hash() -> str:
    """Short hash of the C++ source for cache-busting the module name."""
    h = hashlib.md5(_SRC_PATH.read_bytes()).hexdigest()[:8]
    return h


def _try_cached_import(name: str):
    """Try to import an already-built extension without recompilation."""
    build_dir = _BUILD_DIR / name
    if not build_dir.exists():
        return None
    so_files = list(build_dir.glob("*.so")) + list(build_dir.glob("*.dylib"))
    if not so_files:
        return None
    so_path = so_files[0]
    spec = importlib.util.spec_from_file_location(name, str(so_path))
    if spec is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _cpu_ext():
    global _CPU_EXT, _CPU_EXT_FAILED
    if _CPU_EXT is not None:
        return _CPU_EXT
    if _CPU_EXT_FAILED:
        return None

    name = f"diff_utils_cpu_{_source_hash()}"

    # Fast path: try loading already-compiled .so
    cached = _try_cached_import(name)
    if cached is not None:
        _CPU_EXT = cached
        return _CPU_EXT

    # Slow path: compile
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

    include_paths = []
    try:
        import pybind11

        include_paths = [pybind11.get_include()]
    except Exception:
        pass

    sdk_flag = []
    sdkroot = os.environ.get("SDKROOT", "")
    if sdkroot:
        sdk_flag = [f"-isysroot{sdkroot}"]

    build_dir = _BUILD_DIR / name
    build_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torch.utils.cpp_extension import load

        _CPU_EXT = load(
            name=name,
            sources=[str(_SRC_PATH)],
            build_directory=str(build_dir),
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
