# ============================================================
# Custom RISC-V XSMT Toolchain File
# ============================================================
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Custom LLVM as compiler (XSMT support!)
set(CMAKE_C_COMPILER $ENV{CUSTOM_CLANG})
set(CMAKE_CXX_COMPILER $ENV{CUSTOM_CLANGXX})
set(CMAKE_C_COMPILER_TARGET riscv64-unknown-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET riscv64-unknown-linux-gnu)

# ISA + ABI + GCC runtime
set(_XSMT_FLAGS "-march=rv64gc_xsmtvdot -mabi=lp64d")
set(_GCC_PATH "--gcc-toolchain=$ENV{RISCV_TOOLCHAIN_ROOT}")

set(CMAKE_C_FLAGS_INIT   "${_XSMT_FLAGS} ${_GCC_PATH}")
set(CMAKE_CXX_FLAGS_INIT "${_XSMT_FLAGS} ${_GCC_PATH}")

# Linker
set(_LINK_FLAGS "${_GCC_PATH} -fuse-ld=$ENV{RISCV_TOOLCHAIN_ROOT}/bin/ld.lld")
set(CMAKE_EXE_LINKER_FLAGS_INIT    "${_LINK_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_LINK_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${_LINK_FLAGS}")

# Sysroot
set(CMAKE_SYSROOT $ENV{SYSROOT})
set(CMAKE_FIND_ROOT_PATH $ENV{SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Build tools
set(CMAKE_AR     $ENV{CUSTOM_LLVM_AR})
set(CMAKE_RANLIB $ENV{CUSTOM_LLVM_RANLIB})
set(CMAKE_STRIP  $ENV{CUSTOM_LLVM_STRIP})
