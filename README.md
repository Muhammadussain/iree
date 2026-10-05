# IREE — SpacemiT IME Integration

## Branches

| Branch | Description |
|---|---|
| `phase2-mmt4d-ukernel` | Phase 2: mmt4d ukernel |
| `phase3-iree-integration` | Phase 3: RISCVIME plugin |

## Prerequisites
First build custom LLVM:
https://github.com/Muhammadussain/llvm-project/tree/phase3-riscvime-dialect

## Clone
git clone --recursive https://github.com/Muhammadussain/iree.git
cd iree
git checkout phase3-iree-integration
git submodule update --init --recursive

cd third_party/llvm-project
git fetch origin phase3-riscvime-dialect
git checkout -b phase3-riscvime-dialect FETCH_HEAD

## Build IREE
export IREE_SRC=$HOME/iree
export IREE_BUILD=$HOME/iree-build-tcp
export LLVM_BUILD=$HOME/llvm-xsmt-build

rm -rf $IREE_BUILD
mkdir -p $IREE_BUILD
cd $IREE_BUILD

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm \
  -DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir \
  -DCMAKE_INSTALL_PREFIX=$IREE_BUILD/install \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  $IREE_SRC

cmake --build . --target iree-compile -j$(nproc)
