// RUN: iree-compile %s \
// RUN:   --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-triple=riscv64 \
// RUN:   --iree-llvmcpu-target-abi=lp64d \
// RUN:   --iree-llvmcpu-target-cpu=spacemit-x60 \
// RUN:   --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+v,+zvl256b,+xsmtvdot" \
// RUN:   --iree-llvmcpu-enable-ukernels=none \
// RUN:   -o /tmp/test.vmfb
// RUN: iree-run-module --module=/tmp/test.vmfb \
// RUN:   --function=matmul \
// RUN:   --input=64x64xi8=1 \
// RUN:   --input=64x64xi8=1 \
// RUN:   --input=64x64xi32=0 | FileCheck %s
//
// CHECK: 64

func.func @matmul(%lhs: tensor<64x64xi8>,
                  %rhs: tensor<64x64xi8>,
                  %acc: tensor<64x64xi32>) -> tensor<64x64xi32> {
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<64x64xi8>, tensor<64x64xi8>)
                     outs(%acc : tensor<64x64xi32>) -> tensor<64x64xi32>
  return %0 : tensor<64x64xi32>
}