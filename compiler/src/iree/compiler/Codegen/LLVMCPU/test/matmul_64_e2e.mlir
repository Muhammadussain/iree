func.func @matmul_64x64x64(
    %A: tensor<64x64xi8>,
    %B: tensor<64x64xi8>,
    %C: tensor<64x64xi32>) -> tensor<64x64xi32> {
  %0 = linalg.matmul ins(%A, %B : tensor<64x64xi8>, tensor<64x64xi8>)
                     outs(%C : tensor<64x64xi32>) -> tensor<64x64xi32>
  return %0 : tensor<64x64xi32>
}
