func.func @mm32(%A: tensor<32x32xi8>, %B: tensor<32x32xi8>, %C: tensor<32x32xi32>) -> tensor<32x32xi32> {
  %0 = linalg.matmul ins(%A, %B : tensor<32x32xi8>, tensor<32x32xi8>)
                     outs(%C : tensor<32x32xi32>) -> tensor<32x32xi32>
  return %0 : tensor<32x32xi32>
}
