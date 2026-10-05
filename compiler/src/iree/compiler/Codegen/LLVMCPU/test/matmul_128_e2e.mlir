func.func @matmul_128x128x128(
    %A: tensor<128x128xi8>,
    %B: tensor<128x128xi8>,
    %C: tensor<128x128xi32>) -> tensor<128x128xi32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x128xi8>, tensor<128x128xi8>)
                     outs(%C : tensor<128x128xi32>) -> tensor<128x128xi32>
  return %0 : tensor<128x128xi32>
}
