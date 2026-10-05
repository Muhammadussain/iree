func.func @matmul_512x512x512(
    %A: tensor<512x512xi8>,
    %B: tensor<512x512xi8>,
    %C: tensor<512x512xi32>) -> tensor<512x512xi32> {
  %0 = linalg.matmul ins(%A, %B : tensor<512x512xi8>, tensor<512x512xi8>)
                     outs(%C : tensor<512x512xi32>) -> tensor<512x512xi32>
  return %0 : tensor<512x512xi32>
}
