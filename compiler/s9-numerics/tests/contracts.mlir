// CHECK-LABEL: func.func @qwen_mean
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: return
func.func @qwen_mean(%x: vector<1536xf32>) -> f32 {
  %v = "s9.aten_mean"(%x) {rows = 346 : i64, contract = "pytorch-2.8.0-cuda-f32-mean-v1"} : (vector<1536xf32>) -> f32
  return %v : f32
}

// CHECK-LABEL: func.func @rounding_barrier
// CHECK: arith.truncf
// CHECK-NEXT: {{.*}}arith.extf
func.func @rounding_barrier(%x: f32) -> f32 {
  %a = arith.truncf %x : f32 to bf16
  %b = arith.extf %a : bf16 to f32
  return %b : f32
}

// CHECK-LABEL: func.func @ordinary_reduction
// CHECK: vector.reduction <add>
func.func @ordinary_reduction(%x: vector<1536xf32>) -> f32 {
  %v = vector.reduction <add>, %x : vector<1536xf32> into f32
  return %v : f32
}
