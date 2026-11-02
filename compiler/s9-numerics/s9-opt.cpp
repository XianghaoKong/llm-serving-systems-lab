// Copyright (c) 2026 S9 numerics contributors. MIT license (repository LICENSE).
// The arithmetic schedule follows PyTorch v2.8.0 Reduce.cuh / MeanOps.
// This is explicit contract lowering, NOT a reassociation of arbitrary addf.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <algorithm>
#include <array>
#include <vector>

using namespace mlir;

namespace {
// Semantics: contiguous, aligned FP32 mean of 1536 elements using the CUDA
// PyTorch 2.8.0 vectorized-input schedule for `rows` outputs (1..4096).
// The operand is already squared; BF16 conversion/squaring remain outside.
class AtenMeanOp : public Op<AtenMeanOp, OpTrait::OneOperand,
                             OpTrait::OneResult, OpTrait::ZeroRegions> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "s9.aten_mean"; }
  static ArrayRef<StringRef> getAttributeNames() {
    static const StringRef names[] = {"contract", "rows"};
    return names;
  }
  LogicalResult verify() {
    auto v = dyn_cast<VectorType>(getOperand().getType());
    if (!v || v.isScalable() || v.getRank() != 1 || v.getDimSize(0) != 1536 ||
        !v.getElementType().isF32() || !getOperation()->getResult(0).getType().isF32())
      return emitOpError("requires vector<1536xf32> -> f32");
    auto rows = (*this)->getAttrOfType<IntegerAttr>("rows");
    if (!rows || !rows.getType().isInteger(64) || rows.getInt() < 1 || rows.getInt() > 4096)
      return emitOpError("requires rows : i64 in [1,4096]");
    auto contract = (*this)->getAttrOfType<StringAttr>("contract");
    if (!contract || contract.getValue() != "pytorch-2.8.0-cuda-f32-mean-v1")
      return emitOpError("requires the explicit PyTorch 2.8.0 CUDA mean contract");
    // Never silently accept arbitrary floating-point permissions on this op.
    for (auto attr : (*this)->getAttrs())
      if (attr.getName() != "rows" && attr.getName() != "contract")
        return emitOpError("unsupported attribute: ") << attr.getName();
    return success();
  }
};

// Collective GPU contract: all 32 lanes of one warp call with the same aligned
// BF16 row address. Returns the same FP32 mean in every lane. Squaring is part
// of this op; unlike the scalar oracle, no full-row vector is materialized.
class AtenMeanWarpOp : public Op<AtenMeanWarpOp, OpTrait::OneOperand,
                                 OpTrait::OneResult, OpTrait::ZeroRegions> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "s9.aten_mean_warp"; }
  static ArrayRef<StringRef> getAttributeNames() {
    static const StringRef names[] = {"contract", "rows"};
    return names;
  }
  LogicalResult verify() {
    if (!getOperand().getType().isInteger(64) || !getOperation()->getResult(0).getType().isF32())
      return emitOpError("requires i64 row address -> f32");
    auto rows = (*this)->getAttrOfType<IntegerAttr>("rows");
    if (!rows || !rows.getType().isInteger(64) || rows.getInt()<1 || rows.getInt()>4096)
      return emitOpError("requires rows : i64 in [1,4096]");
    auto contract = (*this)->getAttrOfType<StringAttr>("contract");
    if (!contract || contract.getValue() != "pytorch-2.8.0-cuda-bf16-square-mean-warp32-v1")
      return emitOpError("requires the explicit full-warp PyTorch 2.8.0 contract");
    for (auto attr : (*this)->getAttrs())
      if (attr.getName() != "rows" && attr.getName() != "contract")
        return emitOpError("unsupported attribute: ") << attr.getName();
    return success();
  }
};

class S9Dialect : public Dialect {
public:
  explicit S9Dialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<S9Dialect>()) {
    addOperations<AtenMeanOp, AtenMeanWarpOp>();
  }
  static StringRef getDialectNamespace() { return "s9"; }
};

static int floorPow2(int n) {
  int p = 1;
  while (p <= n / 2) p *= 2;
  return p;
}

class LowerMean : public OpRewritePattern<AtenMeanOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMeanOp op, PatternRewriter &rw) const override {
    // This reproduces ReduceConfig::set_block_dimension for width 1536,
    // input_vec_size=4, output_vec_size=1, CUDA warp size=32, max threads=512.
    // For the supported shape/range there is no block-Y or CTA reduction.
    const int rows = op->getAttrOfType<IntegerAttr>("rows").getInt();
    const int blockHeight = std::min(floorPow2(rows), 16);
    const int blockWidth = std::min(256, 512 / blockHeight);
    const int width = 1536;
    Location loc = op.getLoc();
    Value zero = rw.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), rw.getF32Type());
    auto add = [&](Value a, Value b) -> Value {
      // No reassoc, contract, nsz, or other fast-math flags.
      return rw.create<arith::AddFOp>(loc, a, b);
    };
    std::vector<Value> lanes;
    lanes.reserve(blockWidth);
    for (int lane = 0; lane < blockWidth; ++lane) {
      std::array<Value, 4> accum = {zero, zero, zero, zero};
      for (int base = lane * 4; base < width; base += blockWidth * 4) {
        for (int component = 0; component < 4; ++component) {
          Value value = rw.create<vector::ExtractOp>(loc, op.getOperand(),
                                                     int64_t(base + component));
          accum[component] = add(accum[component], value);
        }
      }
      // Match the left-to-right four-accumulator combine in Reduce.cuh.
      Value value = accum[0];
      for (int component = 1; component < 4; ++component)
        value = add(value, accum[component]);
      lanes.push_back(value);
    }
    // Shared-memory portion of block_x_reduce, before the first warp remains.
    for (int offset = blockWidth / 2; offset >= 32; offset /= 2)
      for (int lane = 0; lane < offset; ++lane)
        lanes[lane] = add(lanes[lane], lanes[lane + offset]);
    // CUDA warp_shfl_down uses offsets 1,2,4,8,16 (ascending).
    // Each shuffle observes the previous state, not partially updated lanes.
    for (int offset = 1; offset < 32; offset *= 2) {
      auto previous = lanes;
      for (int lane = 0; lane < 32; ++lane) {
        int source = lane + offset < 32 ? lane + offset : lane;
        lanes[lane] = add(previous[lane], previous[source]);
      }
    }
    Value factor = rw.create<arith::ConstantFloatOp>(
        loc, APFloat(1.0f / 1536.0f), rw.getF32Type());
    Value result = rw.create<arith::MulFOp>(loc, lanes[0], factor);
    rw.replaceOp(op, result);
    return success();
  }
};

class LowerWarpMean : public OpRewritePattern<AtenMeanWarpOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMeanWarpOp op, PatternRewriter &rw) const override {
    int rows = op->getAttrOfType<IntegerAttr>("rows").getInt();
    int blockWidth = std::min(256, 512 / std::min(floorPow2(rows), 16));
    Location loc = op.getLoc();
    auto i32 = [&](int v) -> Value { return rw.create<arith::ConstantIntOp>(loc,v,32); };
    auto add = [&](Value a, Value b) -> Value { return rw.create<arith::AddFOp>(loc,a,b); };
    auto assembly = [&](Type ty, ValueRange values, StringRef code, StringRef constraints) -> Value {
      // Side effects prevent moving/duplicating a warp collective as a pure
      // elementwise expression. The enclosing call is also marked impure.
      return rw.create<LLVM::InlineAsmOp>(loc,ty,values,code,constraints,
          true,false,LLVM::AsmDialectAttr(),ArrayAttr()).getRes();
    };
    Value lane = assembly(rw.getI32Type(),{},"mov.u32 $0, %laneid;","=r");
    Value laneBase = rw.create<arith::MulIOp>(loc,lane,i32(4));
    Value zero = rw.create<arith::ConstantFloatOp>(loc,APFloat(0.0f),rw.getF32Type());
    auto vb = VectorType::get({4},rw.getBF16Type());
    auto vf = VectorType::get({4},rw.getF32Type());
    std::vector<Value> groups;
    // Logical lane g*32+lane has exactly the ATen four-accumulator schedule.
    // Independent logical lanes share a physical lane, without reassociation.
    for (int g=0; g<blockWidth/32; ++g) {
      std::array<Value,4> accum = {zero,zero,zero,zero};
      for (int base=g*128; base<1536; base+=blockWidth*4) {
        Value index = rw.create<arith::AddIOp>(loc,laneBase,i32(base));
        Value wide = rw.create<arith::ExtUIOp>(loc,rw.getI64Type(),index);
        Value two = rw.create<arith::ConstantIntOp>(loc,2,64);
        Value offset = rw.create<arith::MulIOp>(loc,wide,two);
        Value address = rw.create<arith::AddIOp>(loc,op.getOperand(),offset);
        Value ptr = rw.create<LLVM::IntToPtrOp>(loc,LLVM::LLVMPointerType::get(rw.getContext(),1),address);
        Value packed = rw.create<LLVM::LoadOp>(loc,vb,ptr,8);
        Value values = rw.create<arith::ExtFOp>(loc,vf,packed);
        for (int c=0;c<4;++c) {
          Value x = rw.create<vector::ExtractOp>(loc,values,int64_t(c));
          Value sq = rw.create<arith::MulFOp>(loc,x,x);
          accum[c] = add(accum[c],sq);
        }
      }
      Value sum=accum[0];
      for(int c=1;c<4;++c) sum=add(sum,accum[c]);
      groups.push_back(sum);
    }
    // Same cross-warp tree as ATen, implemented in lane-local registers.
    for(int offset=groups.size()/2;offset>0;offset/=2)
      for(int g=0;g<offset;++g) groups[g]=add(groups[g],groups[g+offset]);
    Value sum=groups[0];
    for(int offset=1;offset<32;offset*=2) {
      std::string code="shfl.sync.down.b32 $0, $1, "+std::to_string(offset)+", 31, -1;";
      Value other=assembly(rw.getF32Type(),{sum},code,"=f,f");
      sum=add(sum,other);
    }
    Value total=assembly(rw.getF32Type(),{sum},"shfl.sync.idx.b32 $0, $1, 0, 31, -1;","=f,f");
    Value factor=rw.create<arith::ConstantFloatOp>(loc,APFloat(1.0f/1536.0f),rw.getF32Type());
    rw.replaceOpWithNewOp<arith::MulFOp>(op,total,factor);
    return success();
  }
};

class LowerMeanPass : public PassWrapper<LowerMeanPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMeanPass)
  StringRef getArgument() const final { return "s9-lower-aten-mean"; }
  StringRef getDescription() const final {
    return "Lower an explicit S9 ATen mean contract to ordered FP32 arithmetic";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, vector::VectorDialect, LLVM::LLVMDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerMean, LowerWarpMean>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<S9Dialect, arith::ArithDialect, func::FuncDialect,
                  vector::VectorDialect, LLVM::LLVMDialect>();
  PassRegistration<LowerMeanPass>();
  return asMainReturnCode(MlirOptMain(argc, argv, "S9 numeric contract optimizer\n", registry));
}
