/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include <memory>
#include <utility>
#include <stdexcept>
#include <string>

using namespace mlir;

namespace {
    struct PartialRead : public OpConversionPattern<daphne::ReadOp> {

        using OpConversionPattern<daphne::ReadOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(daphne::ReadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
            for (const auto& u : (&op)->getResult().getUsers()) {
                if (auto rOp = llvm::dyn_cast<mlir::daphne::SliceTensorOp>(u)) {
                    auto ops = u->getOperands();
                    auto uint32Type = rewriter.getIntegerType(32, false);
                    auto v1 = rewriter.create<daphne::CastOp>(ops[1].getLoc(), uint32Type, ops[1]);
                    auto v2 = rewriter.create<daphne::CastOp>(ops[2].getLoc(), uint32Type, ops[2]);
                    auto v3 = rewriter.create<daphne::CastOp>(ops[3].getLoc(), uint32Type, ops[3]);
                    auto v4 = rewriter.create<daphne::CastOp>(ops[4].getLoc(), uint32Type, ops[4]);
                    auto v5 = rewriter.create<daphne::CastOp>(ops[5].getLoc(), uint32Type, ops[5]);
                    auto v6 = rewriter.create<daphne::CastOp>(ops[6].getLoc(), uint32Type, ops[6]);
                    rewriter.replaceOpWithNewOp<daphne::PartialReadOp>(op, op.getResult().getType(), op.getOperand(), v1, v2, v3, v4, v5, v6);
                    return success();
                } else {
                    return failure();
                }
            }
        }
    };

    struct PartialReadPass : public PassWrapper <PartialReadPass, OperationPass<ModuleOp>> {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartialReadPass)

        void runOnOperation() final;

        StringRef getArgument() const final { return "partial read-op"; }
        StringRef getDescription() const final { return "TODO"; }
    };
}

void PartialReadPass::runOnOperation() {
    auto module = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();

    // if return val is false, rewrite
    auto func = [](daphne::ReadOp op) { 
        bool hasExactlyOneUse = (&op)->getResult().hasOneUse();
        auto u = *((&op)->getResult().getUsers().begin());
        bool isSlicingOp = llvm::dyn_cast<mlir::daphne::SliceTensorOp>(u);
        bool isChunkedTensorType = false;
        if (op.getResult().getType().isa<daphne::TensorType>()) {
            daphne::TensorType type = op.getResult().getType().dyn_cast<daphne::TensorType>();
            isChunkedTensorType = type.getRepresentation() == daphne::TensorRepresentation::Chunked;
        }
        return !(hasExactlyOneUse && isSlicingOp && isChunkedTensorType);
    };

    //target.addDynamicallyLegalOp<daphne::ReadOp>([](daphne::ReadOp op) { return true; });
    target.addDynamicallyLegalOp<daphne::ReadOp>(func);

    RewritePatternSet patterns(&getContext());
    patterns.add<PartialRead>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<Pass> daphne::createPartialReadPass() {
    return std::make_unique<PartialReadPass>();
}
