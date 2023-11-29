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
#include <parser/sql/SQLParser.h>
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace mlir;

namespace {
    struct PartialRead : public RewritePattern {

        PartialRead(MLIRContext * context, PatternBenefit benefit = 1) : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context) {}

        LogicalResult matchAndRewrite(Operation* op, PatternRewriter &rewriter) const override {
            if (auto readOp = llvm::dyn_cast<mlir::daphne::ReadOp>(op)){
                std::cout << "read op discovered! " << readOp->getName().getStringRef().str() << " operands: " << readOp->getNumOperands() << std::endl;
                return success();
            }
            return failure();
        }
    };

    struct PartialReadPass1 : public PassWrapper <PartialReadPass1, OperationPass<ModuleOp>> {
        void runOnOperation() final;

        StringRef getArgument() const final { return "rewrite-sqlop"; }
        StringRef getDescription() const final { return "TODO"; }
    };
}

void PartialReadPass1::runOnOperation() {
    auto module = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp>();
    //target.addIllegalOp<mlir::daphne::ReadOp>();

    patterns.add<PartialRead>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> daphne::createPartialReadPass()
{
    return std::make_unique<PartialReadPass1>();
}
