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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H
#define SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/kernels/UnaryOpCode.h>
#include <runtime/local/kernels/EwUnarySca.h>

#include <cassert>
#include <cstddef>

// TODO: make opCode constexpr since it is constexpr known and removes the fn ptr stuff and runtime switches

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct EwUnaryMat {
    static void apply(UnaryOpCode opCode, DTRes *& res, const DTArg * arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void ewUnaryMat(UnaryOpCode opCode, DTRes *& res, const DTArg * arg, DCTX(ctx)) {
    EwUnaryMat<DTRes, DTArg>::apply(opCode, res, arg, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct EwUnaryMat<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(UnaryOpCode opCode, DenseMatrix<VT> *& res, const DenseMatrix<VT> * arg, DCTX(ctx)) {
        const size_t numRows = arg->getNumRows();
        const size_t numCols = arg->getNumCols();
        
        if(res == nullptr)
            res = DataObjectFactory::create<DenseMatrix<VT>>(numRows, numCols, false);
        
        const VT * valuesArg = arg->getValues();
        VT * valuesRes = res->getValues();
        
        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);
        
        for(size_t r = 0; r < numRows; r++) {
            for(size_t c = 0; c < numCols; c++)
                valuesRes[c] = func(valuesArg[c], ctx);
            valuesArg += arg->getRowSkip();
            valuesRes += res->getRowSkip();
        }
    }
};

// ----------------------------------------------------------------------------
// ChunkedTensor <- ChunkedTensor
// ----------------------------------------------------------------------------

template<typename VT>
struct EwUnaryMat<ChunkedTensor<VT>, ChunkedTensor<VT>> {
    static void apply(UnaryOpCode opCode, ChunkedTensor<VT> *& res, const ChunkedTensor<VT> * arg, DCTX(ctx)) {
        if(res == nullptr)
            res = DataObjectFactory::create<ChunkedTensor<VT>>(arg->tensor_shape, arg->chunk_shape, InitCode::NONE);

        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);
        
        for(size_t i = 0; i < res->total_size_in_elements; i++) {
            res->data[i] = func(arg->data[i], ctx);
        }
    }
};

// ----------------------------------------------------------------------------
// ContiguousTensor <- ContiguousTensor
// ----------------------------------------------------------------------------

template<typename VT>
struct EwUnaryMat<ContiguousTensor<VT>, ContiguousTensor<VT>> {
    static void apply(UnaryOpCode opCode, ContiguousTensor<VT> *& res, const ContiguousTensor<VT> * arg, DCTX(ctx)) {
        if(res == nullptr)
            res = DataObjectFactory::create<ContiguousTensor<VT>>(arg->data.get(), arg->tensor_shape);

        EwUnaryScaFuncPtr<VT, VT> func = getEwUnaryScaFuncPtr<VT, VT>(opCode);
        
        for(size_t i = 0; i < res->total_element_count; i++) {
            res->data[i] = func(arg->data[i], ctx);
        }
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_EWUNARYMAT_H