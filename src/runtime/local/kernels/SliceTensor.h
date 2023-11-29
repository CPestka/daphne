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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_SLICETENSOR_H
#define SRC_RUNTIME_LOCAL_KERNELS_SLICETENSOR_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <stdexcept>

#include <cstddef>
#include <cstdint>

// IMO its a dice not a slice, but shrug*

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct SliceTensor {
    static void apply(DTRes *& res, const DTArg * arg, size_t lowerInclX, size_t upperExclX, size_t lowerInclY, size_t upperExclY, size_t lowerInclZ, size_t upperExclZ, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void sliceTensor(DTRes *& res, const DTArg * arg, size_t lowerInclX, size_t upperExclX, size_t lowerInclY, size_t upperExclY, size_t lowerInclZ, size_t upperExclZ, DCTX(ctx)) {
    SliceTensor<DTRes, DTArg>::apply(res, arg, lowerInclX, upperExclX, lowerInclY, upperExclY, lowerInclZ, upperExclZ, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// ContiguousTensor <- ContiguousTensor
// ----------------------------------------------------------------------------

template<typename VT>
struct SliceTensor<ContiguousTensor<VT>, ContiguousTensor<VT>> {
    static void apply(ContiguousTensor<VT> *& res, const ContiguousTensor<VT> * arg, size_t lowerInclX, size_t upperExclX, size_t lowerInclY, size_t upperExclY, size_t lowerInclZ, size_t upperExclZ, DCTX(ctx)) {
        std::vector<std::pair<size_t, size_t>> index_ranges {{lowerInclX, upperExclX}, {lowerInclY, upperExclY}, {lowerInclZ, upperExclZ}};
        res = arg->tryDice(index_ranges);
    }
};

// ----------------------------------------------------------------------------
// ChunkedTensor <- ChunkedTensor
// ----------------------------------------------------------------------------

template<typename VT>
struct SliceTensor<ChunkedTensor<VT>, ChunkedTensor<VT>> {
    static void apply(ChunkedTensor<VT> *& res, const ChunkedTensor<VT> * arg, size_t lowerInclX, size_t upperExclX, size_t lowerInclY, size_t upperExclY, size_t lowerInclZ, size_t upperExclZ, DCTX(ctx)) {
        std::vector<std::pair<size_t, size_t>> index_ranges {{lowerInclX, upperExclX}, {lowerInclY, upperExclY}, {lowerInclZ, upperExclZ}};

        //TODO: use tryDice if id ranges dont match chunk boundaries and trsDiceAtChunkLevel otherwise
        //res = arg->tryDice(index_ranges);
    }
};

// TODO add slice chunked -> contiguous
#endif //SRC_RUNTIME_LOCAL_KERNELS_SLICETENSOR_H
