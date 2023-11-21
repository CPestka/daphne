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

#pragma once

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/ContiguousTensorMatrix.h>
#include <runtime/local/datastructures/ChunkedTensorMatrix.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <stdexcept>
#include <vector>
#include <cstddef>
#include <memory>

#include "AggOpCode.h"

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<typename VTRes, class DTArg>
struct Agg {
    static VTRes apply(AggOpCode opCode,
                       const std::vector<bool>& aggregate_dimension,
                       const DTArg* arg,
                       DCTX(ctx)) = delete;
};

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<typename VTRes, class DTArg>
struct AggSparse {
    static VTRes apply(AggOpCode opCode,
                       const std::vector<bool>& aggregate_dimension,
                       const std::vector<std::vector<size_t>>& chunk_list,
                       const DTArg* arg,
                       DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<typename VTRes, class DTArg>
VTRes agg(AggOpCode opCode, const std::vector<bool>& aggregate_dimension, const DTArg* arg, DCTX(ctx)) {
    return Agg<VTRes, DTArg>::apply(opCode, aggregate_dimension, arg, ctx);
}

template<typename VTRes, class DTArg>
VTRes agg(AggOpCode opCode,
          const std::vector<bool>& aggregate_dimension,
          const std::vector<std::vector<size_t>>& chunk_list,
          const DTArg* arg,
          DCTX(ctx)) {
    return AggSparse<VTRes, DTArg>::apply(opCode, aggregate_dimension, chunk_list, arg, ctx);
}

template<typename VTRes, class DTArg>
VTRes agg(AggOpCode opCode,
          const std::vector<bool>& aggregate_dimension,
          const std::vector<std::pair<size_t, size_t>>& chunk_ranges,
          const DTArg* arg,
          DCTX(ctx)) {
    return AggSparse<VTRes, DTArg>::apply(
      opCode, aggregate_dimension, arg->GetChunkListFromIdRange(chunk_ranges), arg, ctx);
}

template<Scalar_t VTRes, typename VTArg>
struct Agg<ContiguousTensor<VTRes>, ContiguousTensor<VTArg>> {
    static ContiguousTensor<VTRes>* apply(AggOpCode opCode,
                                          const std::vector<bool>& aggregate_dimension,
                                          const ContiguousTensor<VTArg>* arg,
                                          DCTX(ctx)) {
        size_t rank                             = arg->rank;
        if (aggregate_dimension.size() != rank) {
            throw std::runtime_error("Rank of tensor to reduce and size of aggregation map do not match!");
        }

        std::vector<size_t> result_tensor_shape = arg->tensor_shape;

        for (size_t i = 0; i < rank; i++) {
            if (aggregate_dimension[i]) {
                result_tensor_shape[i] = 1;
            }
        }

        ContiguousTensor<VTRes>* result =
          DataObjectFactory::create<ContiguousTensor<VTRes>>(result_tensor_shape, InitCode::NONE);

        AggChunkOPDispatch(opCode,
                           result->data.get(),
                           arg->data.get(),
                           arg->tensor_shape,
                           arg->total_element_count,
                           aggregation_dimension);

        return result;
    }
};

// TODO: !!!deccide where to put "post per chunk" ops and if not here set matrialization bit for reduced chunks!!!
//  Assumes chunks are either matrialized or are async matrialized by other thread -> will hang otherwise
template<Scalar_t VTRes, typename VTArg>
struct Agg<ChunkedTensor<VTRes>, ChunkedTensor<VTArg>> {
    static ChunkedTensor<VTRes>* apply(AggOpCode opCode,
                                       const std::vector<bool>& aggregate_dimension,
                                       const ChunkedTensor<VTArg>* arg,
                                       DCTX(ctx)) {
        size_t rank                             = arg->rank;
        if (aggregate_dimension.size() != rank) {
            throw std::runtime_error("Rank of tensor to reduce and size of aggregation map do not match!");
        }
        std::vector<size_t> result_tensor_shape = arg->tensor_shape;
        std::vector<size_t> result_chunk_shape  = arg->chunk_shape;

        for (size_t i = 0; i < rank; i++) {
            if (aggregate_dimension[i]) {
                result_tensor_shape[i] = 1;
                result_chunk_shape[i]  = 1;
            }
        }

        ChunkedTensor<VTRes>* result =
          DataObjectFactory::create<ChunkedTensor<VTRes>>(result_tensor_shape, result_chunk_shape, InitCode::NONE);

        auto chunk_status = std::make_unique<bool[]>(arg->total_chunk_count);
        for (size_t i = 0; i < arg->total_chunk_count; i++) {
            remaining_chunks[i] = true;
        }
        size_t remaining_chunks = arg->total_chunk_count;
        while (remaining_chunnks != 0) {
            for (size_t i = 0; i < arg->total_chunk_count; i++) {
                if (chunk_status[i]) {
                    bool chunk_can_be_proccessed = false;
                    if (arg->chunk_materialization_flags[i]) {
                        chunk_can_be_proccessed = true;
                    } else {    // chunk not marked materialized, but it may has arrived due to async io
                        IO_STATUS current_chunk_io_status = arg->chunk_io_futures[i].status;

                        switch (current_chunk_io_status) {
                            using enum IO_STATUS;
                            case PRE_SUBMISSION:
                                break;
                            case IN_FLIGHT:
                                break;
                            case SUCCESS:
                                if (arg->chunk_io_futures[i].needs_byte_reversal) {
                                    ReverseArray<VTArg>(arg->getPtrToChunk(i), arg->chunk_element_count);
                                    arg->chunk_io_futures[i].needs_byte_reversal = false;
                                }
                                chunk_can_be_proccessed             = true;
                                arg->chunk_materialization_flags[i] = true;
                                break;
                            default:
                                // Error cases like BAD_FD
                                throw std::runtime_error("Async load of chunk failed");
                                break;
                        }
                    }

                    if (chunk_can_be_proccessed) {
                        AggChunkOPDispatch(opCode,
                                           result->getPtrToChunk(i),
                                           arg->getPtrToChunk(i),
                                           arg->chunk_shape,
                                           arg->chunk_element_count,
                                           aggregation_dimension);
                        chunk_status[i] = false;
                        remaining_chunks--;
                    }
                }
            }
        }

        return result;
    }
};

// TODO: !!!deccide where to put "post per chunk" ops and if not here set matrialization bit for reduced chunks!!!
//  Assumes chunks are either matrialized or are async matrialized by other thread -> will hang otherwise
template<Scalar_t VTRes, typename VTArg>
struct Agg<ChunkedTensor<VTRes>, ChunkedTensor<VTArg>> {
    static ChunkedTensor<VTRes>* apply(AggOpCode opCode,
                                       const std::vector<bool>& aggregate_dimension,
                                       const std::vector<std::vector<size_t>>& chunk_list,
                                       const ChunkedTensor<VTArg>* arg,
                                       DCTX(ctx)) {
        size_t rank                             = arg->rank;
        if (aggregate_dimension.size() != rank) {
            throw std::runtime_error("Rank of tensor to reduce and size of aggregation map do not match!");
        }
        std::vector<size_t> result_tensor_shape = arg->tensor_shape;
        std::vector<size_t> result_chunk_shape  = arg->chunk_shape;

        for (size_t i = 0; i < rank; i++) {
            if (aggregate_dimension[i]) {
                result_tensor_shape[i] = 1;
                result_chunk_shape[i]  = 1;
            }
        }

        ChunkedTensor<VTRes>* result =
          DataObjectFactory::create<ChunkedTensor<VTRes>>(result_tensor_shape, result_chunk_shape, InitCode::NONE);

        auto chunk_status = std::make_unique<std::pair<size_t, bool>[]>(chunk_list.size());
        for (size_t i = 0; i < chunk_list.size(); i++) {
            remaining_chunks[i] = {arg->getLinearChunkIdFromChunkIds(chunk_list[i]), true};
        }
        size_t remaining_chunks = chunk_list.size();
        while (remaining_chunnks != 0) {
            for (size_t i = 0; i < chunk_list.size(); i++) {
                if (std::get<1>(chunk_status[i])) {
                    bool chunk_can_be_proccessed = false;
                    if (arg->chunk_materialization_flags[std::get<0>(chunk_list[i])]) {
                        chunk_can_be_proccessed = true;
                    } else {    // chunk not marked materialized, but it may has arrived due to async io
                        IO_STATUS current_chunk_io_status = arg->chunk_io_futures[std::get<0>(chunk_list[i])].status;

                        switch (current_chunk_io_status) {
                            using enum IO_STATUS;
                            case PRE_SUBMISSION:
                                break;
                            case IN_FLIGHT:
                                break;
                            case SUCCESS:
                                if (arg->chunk_io_futures[std::get<0>(chunk_list[i])].needs_byte_reversal) {
                                    ReverseArray<VTArg>(arg->getPtrToChunk(std::get<0>(chunk_list[i])),
                                                        arg->chunk_element_count);
                                    arg->chunk_io_futures[std::get<0>(chunk_list[i])].needs_byte_reversal = false;
                                }
                                chunk_can_be_proccessed                                      = true;
                                arg->chunk_materialization_flags[std::get<0>(chunk_list[i])] = true;
                                break;
                            default:
                                // Error cases like BAD_FD
                                throw std::runtime_error("Async load of chunk failed");
                                break;
                        }
                    }

                    if (chunk_can_be_proccessed) {
                        AggChunkOPDispatch(opCode,
                                           result->getPtrToChunk(std::get<0>(chunk_list[i])),
                                           arg->getPtrToChunk(std::get<0>(chunk_list[i])),
                                           arg->chunk_shape,
                                           arg->chunk_element_count,
                                           aggregation_dimension);
                        chunk_status[i] = {std::get<0>(chunk_list[i]), false};
                        remaining_chunks--;
                    }
                }
            }
        }

        return result;
    }
};

template<typename VTRes, typename VTArg>
void AggChunkOPDispatch(AggOpCode opCode,
                        VTRes* dest,
                        VTArg* src,
                        const std::vector<size_t>& chunk_shape,
                        size_t chunk_size,
                        const std::vector<bool>& aggregation_dimension) {
    switch (opCode) {
        using enum AggOpCode;
        case MIN:
            AggChunk<VTRes, VTArg, AggOpCode::MIN>(dest, src, chunk_shape, chunk_size, aggregation_dimension);
            break;
        case MAX:
            AggChunk<VTRes, VTArg, AggOpCode::MAX>(dest, src, chunk_shape, chunk_size, aggregation_dimension);
            break;
        case SUM:
            AggChunk<VTRes, VTArg, AggOpCode::MIN>(dest, src, chunk_shape, chunk_size, aggregation_dimension);
            break;
        case PROD:
            AggChunk<VTRes, VTArg, AggOpCode::MIN>(dest, src, chunk_shape, chunk_size, aggregation_dimension);
            break;
        default:
            // TODO: IDXmin/max, stddev mean
            throw std::runtime_error("unsupported op_code reached");
    }
}

template<typename VTRes, typename VTArg, AggOpCode opCode>
void AggChunk(VTRes* dest,
              VTArg* src,
              std::vector<size_t> chunk_shape,    // intentionally per value
              size_t chunk_size,
              const std::vector<bool>& aggregate_dimension) {
    size_t rank = chunk_size.size();
    if (rank == 0) {
        dest[0] = static_cast<VTRes>(src[0]);
        return;
    }

    auto scratch_space = std::make_unique<VTRes[]>(2 * chunk_size);

    VTRes* current_dest = scratch_space.get();
    VTRes* current_src;    // used after the first swap
    size_t current_chunk_size = chunk_size;

    std::vector<size_t> chunk_strides;
    chunk_strides.resize(rank);

    bool is_first_swap = true;
    for (size_t i = 0; i < chunk_shape.size(); i++) {
        // Ignore dims not flaged for reduction
        if (aggregate_dimension[i]) {
            // recalculate strides since they change in each iteration
            chunk_strides[0] = 1;
            for (size_t j = 1; j < rank - 1; j++) {
                chunk_strides[j] = chunk_strides[j - 1] * chunk_shape[j - 1];
            }

            if (is_first_swap) {
                AggChunkSingleDim<VTRes, VTArg, opCode>(
                  current_dest, src, chunk_shape, chunk_strides, current_chunk_size, i);
                current_src  = current_dest;
                current_dest = &(scratch_space[chunk_size]);
            } else {
                AggChunkSingleDim<VTRes, VTRes, opCode>(
                  current_dest, current_src, chunk_shape, chunk_strides, current_chunk_size, i);
                std::swap(current_src, current_dest);
            }
            current_chunk_size = current_chunk_size / chunk_shape[i];
            chunk_shape[i]     = 1;

            // adjust shape and strides and size
            // swap ptrs
            is_first_swap = false;
        }
    }

    std::memcpy(dest, current_src, sizeof(VTRes) * current_chunk_size);
}

template<typename VT, typename VT, AggOpCode opCode>
void AggChunkSingleDim(VT* dest,
                       VT* src,
                       const std::vector<size_t>& chunk_shape,
                       const std::vector<size_t>& chunk_strides,
                       size_t chunk_size,
                       size_t aggregate_dimension) {
    int32_t rank = static_cast<int32_t>(chunk_shape.size());

    for (size_t i = 0; i < chunk_size; i++) {
        if (!isFirstEntryInAggDim(i, chunk_strides)) {
            continue;
        }

        dest[i] = static_cast<VTRes>(src[i]);
        for (size_t j = 1; j < chunk_shape[aggregate_dimension]; j++) {
            if constexpr (opCode == AggOpCode::SUM) {
                dest[i] += static_cast<VTRes>(src[i + (j * chunk_strides[aggregate_dimension])]);
            } else if constexpr (opCode == AggOpCode::PROD) {
                dest[i] *= static_cast<VTRes>(src[i + (j * chunk_strides[aggregate_dimension])]);
            } else if constexpr (opCode == AggOpCode::MIN) {
                dest[i] = std::min(dest[i], static_cast<VTRes>(src[i + (j * chunk_strides[aggregate_dimension])]));
            } else if constexpr (opCode == AggOpCode::MAX) {
                dest[i] = std::min(dest[i], static_cast<VTRes>(src[i + (j * chunk_strides[aggregate_dimension])]));
            } else {
                throw std::runtime_error("unsupported op_code reached");
                // Todo handle idxmin/max and mean stddev further up
            }
        }
    }
}
