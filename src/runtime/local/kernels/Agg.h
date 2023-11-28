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
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/kernels/EwBinarySca.h>

#include <cmath>
#include <stdexcept>
#include <vector>
#include <cstddef>
#include <memory>
#include <concepts>
#include <type_traits>
#include <algorithm>
#include <optional>

#include "AggOpCode.h"

// TODO: IDXmin/max

// The arg are not const here since the kernels check and update the materilization flags / io status of chunks
// The alternative way of doing this would be to add callbacks that do that to the io engine so th compute kernels do
// not have to modify the arg tensor
// TODO: handle overhanging chunks properly

template<typename T>
concept Scalar_t = std::is_arithmetic<T>::value;

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<typename VTRes, class DTArg>
struct Agg {
    static VTRes* apply(AggOpCode opCode, const std::vector<bool>& aggregate_dimension, DTArg* arg, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<typename VTRes, class DTArg>
struct AggSparse {
    static VTRes* apply(AggOpCode opCode,
                        const std::vector<bool>& aggregate_dimension,
                        const std::vector<std::vector<size_t>>& chunk_list,
                        DTArg* arg,
                        DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<typename VTRes, class DTArg>
VTRes* agg(AggOpCode opCode, const std::vector<bool>& aggregate_dimension, DTArg* arg, DCTX(ctx)) {
    return Agg<VTRes, DTArg>::apply(opCode, aggregate_dimension, arg, ctx);
}

template<typename VTRes, class DTArg>
VTRes* agg(AggOpCode opCode,
           const std::vector<bool>& aggregate_dimension,
           const std::vector<std::pair<size_t, size_t>>& chunk_ranges,
           DTArg* arg,
           DCTX(ctx)) {
    return AggSparse<VTRes, DTArg>::apply(opCode, aggregate_dimension, chunk_ranges, arg, ctx);
}

template<Scalar_t VTRes, Scalar_t VTArg>
struct Agg<ContiguousTensor<VTRes>, ContiguousTensor<VTArg>> {
    static ContiguousTensor<VTRes>* apply(AggOpCode opCode,
                                          const std::vector<bool>& aggregate_dimension,
                                          const ContiguousTensor<VTArg>* arg,
                                          DCTX(ctx)) {
        size_t rank = arg->rank;
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

        AggChunkOPDispatch<VTRes, VTArg, true>(opCode,
                                               result->data.get(),
                                               arg->data.get(),
                                               arg->tensor_shape,
                                               arg->total_element_count,
                                               aggregate_dimension,
                                               true);

        return result;
    }
};

struct ChunkMap {
    size_t linear_src_id;
    size_t linear_dest_id;
    bool not_processed_yet;
};

std::vector<std::vector<std::vector<size_t>>> GetChunkAggregationLists(
  std::vector<std::vector<size_t>> chunk_list,    // intentional per value
  size_t aggregation_dim) {
    size_t rank = chunk_list[0].size();
    std::vector<std::vector<std::vector<size_t>>> chunk_lists;

    while (chunk_list.size() != 0) {
        std::vector<size_t> current_ids = chunk_list.back();
        chunk_list.pop_back();

        chunk_lists.push_back({{}});
        chunk_lists.back().push_back(current_ids);

        for (size_t i = 0; i < chunk_list.size(); i++) {
            bool found_match = true;
            for (size_t j = 0; j < rank; j++) {
                if (j != aggregation_dim) {
                    if (chunk_list[i][j] != chunk_lists.back()[0][j]) {
                        found_match = false;
                    }
                }
            }
            if (found_match) {
                chunk_lists.back().push_back(chunk_list[i]);
                chunk_list.erase(chunk_list.begin() + static_cast<int64_t>(i));
                i--;
            }
        }
    }
    return chunk_lists;
}

//  Assumes chunks are either matrialized or are async matrialized by other thread -> will hang otherwise
template<Scalar_t VTRes, Scalar_t VTArg>
struct Agg<ChunkedTensor<VTRes>, ChunkedTensor<VTArg>> {
    static ChunkedTensor<VTRes>* apply(AggOpCode opCode,
                                       const std::vector<bool>& aggregate_dimension,
                                       ChunkedTensor<VTArg>* arg,
                                       DCTX(ctx)) {
        size_t rank = arg->rank;
        if (aggregate_dimension.size() != rank) {
            throw std::runtime_error("Rank of tensor to reduce and size of aggregation map do not match!");
        }

        std::vector<size_t> result_tensor_shape = arg->tensor_shape;
        std::vector<size_t> result_chunk_shape  = arg->chunk_shape;
        size_t chunk_count                      = arg->total_chunk_count;

        for (size_t i = 0; i < rank; i++) {
            if (aggregate_dimension[i]) {
                result_tensor_shape[i] = 1;
                result_chunk_shape[i]  = 1;
            }
        }

        ChunkedTensor<VTRes>* result =
          DataObjectFactory::create<ChunkedTensor<VTRes>>(result_tensor_shape, result_chunk_shape, InitCode::NONE);

        std::vector<bool> dest_chunk_has_been_touched(result->total_chunk_count, false);

        if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {    // Sum,Prod,min,max,idxmin,idxmax
            auto chunk_status = std::make_unique<ChunkMap[]>(chunk_count);
            std::vector<size_t> dest_chunk_ids;
            dest_chunk_ids.resize(rank);
            for (size_t i = 0; i < chunk_count; i++) {
                std::vector<size_t> tmp_src_ids = arg->getChunkIdsFromLinearChunkId(i);

                for (size_t j = 0; j < rank; j++) {
                    dest_chunk_ids[j] = aggregate_dimension[j] ? 0 : tmp_src_ids[j];
                }
                chunk_status[i] = {i, result->getLinearChunkIdFromChunkIds(dest_chunk_ids), true};
            }

            size_t remaining_chunks = chunk_count;
            while (remaining_chunks != 0) {
                for (size_t i = 0; i < chunk_count; i++) {
                    if (chunk_status[i].not_processed_yet) {
                        bool chunk_can_be_proccessed = false;
                        if (arg->chunk_materialization_flags[chunk_status[i].linear_src_id]) {
                            chunk_can_be_proccessed = true;
                        } else {    // chunk not marked materialized, but it may has arrived due to async io
                            IO_STATUS current_chunk_io_status =
                              arg->chunk_io_futures[chunk_status[i].linear_src_id].status;

                            switch (current_chunk_io_status) {
                                using enum IO_STATUS;
                                case PRE_SUBMISSION:
                                    break;
                                case IN_FLIGHT:
                                    break;
                                case SUCCESS:
                                    if (arg->chunk_io_futures[chunk_status[i].linear_src_id].needs_byte_reversal) {
                                        ReverseArray<VTArg>(arg->getPtrToChunk(chunk_status[i].linear_src_id),
                                                            arg->chunk_element_count);
                                        arg->chunk_io_futures[chunk_status[i].linear_src_id].needs_byte_reversal =
                                          false;
                                    }
                                    chunk_can_be_proccessed                                         = true;
                                    arg->chunk_materialization_flags[chunk_status[i].linear_src_id] = true;
                                    break;
                                default:
                                    // Error cases like BAD_FD
                                    throw std::runtime_error("Async load of chunk failed");
                                    break;
                            }
                        }

                        if (chunk_can_be_proccessed) {
                            bool is_first_op = !dest_chunk_has_been_touched[chunk_status[i].linear_dest_id];
                            AggChunkOPDispatch<VTRes, VTArg, false>(
                              opCode,
                              result->getPtrToChunk(chunk_status[i].linear_dest_id),
                              arg->getPtrToChunk(chunk_status[i].linear_src_id),
                              arg->chunk_shape,
                              arg->chunk_element_count,
                              aggregate_dimension,
                              is_first_op);
                            dest_chunk_has_been_touched[chunk_status[i].linear_dest_id] = true;
                            chunk_status[i].not_processed_yet                           = false;
                            remaining_chunks--;
                        }
                    }
                }
            }
        } else {
            // i.e. MEAN and STDDev

            // Restrict mean and stddev to only be applied in one dim (for simplicity and also it seems to me that
            // it is not likely to be sensible to apply it to multiple dims consecutively)
            size_t dims_to_reduce = 0;
            for (size_t i = 0; i < rank; i++) {
                if (aggregate_dimension[i]) {
                    dims_to_reduce++;
                }
            }
            if (dims_to_reduce > 1) {
                throw std::runtime_error(
                  "Applying mean and stddev over more than one dim at once is currently not supported");
            }

            bool no_aggregation_dim = true;
            for (size_t i = 0; i < rank; i++) {
                if (aggregate_dimension[i]) {
                    no_aggregation_dim = false;
                    std::vector<std::pair<size_t, size_t>> full_tensor_range;
                    full_tensor_range.resize(rank);
                    for (size_t j = 0; j < rank; j++) {
                        full_tensor_range[j] = {0, arg->tensor_shape[j] + 1};
                    }
                    // Fill the lists
                    std::vector<std::vector<std::vector<size_t>>> current_lists_of_chunks =
                      GetChunkAggregationLists(arg->GetChunkListFromIdRange(full_tensor_range).value(), i);

                    std::vector<bool> chunk_list_fully_arrived(current_lists_of_chunks.size(), false);
                    size_t remaining_lists_to_proccess = current_lists_of_chunks.size();

                    while (remaining_lists_to_proccess != 0) {
                        std::optional<size_t> fully_matrialized_chunk_list_id =
                          CheckChunkListArrival(current_lists_of_chunks, chunk_list_fully_arrived, arg);

                        if (fully_matrialized_chunk_list_id) {
                            std::vector<size_t> dest_chunk_id =
                              current_lists_of_chunks[fully_matrialized_chunk_list_id.value()][0];
                            dest_chunk_id[i]            = 0;
                            size_t linear_dest_chunk_id = result->getLinearChunkIdFromChunkIds(dest_chunk_id);

                            std::vector<VTArg*> src_chunk_id_ptrs;
                            src_chunk_id_ptrs.resize(
                              current_lists_of_chunks[fully_matrialized_chunk_list_id.value()].size());
                            for (size_t j = 0; j < src_chunk_id_ptrs.size(); j++) {
                                src_chunk_id_ptrs[j] = arg->getPtrToChunk(
                                  current_lists_of_chunks[fully_matrialized_chunk_list_id.value()][j]);
                            }

                            AggChunkList<VTRes, VTArg>(result->getPtrToChunk(dest_chunk_id),
                                                       src_chunk_id_ptrs,
                                                       i,
                                                       arg->chunk_shape,
                                                       arg->chunk_element_count,
                                                       opCode);
                            dest_chunk_has_been_touched[linear_dest_chunk_id] = true;
                            remaining_lists_to_proccess--;
                        }
                    }
                }
            }
            if (no_aggregation_dim) {    // for correctness no need to be efficient
                for (size_t i = 0; i < chunk_count; i++) {
                    while (!arg->IsChunkMaterialized(arg->getChunkIdsFromLinearChunkId(i))) {
                    };
                    VTRes* dest = result->getPtrToChunk(result->getChunkIdsFromLinearChunkId(i));
                    VTRes* src  = arg->getPtrToChunk(arg->getChunkIdsFromLinearChunkId(i));
                    for (size_t j = 0; j < arg->chunk_element_count; j++) {
                        dest[j] = static_cast<VTRes>(src[j]);
                    }
                    dest_chunk_has_been_touched[i] = true;
                }
            }
        }

        for (size_t i = 0; i < result->total_chunk_count; i++) {
            if (dest_chunk_has_been_touched[i]) {
                result->chunk_materialization_flags[i] = true;
            }
        }

        return result;
    }
};

//  Assumes chunks are either matrialized or are async matrialized by other thread -> will hang otherwise
template<Scalar_t VTRes, Scalar_t VTArg>
struct AggSparse<ChunkedTensor<VTRes>, ChunkedTensor<VTArg>> {
    static ChunkedTensor<VTRes>* apply(AggOpCode opCode,
                                       const std::vector<bool>& aggregate_dimension,
                                       const std::vector<std::pair<size_t, size_t>>& chunk_ranges,
                                       ChunkedTensor<VTArg>* arg,
                                       DCTX(ctx)) {
        size_t rank = arg->rank;
        if (aggregate_dimension.size() != rank) {
            throw std::runtime_error("Rank of tensor to reduce and size of aggregation map do not match!");
        }

        if (aggregate_dimension.size() != rank) {
            throw std::runtime_error("Rank of tensor to reduce and size of aggregation map do not match!");
        }
        if (chunk_ranges.size() != rank) {
            throw std::runtime_error("Rank of tensor to reduce and size of chunk ranges do not match!");
        }
        for (size_t i = 0; i < rank; i++) {
            if ((std::get<0>(chunk_ranges[i]) >= std::get<1>(chunk_ranges[i])) ||
                (std::get<0>(chunk_ranges[i]) >= arg->tensor_shape[i]) ||
                (std::get<1>(chunk_ranges[i]) > arg->tensor_shape[i])) {
                throw std::runtime_error(
                  "Invalid chunk range! lhs must be larger than rhs and neither may be out >= tensor_shape[i]");
            }
        }

        std::vector<std::vector<size_t>> chunk_list = arg->GetChunkListFromChunkRange(chunk_ranges).value();
        std::vector<size_t> result_tensor_shape;
        result_tensor_shape.resize(rank);
        for (size_t i = 0; i < rank; i++) {
            result_tensor_shape[i] =
              (std::get<1>(chunk_ranges[i]) - std::get<0>(chunk_ranges[i])) * arg->chunk_shape[i];
        }
        std::vector<size_t> result_chunk_shape = arg->chunk_shape;

        for (size_t i = 0; i < rank; i++) {
            if (aggregate_dimension[i]) {
                result_tensor_shape[i] = 1;
                result_chunk_shape[i]  = 1;
            }
        }

        ChunkedTensor<VTRes>* result;
        if (opCode == AggOpCode::STDDEV) {
            result = agg<ChunkedTensor<VTRes>, ChunkedTensor<VTArg>>(
              AggOpCode::SUM, aggregate_dimension, chunk_ranges, arg, nullptr);
        } else {
            result =
              DataObjectFactory::create<ChunkedTensor<VTRes>>(result_tensor_shape, result_chunk_shape, InitCode::NONE);
        }

        std::vector<bool> dest_chunk_has_been_touched(result->total_chunk_count, false);

        if (AggOpCodeUtils::isPureBinaryReduction(opCode)) {    // Sum,Prod,min,max,idxmin,idxmax
            // TODO handle idxmin/max

            auto chunk_status = std::make_unique<ChunkMap[]>(chunk_list.size());
            std::vector<size_t> dest_chunk_ids;
            dest_chunk_ids.resize(rank);
            for (size_t i = 0; i < chunk_list.size(); i++) {
                for (size_t j = 0; j < rank; j++) {
                    dest_chunk_ids[j] = aggregate_dimension[j] ? 0 : chunk_list[i][j] - std::get<0>(chunk_ranges[j]);
                }
                chunk_status[i] = {arg->getLinearChunkIdFromChunkIds(chunk_list[i]),
                                   result->getLinearChunkIdFromChunkIds(dest_chunk_ids),
                                   true};
            }

            size_t remaining_chunks = chunk_list.size();
            while (remaining_chunks != 0) {
                for (size_t i = 0; i < chunk_list.size(); i++) {
                    if (chunk_status[i].not_processed_yet) {
                        bool chunk_can_be_proccessed = false;
                        if (arg->chunk_materialization_flags[chunk_status[i].linear_src_id]) {
                            chunk_can_be_proccessed = true;
                        } else {    // chunk not marked materialized, but it may has arrived due to async io
                            IO_STATUS current_chunk_io_status =
                              arg->chunk_io_futures[chunk_status[i].linear_src_id].status;

                            switch (current_chunk_io_status) {
                                using enum IO_STATUS;
                                case PRE_SUBMISSION:
                                    break;
                                case IN_FLIGHT:
                                    break;
                                case SUCCESS:
                                    if (arg->chunk_io_futures[chunk_status[i].linear_src_id].needs_byte_reversal) {
                                        ReverseArray<VTArg>(arg->getPtrToChunk(chunk_status[i].linear_src_id),
                                                            arg->chunk_element_count);
                                        arg->chunk_io_futures[chunk_status[i].linear_src_id].needs_byte_reversal =
                                          false;
                                    }
                                    chunk_can_be_proccessed                                         = true;
                                    arg->chunk_materialization_flags[chunk_status[i].linear_src_id] = true;
                                    break;
                                default:
                                    // Error cases like BAD_FD
                                    throw std::runtime_error("Async load of chunk failed");
                                    break;
                            }
                        }

                        if (chunk_can_be_proccessed) {
                            bool is_first_op = !dest_chunk_has_been_touched[chunk_status[i].linear_dest_id];
                            AggChunkOPDispatch<VTRes, VTArg, false>(
                              opCode,
                              result->getPtrToChunk(chunk_status[i].linear_dest_id),
                              arg->getPtrToChunk(chunk_status[i].linear_src_id),
                              arg->chunk_shape,
                              arg->chunk_element_count,
                              aggregate_dimension,
                              is_first_op);
                            dest_chunk_has_been_touched[chunk_status[i].linear_dest_id] = true;
                            chunk_status[i].not_processed_yet                           = false;
                            remaining_chunks--;
                        }
                    }
                }
            }
        } else {
            // i.e. MEAN and STDDev

            // Restrict mean and stddev to only be applied in one dim (for simplicity and also it seems to me that
            // it is not likely to be sensible to apply it to multiple dims consecutively)
            size_t dims_to_reduce = 0;
            for (size_t i = 0; i < rank; i++) {
                if (aggregate_dimension[i]) {
                    dims_to_reduce++;
                }
            }
            if (dims_to_reduce > 1) {
                throw std::runtime_error(
                  "Applying mean and stddev over more than one dim at once is currently not supported");
            }

            bool no_aggregation_dim = true;
            for (size_t i = 0; i < rank; i++) {
                if (aggregate_dimension[i]) {
                    no_aggregation_dim = false;
                    // Fill the lists
                    std::vector<std::vector<std::vector<size_t>>> current_lists_of_chunks =
                      GetChunkAggregationLists(chunk_list, i);

                    std::vector<bool> chunk_list_fully_arrived(current_lists_of_chunks.size(), false);
                    size_t remaining_lists_to_proccess = current_lists_of_chunks.size();

                    while (remaining_lists_to_proccess != 0) {
                        std::optional<size_t> fully_matrialized_chunk_list_id =
                          CheckChunkListArrival(current_lists_of_chunks, chunk_list_fully_arrived, arg);

                        if (fully_matrialized_chunk_list_id) {
                            std::vector<size_t> dest_chunk_id =
                              current_lists_of_chunks[fully_matrialized_chunk_list_id.value()][0];
                            dest_chunk_id[i]            = 0;
                            size_t linear_dest_chunk_id = result->getLinearChunkIdFromChunkIds(dest_chunk_id);

                            std::vector<VTArg*> src_chunk_id_ptrs;
                            src_chunk_id_ptrs.resize(
                              current_lists_of_chunks[fully_matrialized_chunk_list_id.value()].size());
                            for (size_t j = 0; j < src_chunk_id_ptrs.size(); j++) {
                                src_chunk_id_ptrs[j] = arg->getPtrToChunk(
                                  current_lists_of_chunks[fully_matrialized_chunk_list_id.value()][j]);
                            }

                            AggChunkList<VTRes, VTArg>(result->getPtrToChunk(dest_chunk_id),
                                                       src_chunk_id_ptrs,
                                                       i,
                                                       arg->chunk_shape,
                                                       arg->chunk_element_count,
                                                       opCode);
                            dest_chunk_has_been_touched[linear_dest_chunk_id] = true;
                            remaining_lists_to_proccess--;
                        }
                    }
                }
            }

            if (no_aggregation_dim) {    // for correctness no need to be efficient
                for (size_t i = 0; i < arg->total_chunk_count; i++) {
                    while (!arg->IsChunkMaterialized(arg->getChunkIdsFromLinearChunkId(i))) {
                    };
                    VTRes* dest = result->getPtrToChunk(result->getChunkIdsFromLinearChunkId(i));
                    VTRes* src  = arg->getPtrToChunk(arg->getChunkIdsFromLinearChunkId(i));
                    for (size_t j = 0; j < arg->chunk_element_count; j++) {
                        dest[j] = static_cast<VTRes>(src[j]);
                    }
                    dest_chunk_has_been_touched[i] = true;
                }
            }
        }

        for (size_t i = 0; i < result->total_chunk_count; i++) {
            if (dest_chunk_has_been_touched[i]) {
                result->chunk_materialization_flags[i] = true;
            }
        }

        return result;
    }
};

template<typename VT>
std::optional<size_t> CheckChunkListArrival(
  const std::vector<std::vector<std::vector<size_t>>>& current_lists_of_chunks,
  std::vector<bool>& chunk_list_fully_arrived,
  ChunkedTensor<VT>* arg) {
    for (size_t i = 0; i < chunk_list_fully_arrived.size(); i++) {
        if (!chunk_list_fully_arrived[i]) {
            size_t chunk_arrival_counter = 0;
            for (size_t j = 0; j < current_lists_of_chunks[i].size(); j++) {
                if (arg->IsChunkMaterialized(current_lists_of_chunks[i][j])) {
                    chunk_arrival_counter++;
                } else {    // chunk not marked materialized, but it may has arrived due to async io
                    size_t linear_chunk_id = arg->getLinearChunkIdFromChunkIds(current_lists_of_chunks[i][j]);
                    IO_STATUS current_chunk_io_status = arg->chunk_io_futures[linear_chunk_id].status;

                    switch (current_chunk_io_status) {
                        using enum IO_STATUS;
                        case PRE_SUBMISSION:
                            break;
                        case IN_FLIGHT:
                            break;
                        case SUCCESS:
                            if (arg->chunk_io_futures[linear_chunk_id].needs_byte_reversal) {
                                ReverseArray<VT>(arg->getPtrToChunk(linear_chunk_id), arg->chunk_element_count);
                                arg->chunk_io_futures[linear_chunk_id].needs_byte_reversal = false;
                            }
                            chunk_arrival_counter++;
                            arg->chunk_materialization_flags[linear_chunk_id] = true;
                            break;
                        default:
                            // Error cases like BAD_FD
                            throw std::runtime_error("Async load of chunk failed");
                            break;
                    }
                }
            }
            if (chunk_arrival_counter == current_lists_of_chunks[i].size()) {
                chunk_list_fully_arrived[i] = true;
                return i;
            }
        }
    }
    return std::nullopt;
}

template<Scalar_t VTRes, Scalar_t VTArg>
void AggChunkList(VTRes* dest,
                  const std::vector<VTArg*>& src_chunks,
                  size_t aggregation_dim,
                  const std::vector<size_t>& chunk_shape,
                  size_t chunk_size,
                  AggOpCode opCode) {
    size_t rank = chunk_shape.size();

    std::vector<std::vector<VTRes>> rows;
    rows.resize(chunk_size / chunk_shape[aggregation_dim]);
    for (size_t i = 0; i < rows.size(); i++) {
        rows.resize(chunk_shape[aggregation_dim]);
    }

    std::vector<size_t> strides_src;
    std::vector<size_t> strides_row;
    std::vector<size_t> strides_dest;
    strides_src.resize(rank);
    strides_row.resize(rank);
    strides_dest.resize(rank);
    strides_src[0]  = 1;
    strides_row[0]  = 1;
    strides_dest[0] = 1;
    for (size_t i = 1; i < rank; i++) {
        strides_src[i] = strides_src[i - 1] * chunk_shape[i - 1];
        if (i == aggregation_dim) {
            strides_row[i]  = strides_row[i - 1] * chunk_shape[i - 1] * src_chunks.size();
            strides_dest[i] = strides_dest[i - 1];
        } else {
            strides_src[i]  = strides_src[i - 1] * chunk_shape[i - 1];
            strides_dest[i] = strides_dest[i - 1] * chunk_shape[i - 1];
        }
    }

    size_t total_elements = chunk_size * src_chunks.size();
    std::vector<size_t> current_ids;
    current_ids.resize(rank);
    for (size_t i = 0; i < total_elements; i++) {
        size_t tmp = i;
        for (int64_t j = rank - 1; j >= 0; j--) {
            current_ids[j] = tmp / strides_row[j];
            tmp            = tmp % strides_row[j];
        }

        size_t row_id             = current_ids[aggregation_dim];
        size_t src_chunk_id       = current_ids[aggregation_dim] / chunk_shape[aggregation_dim];
        size_t inter_src_chunk_id = current_ids[aggregation_dim] % chunk_shape[aggregation_dim];

        current_ids[aggregation_dim] = 0;

        size_t linear_src_inter_chunk_id = current_ids[0];
        size_t linear_id_dest            = current_ids[0];
        for (size_t j = 1; j < rank; j++) {
            linear_id_dest += (current_ids[j] * strides_dest[j]);
        }
        current_ids[aggregation_dim] = inter_src_chunk_id;
        for (size_t j = 1; j < rank; j++) {
            linear_src_inter_chunk_id += (current_ids[j] * strides_src[j]);
        }

        rows[linear_id_dest][row_id] = static_cast<VTRes>(src_chunks[linear_src_inter_chunk_id][src_chunk_id]);
    }

    switch (opCode) {
        using enum AggOpCode;
        case MEAN:
            for (size_t i = 0; i < rows.size(); i++) {
                std::sort(rows[i].begin(), rows[i].end());
                dest[i] = rows[i][rows[i].size() / 2];
            }
            break;
        case STDDEV: {
            for (size_t i = 0; i < rows.size(); i++) {
                VTRes avg = (dest[i] / static_cast<VTRes>(rows.size()));

                VTRes tmp = 0;
                for (size_t j = 0; j < rows[i].size(); j++) {
                    tmp += ((rows[i][j] - avg) * (rows[i][j] - avg));
                }
                dest[i] = std::sqrt(tmp / rows[i].size());
            }
        } break;
        default:
            throw std::runtime_error("unsupported op_code reached");
            break;
    }
}

bool IsFirstEntryInAggDim(size_t linear_id, const std::vector<size_t>& strides, size_t aggregation_dim) {
    size_t rank = strides.size();
    for (size_t i = 0; i < rank; i++) {
        if (i == aggregation_dim) {
            return (linear_id / strides[i]) == 0;
        }
        linear_id = linear_id % strides[i];
    }
    // unreachable
    throw std::runtime_error("unreachable control flow reached");
    return false;
}

template<Scalar_t VTRes, Scalar_t VTArg, AggOpCode opCode, bool is_only_chunk>
void AggChunkSingleDim(VTRes* dest,
                       VTArg* src,
                       const std::vector<size_t>& chunk_shape,
                       const std::vector<size_t>& src_chunk_strides,
                       const std::vector<size_t>& dest_chunk_strides,
                       size_t chunk_size,
                       size_t aggregate_dimension) {
    size_t dest_chunk_size = chunk_size / chunk_shape[aggregate_dimension];
    size_t rank            = chunk_shape.size();

    if constexpr (is_only_chunk && (opCode == AggOpCode::STDDEV)) {
        AggChunkSingleDim<VTRes, VTArg, AggOpCode::SUM, true>(
          dest, src, chunk_shape, src_chunk_strides, dest_chunk_strides, chunk_size, aggregate_dimension);
    }

    for (size_t i = 0; i < dest_chunk_size; i++) {
        std::vector<size_t> ids;
        ids.resize(rank);
        size_t tmp = i;
        for (int32_t j = rank - 1; j >= 0; j--) {
            ids[j] = tmp / dest_chunk_strides[j];
            tmp    = tmp % dest_chunk_strides[j];
        }

        size_t zero_element_in_agg_dim_linear_id = 0;
        for (size_t j = 0; j < rank; j++) {
            zero_element_in_agg_dim_linear_id += (ids[j] * src_chunk_strides[j]);
        }

        if constexpr (opCode == AggOpCode::STDDEV && is_only_chunk) {
            dest[i]    = dest[i] / chunk_shape[aggregate_dimension];    // now contains the average
            size_t tmp = 0;
            for (size_t j = 0; j < chunk_shape[aggregate_dimension]; j++) {
                size_t current_offset =
                  zero_element_in_agg_dim_linear_id + (j * src_chunk_strides[aggregate_dimension]);
                size_t tmp2 = static_cast<VTRes>(src[current_offset] - dest[i]);
                tmp += (tmp2 * tmp2);
            }
            tmp     = tmp / (chunk_shape[aggregate_dimension] - 1);
            dest[i] = std::sqrt(tmp);
        } else if constexpr (opCode == AggOpCode::MEAN && is_only_chunk) {
            std::vector<VTRes> values;
            values.resize(chunk_shape[aggregate_dimension]);
            for (size_t j = 0; j < chunk_shape[aggregate_dimension]; j++) {
                size_t current_offset =
                  zero_element_in_agg_dim_linear_id + (j * src_chunk_strides[aggregate_dimension]);
                values[j] = static_cast<VTRes>(src[current_offset]);
            }
            std::sort(values.begin(), values.end());
            dest[i] = values[chunk_shape[aggregate_dimension] / 2];
        } else {
            dest[i] = static_cast<VTRes>(src[zero_element_in_agg_dim_linear_id]);

            for (size_t j = 1; j < chunk_shape[aggregate_dimension]; j++) {
                size_t current_offset =
                  zero_element_in_agg_dim_linear_id + (j * src_chunk_strides[aggregate_dimension]);

                if constexpr (opCode == AggOpCode::SUM) {
                    dest[i] += static_cast<VTRes>(src[current_offset]);
                } else if constexpr (opCode == AggOpCode::PROD) {
                    dest[i] *= static_cast<VTRes>(src[current_offset]);
                } else if constexpr (opCode == AggOpCode::MIN) {
                    dest[i] = std::min(dest[i], static_cast<VTRes>(src[current_offset]));
                } else if constexpr (opCode == AggOpCode::MAX) {
                    dest[i] = std::max(dest[i], static_cast<VTRes>(src[current_offset]));
                } else {
                    throw std::runtime_error("unsupported op_code reached");
                }
            }
        }
    }
}

template<Scalar_t VTRes, Scalar_t VTArg, AggOpCode opCode, bool is_only_chunk>
void AggChunk(VTRes* dest,
              VTArg* src,
              std::vector<size_t> chunk_shape,    // intentionally per value
              size_t chunk_size,
              const std::vector<bool>& aggregate_dimension,
              bool is_initial_chunk_in_dest_location) {
    size_t rank = chunk_shape.size();

    if (rank == 0) {
        dest[0] = static_cast<VTRes>(src[0]);
        return;
    }

    auto scratch_space = std::make_unique<VTRes[]>(2 * chunk_size);

    bool no_aggregation_dim = true;
    for (size_t i = 0; i < aggregate_dimension.size(); i++) {
        if (aggregate_dimension[i]) {
            no_aggregation_dim = false;
        }
    }
    if (no_aggregation_dim) {
        for (size_t i = 0; i < chunk_size; i++) {
            dest[i] = static_cast<VTRes>(src[i]);
        }
        return;
    }

    VTRes* current_dest = scratch_space.get();
    VTRes* current_src;    // used after the first swap
    size_t current_chunk_size = chunk_size;

    std::vector<size_t> src_chunk_strides;
    std::vector<size_t> dest_chunk_strides;
    src_chunk_strides.resize(rank);
    dest_chunk_strides.resize(rank);

    bool is_first_swap = true;
    for (size_t i = 0; i < chunk_shape.size(); i++) {
        // Ignore dims not flaged for reduction
        if (aggregate_dimension[i]) {
            // recalculate strides since they change in each iteration
            src_chunk_strides[0]  = 1;
            dest_chunk_strides[0] = 1;
            for (size_t j = 1; j < rank; j++) {
                src_chunk_strides[j] = src_chunk_strides[j - 1] * chunk_shape[j - 1];
                if ((j - 1) == i) {
                    dest_chunk_strides[j] = dest_chunk_strides[j - 1];
                } else {
                    dest_chunk_strides[j] = dest_chunk_strides[j - 1] * chunk_shape[j - 1];
                }
            }

            if (is_first_swap) {
                AggChunkSingleDim<VTRes, VTArg, opCode, is_only_chunk>(
                  current_dest, src, chunk_shape, src_chunk_strides, dest_chunk_strides, current_chunk_size, i);
                current_src  = current_dest;
                current_dest = &(scratch_space[chunk_size]);
            } else {
                AggChunkSingleDim<VTRes, VTRes, opCode, is_only_chunk>(
                  current_dest, current_src, chunk_shape, src_chunk_strides, dest_chunk_strides, current_chunk_size, i);
                std::swap(current_src, current_dest);
            }
            current_chunk_size = current_chunk_size / chunk_shape[i];
            chunk_shape[i]     = 1;

            is_first_swap = false;
        }
    }

    if (is_initial_chunk_in_dest_location) {
        std::memcpy(dest, current_src, sizeof(VTRes) * current_chunk_size);
    } else {
        switch (opCode) {
            using enum AggOpCode;
            case SUM:
                for (size_t i = 0; i < current_chunk_size; i++) {
                    dest[i] += current_src[i];
                }
                break;
            case PROD:
                for (size_t i = 0; i < current_chunk_size; i++) {
                    dest[i] *= current_src[i];
                }
                break;
            case MIN:
                for (size_t i = 0; i < current_chunk_size; i++) {
                    dest[i] = std::min(dest[i], current_src[i]);
                }
                break;
            case MAX:
                for (size_t i = 0; i < current_chunk_size; i++) {
                    dest[i] = std::max(dest[i], current_src[i]);
                }
                break;
            default:
                throw std::runtime_error("unsupported op_code reached");
                // Todo handle idxmin/max and mean stddev further up
        }
    }
}

template<Scalar_t VTRes, Scalar_t VTArg, bool is_only_chunk>
void AggChunkOPDispatch(AggOpCode opCode,
                        VTRes* dest,
                        VTArg* src,
                        const std::vector<size_t>& chunk_shape,
                        size_t chunk_size,
                        const std::vector<bool>& aggregation_dimension,
                        bool is_initial_chunk_in_dest_location) {
    switch (opCode) {
        using enum AggOpCode;
        case MIN:
            AggChunk<VTRes, VTArg, AggOpCode::MIN, is_only_chunk>(
              dest, src, chunk_shape, chunk_size, aggregation_dimension, is_initial_chunk_in_dest_location);
            break;
        case MAX:
            AggChunk<VTRes, VTArg, AggOpCode::MAX, is_only_chunk>(
              dest, src, chunk_shape, chunk_size, aggregation_dimension, is_initial_chunk_in_dest_location);
            break;
        case SUM:
            AggChunk<VTRes, VTArg, AggOpCode::SUM, is_only_chunk>(
              dest, src, chunk_shape, chunk_size, aggregation_dimension, is_initial_chunk_in_dest_location);
            break;
        case PROD:
            AggChunk<VTRes, VTArg, AggOpCode::PROD, is_only_chunk>(
              dest, src, chunk_shape, chunk_size, aggregation_dimension, is_initial_chunk_in_dest_location);
            break;
        default:
            // TODO: IDXmin/max, stddev mean
            throw std::runtime_error("unsupported op_code reached");
    }
}