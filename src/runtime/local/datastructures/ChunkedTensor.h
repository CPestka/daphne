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

#include "ContiguousTensor.h"
#include <memory>
#include <ostream>
#include <cstddef>
#include <tuple>
#include <vector>
#include <limits>
#include <optional>
#include <tuple>

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Tensor.h>
#include <runtime/local/datastructures/ContiguousTensor.h>

enum InitCode { NONE, ZERO, MAX, MIN, IOTA};

template <typename ValueType>
class ChunkedTensor : public Tensor {
  std::vector<size_t> chunk_shape;
  size_t chunk_element_count;
  std::vector<size_t> chunk_strides;
  std::vector<size_t> intra_chunk_strides;
  std::vector<size_t> chunks_per_dim;
  // In this implementation overhanging chunks are stored as full chunks instead
  // of partial chunks. This also means that total_size_in_elements and
  // total_element_count are not neccesarily the same
  size_t total_size_in_elements;
  size_t total_chunk_count;

  std::shared_ptr<ValueType> data;

  ChunkedTensor<ValueType>(const std::vector<size_t> &tensor_shape,
                           const std::vector<size_t> &chunk_shape,
                           InitCode init_code)
      : Tensor<ValueType>(tensor_shape), chunk_shape(chunk_shape) {
    dim_strides.resize(rank);
    chunk_strides.resize(rank);

    if (rank > 0) {
      intra_chunk_strides[0] = 1;
    }
    for (size_t i = 1; i < rank; i++) {
      intra_chunk_strides[i] = intra_chunk_strides[i-1] * chunk_shape[i-1];
    }

    chunk_element_count = rank > 0 ? chunk_shape[0] : 1;
    for (size_t i = 1; i < rank; i++) {
      chunk_element_count *=  chunk_shape[i];
    }

    total_chunk_count = 0;
    for (size_t i = 0; i < rank; i++) {
      chunks_per_dim[i] = tensor_shape[i] % chunk_shape[i] == 0
                              ? tensor_shape[i] / chunk_shape[i]
                              : (tensor_shape[i] / chunk_shape[i]) + 1;
      total_chunk_count =
          i == 0 ? chunks_per_dim[0] : total_chunk_count * chunks_per_dim[i];
      chunk_strides[i] = i == 0 ? chunk_element_count
                                : chunk_strides[i - 1] * chunks_per_dim[i - 1];
    }

    total_size_in_elements = total_chunk_count * chunk_element_count;

    data = std::make_shared<ValueType[]>(total_size_in_elements);

    // switch (init_code) {
    // case NONE:
    //   break;
    // case ZERO:
    //   for (size_t i = 0; i < total_element_count; i++) {
    //     data[i] = 0;
    //   }
    //   break;
    // case MAX:
    //   for (size_t i = 0; i < total_element_count; i++) {
    //     data[i] = std::numeric_limits<ValueType>::max();
    //   }
    //   break;
    // case MIN:
    //   for (size_t i = 0; i < total_element_count; i++) {
    //     data[i] = std::numeric_limits<ValueType>::min();
    //   }
    //   break;
    // case IOTA:
    //   for (size_t i = 0; i < total_element_count; i++) {
    //     data[i] = i;
    //   }
    //   break;
    // default:
    //   // unreachable
    //   std::abort();
    //   break;
    // }
  }

  //Copies data
  explicit ChunkedTensor<ValueType>(const ChunkedTensor<ValueType> &other)
      : Tensor<ValueType>(other.tensor_shape), chunk_shape(other.chunk_shape),
        chunk_element_count(other.chunk_element_count),
        chunk_strides(other.chunk_strides),
        intra_chunk_strides(other.intra_chunk_strides),
        chunks_per_dim(other.chunks_per_dim),
        total_size_in_elements(other.total_size_in_elements),
        total_chunk_count(other.total_chunk_count) {
    data = std::make_shared<ValueType[]>(total_size_in_elements);
    std::memcpy(data.get(), other.data.get(), total_size_in_elements * sizeof(ValueType));
  }

  explicit ChunkedTensor<ValueType>(ChunkedTensor<ValueType> &&other)
      : Tensor<ValueType>(other.tensor_shape), chunk_shape(other.chunk_shape),
        chunk_element_count(other.chunk_element_count),
        chunk_strides(other.chunk_strides),
        intra_chunk_strides(other.intra_chunk_strides),
        chunks_per_dim(other.chunks_per_dim),
        total_size_in_elements(other.total_size_in_elements),
        total_chunk_count(other.total_chunk_count), data(std::move(other.data)){};

  ChunkedTensor<ValueType>(const DenseMatrix<ValueType> &matrix,
                           size_t chunk_size_x, size_t chunk_size_y)
      : Tensor<ValueType>(other.numRows, other.numCols) {
    chunk_shape = {chunk_size_x, chunk_size_y};
    chunk_element_count = chunk_size_x * chunk_size_y;
    chunks_per_dim = {
      tensor_shape[0] % chunk_size_x == 0
          ? tensor_shape[0] / chunk_size_x
          : 1 + (tensor_shape[0] / chunk_size_x),
      tensor_shape[1] % chunk_size_y == 0 ? tensor_shape[1] / chunk_size_y
                                          : 1 + (tensor_shape[1] / chunk_size_y)}
    intra_chunk_strides = {1, chunk_size_x};
    chunk_strides = {chunk_element_count,
                     chunk_element_count * chunks_per_dim[0]};
    total_chunk_count = chunks_per_dim[0] * chunks_per_dim[1];
    total_size_in_elements = total_chunk_count * chunk_element_count;

    data = std::make_shared<ValueType[]>(total_size_in_elements);

    for (size_t i = 0; i < numCols; i++) {
      for (size_t j = 0; j < numRows; j++) {
        std::vector<size_t> ids = {i, j};
        set(ids, other.get(j,i));
      }
    }
  }

  //TODO:
  //from list of contiguous;from contiguous; as freestanding since they can fail

  bool operator==(const ChunkedTensor<ValueType> &rhs) const {
    if (tensor_shape != rhs.tensor_shape ||
        chunk_shape != rhs.chunk_shape) {
      return false;
    }

    std::vector<size_t> linear_strides;
    linear_strides.resize(rank);
    linear_stides[0] = 1;
    for (size_t i = 1; i < rank; i++) {
      linear_strides[i] = linear_stride[i-1] * tensor_shape[i-1];
    }
    std::vector<size_t> current_ids;
    current_ids.resize(rank);
    for (size_t i = 0; i < total_element_count; i++) {
      size_t tmp = i;
      for (int64_t j = rank-1; j >= 0; j--) {
        current_ids[static_cast<size_t>(j)] = tmp / linear_strides[static_cast<size_t>(j)];
        tmp = tmp % linear_strides[static_cast<size_t>(j)];
      }

      if (get(current_ids) != rhs.get(current_ids)) {
        return false;
      }
    }

    return true;
  }

  size_t getLinearId(const std::vector<size_t> &indices) const {
    size_t chunk_id = indices[0] / chunk_shape[0];
    size_t intra_chunk_id = indices[0] % chunk_shape[0];
    size_t linear_id = intra_chunk_id + chunk_strides[0] * chunk_id;

    for (size_t i = 0; i < rank; i++) {
      chunk_id = indices[i] / chunk_shape[i];
      intra_chunk_id = indices[i] % chunk_shape[i];
      linear_id += (intra_chunk_id * intra_chunk_strides[i]) + (chunk_id * chunk_strides[i]);
    }

    return linear_id;
  }

  size_t getLinearIdFromChunkIds(const std::vector<size_t> &chunk_indices) const {
    size_t linear_id = chunk_indices[0] * chunk_strides[0];
    for (size_t i = 0; i < rank; i++) {
      linear_id += (chunk_indices[i] * chunk_strides[i]);
    }

    return linear_id;
  }
  
  std::optional<ValueType> tryGet(const std::vector<size_t> &indices) const {
    if (indices.size() != rank) {
      return std::nullopt;
    }
    if (rank == 0) {
      return data[0];
    }

    for (size_t i = 0; i < rank; i++) {
      if (indices[i] >= tensor_shape[i]) {
        return std::nullopt;
      }
    }

    return data[getLinearId(indices)];
  }
  ValueType get(const std::vector<size_t> &indices) const {
    if (rank == 0) {
      return data[0];
    }

    return data[getLinearId(indices)];
  }

  ValueType* tryGetPtrToChunk(const std::vector<size_t> &chunk_indices) const {
    if (chunk_indices != rank) {
      return nullptr;
    }
    if (rank == 0) {
      return data.get();
    }
    for (size_t i = 0; i < rank; i++) {
      if (chunk_indices[i] >= chunks_per_dim[i]) {
        return nullptr;
      }
    }

    return &data[getLinearIdFromChunkIds(chunk_indices)];
  }
  ValueType* getPtrToChunk(const std::vector<size_t> &chunk_indices) const {
    if (rank == 0) {
      return data.get();
    }

    return &data[getLinearIdFromChunkIds(chunk_indices)];
  }

  std::optional<std::shared_ptr<ValueType[]>>
  tryGetChunk(const std::vector<size_t> &chunk_indices) const {
    if (chunk_indices != rank) {
      return std::nullopt;
    }
    if (rank == 0) {
      auto result = std::make_shared<ValueType[]>(1);
      result[0] = data[0];
      return result;
    }

    for (size_t i = 0; i < rank; i++) {
      if (chunk_indices[i] >= chunks_per_dim[i]) {
        return std::nullopt;
      }
    }

    size_t linear_id = getLinearIdFromChunkIds(chunk_indices);

    auto result = std::make_shared<ValueType[]>(chunk_element_count);
    std::memcpy(result.get(), &data[linear_id],
                chunk_element_count * sizeof(ValueType));
    return result;
  }
  std::shared_ptr<ValueType[]>
  getChunk(const std::vector<size_t> &chunk_indices) const {
    if (rank == 0) {
      auto result = std::make_shared<ValueType[]>(1);
      result[0] = data[0];
      return result;
    }

    size_t linear_id = getLinearIdFromChunkIds(chunk_indices);

    auto result = std::make_shared<ValueType[]>(chunk_element_count);
    std::memcpy(result.get(), &data[linear_id],
                chunk_element_count * sizeof(ValueType));
    return result;
  }

  bool trySet(const std::vector<size_t> &indices, ValueType value) {
    if (indices.size() != rank) {
      return false;
    }
    if (rank == 0) {
      data[0] = value;
      return true;
    }

    for (size_t i = 0; i < rank; i++) {
      if (indices[i] >= tensor_shape[i]) {
        return false;
      }
    }

    data[getLinearId(indices)] = value;
    return true;
  }
  void set(const std::vector<size_t> &indices, ValueType value) {
    if (rank == 0) {
      data[0] = value;
    }
    data[getLinearId(indices)] = value;
  }

  bool trySetChunk(const std::vector<size_t> &chunk_indices, ValueType* values) {
    if (chunk_indices.size() != rank) {
      return false;
    }
    if (rank == 0) {
      data[0] = value;
      return true;
    }

    for (size_t i = 0; i < rank; i++) {
      if (chunk_indices[i] >= tensor_shape[i]) {
        return false;
      }
    }

    size_t linear_id = getLinearIdFromChunkIds(chunk_indices);

    std::memcpy(&data[linear_id], values,
                chunk_element_count * sizeof(ValueType));
    return true;
  }
  void setChunk(const std::vector<size_t> &chunk_indices, ValueType* values) {
    if (rank == 0) {
      data[0] = value;
    }

    size_t linear_id = getLinearIdFromChunkIds(chunk_indices);

    std::memcpy(&data[linear_id], values,
                chunk_element_count * sizeof(ValueType));
  }

  bool tryRechunk(const std::vector<size_t> &new_chunk_shape) {
    if (new_chunk_shape.size() != rank) {
      return false;
    }
    if (rank == 0) {
      return true;
    }

    size_t new_chunk_element_count = new_chunk_shape[0];
    for (size_t i = 1; i < rank; i++) {
      new_chunk_element_count *=  new_chunk_shape[i];
    }

    std::vector<size_t> new_intra_chunk_strides = 1;
    for (size_t i = 1; i < rank; i++) {
      new_intra_chunk_strides[i] = new_intra_chunk_strides[i-1] * new_chunk_shape[i-1];
    }

    std::vector<size_t> new_chunk_strides;
    std::vector<size_t> new_chunks_per_dim;
    size_t new_total_cunk_count = 0;
    for (size_t i = 0; i < rank; i++) {
      new_chunks_per_dim[i] = tensor_shape[i] % new_chunk_shape[i] == 0
                              ? tensor_shape[i] / new_chunk_shape[i]
                              : (tensor_shape[i] / new_chunk_shape[i]) + 1;
      new_total_cunk_count =
          i == 0 ? new_chunks_per_dim[0] : new_total_cunk_count * new_chunks_per_dim[i];
      new_chunk_strides[i] = i == 0 ? new_chunk_element_count
                                : new_chunk_strides[i - 1] * new_chunks_per_dim[i - 1];
    }
  
    size_t new_total_size_in_elements = new_total_cunk_count * new_chunk_element_count;
    std::shared_ptr<ValueType[]> new_data =
        std::make_shared<ValueType[]>(new_total_size_in_elements);

    std::vector<size_t> chunk_count_strides;
    chunk_count_strides.push_back(1);
    for (size_t i = 1; i < rank; i++) {
      chunk_count_strides.push_back(chunk_count_strides[i-1] * chunks_per_dim[i-1]);
    }
    
    std::vector<size_t> current_old_chunk_ids;
    std::vector<size_t> current_old_element_ids;
    current_old_chunk_ids.resize(rank);
    current_old_intra_chunk_ids.resize(rank);
    for (size_t i = 0; i < total_chunk_count; i++) {
      size_t tmp = i;
      for (int64_t j = rank-1; j >= 0; j--) {
        current_old_chunk_ids[static_cast<size_t>(j)] = tmp / chunk_count_strides[static_cast<size_t>(j)];
        tmp = tmp % chunk_count_strides[static_cast<size_t>(j)];
      }

      size_t current_chunk_offset = 0;
      for (size_t k = 0; k < rank; k++) {
        current_chunk_offset += chunk_strides[k] * current_old_chunk_ids[k];
      }

      for (size_t j = 0; j < chunk_element_count; j++) {
        tmp = j;
        for (int64_t k = rank-1; k >= 0; k--) {
          current_old_element_ids[static_cast<size_t>(k)] =
              tmp / intra_chunk_strides[static_cast<size_t>(k)] +
              current_old_chunk_ids[k] * chunk_strides[k];
          tmp = tmp % intra_chunk_strides[static_cast<size_t>(k)];
        }

        // Bounds check for partial chunks
        bool out_of_bounds = false;
        for (size_t k = 0; k < rank; k++) {
          if (current_old_element_ids[k] >= tensor_shape[k]) {
            out_of_bounds = true;
          }
        }

        if (out_of_bounds) {
          continue;
        }

        size_t new_linear_id = 0;
        for (size_t k = 0; k < rank; k++) {
          size_t current_new_chunk_id =
              current_old_element_ids[k] / new_chunk_shape[k];
          size_t current_new_intra_chunk_id = current_old_element_ids[k] % new_chunk_shape[k];

          new_linear_id +=
              current_new_chunk_id * new_chunk_strides[k] +
              current_new_intra_chunk_id * new_intra_chunk_strides[k];
        }

        new_data[new_linear_id] = data[current_chunk_offset + j];
      }
    }

    chunk_shape = new_chunk_shape;
    chunk_element_count = new_chunk_element_count;
    chunk_strides = new_chunk_strides;
    intra_chunk_strides = new_intra_chunk_strides;
    chunks_per_dim = new_chunks_per_dim;
    total_size_in_elements = new_total_size_in_elements;
    total_chunk_count = new_total_cunk_count;

    data = new_data;

    return true;
  }

  //Ranges are inclusive on both boundaries
  std::optional<ChunkedTensor<ValueType>>
  tryDiceAtChunkLvl(const std::vector<std::pair<size_t, size_t>> &chunk_ranges) const {
    if (chunk_ranges.size() != rank) {
      return std::nullopt;
    }
    if (rank == 0) {
      auto tmp = ChunkedTensor<ValueType>(tensor_shape, chunk_shape, NONE);
      tmp.data[0] = data[0];
      return tmp;
    }

    for (size_t i = 0; i < rank; i++) {
      if (std::get<0>(chunk_ranges[i]) >= tensor_shape[i] ||
          std::get<1>(chunk_ranges[i]) >= tensor_shape[i] ||
          std::get<0>(chunk_ranges[i]) > std::get<1>(chunk_ranges[i])) {
        return std::nullopt;  
      }
    }

    std::vector<size_t> new_tensor_shape;
    new_tensor_shape.resize(rank);
    for (size_t i = 0; i < rank; i++) {
      new_tensor_shape[i] =
          (std::get<1>(chunk_ranges[i]) - std::get<0>(chunk_ranges[i]) + 1) *
          chunk_shape[i];
    }

    auto new_tensor =
        ChunkedTensor<ValueType>(new_tensor_shape, chunk_shape, NONE);

    std::vector<size_t> new_chunk_count_strides;
    new_chunk_count_strides.push_back(1);
    for (size_t i = 1; i < rank; i++) {
      new_chunk_count_strides.push_back(new_chunk_count_strides[i-1] * new_tensor.chunks_per_dim[i-1]);
    }

    std::vector<size_t> new_current_chunk_ids;
    std::vector<size_t> old_current_chunk_ids;
    new_current_chunk_ids.resize();
    old_current_chunk_ids.resize();
    for (size_t i = 0; i < new_tensor.total_chunk_count; i++) {
      size_t tmp = i;

      for (int64_t j = rank-1; j >= 0; j--) {
        new_current_chunk_ids[static_cast<size_t>(j)] =
            tmp / new_chunk_count_strides[static_cast<size_t>(j)];
        old_current_chunk_ids[static_cast<size_t>(j)] =
            new_current_chunk_ids[static_cast<size_t>(j)] +
            std::get<0>(chunk_ranges[static_cast<size_t>(j)]);
        tmp = tmp % new_chunk_count_strides[static_cast<size_t>(j)];
      }

      ValueType* ptr_to_old_chunk = getPtrToChunk(old_current_chunk_ids);
      ValueType* ptr_to_new_chunk =
          new_tensor.getPtrToChunk(new_current_chunk_ids);

      std::memcpy(ptr_to_new_chunk, ptr_to_old_chunk, chunk_element_count * sizeof(ValueType));
    }

    return new_tensor;
  }

  //Ranges are inclusive on both boundaries
  std::optional<ChunkedTensor<ValueType>>
  tryDice(const std::vector<std::pair<size_t, size_t>> &index_ranges,
          const std::vector<size_t> &new_chunk_shape) const {
    if (index_ranges != rank || new_chunk_shape != rank) {
      return std::nullopt;
    }

    if (rank == 0) {
      auto tmp = ChunkedTensor<ValueType>(tensor_shape, chunk_shape, NONE);
      tmp.data[0] = data[0];
      return tmp;
    }
    
    for (size_t i = 0; i < rank; i++) {
      if (std::get<0>(index_ranges[i]) >= tensor_shape[i] ||
          std::get<1>(index_ranges[i]) >= tensor_shape[i] ||
          std::get<0>(index_ranges[i]) > std::get<1>(index_ranges[i])) {
        return std::nullopt;  
      }
    }

    std::vector<size_t> new_tensor_shape;
    new_tensor_shape.resize(rank);
    for (size_t i = 0; i < rank; i++) {
      new_tensor_shape[i] =
          std::get<1>(index_ranges[i]) - std::get<0>(index_ranges[i]) + 1;
    }

    auto new_tensor =
        ChunkedTensor<ValueType>(new_tensor_shape, new_chunk_shape, NONE);

    std::vector<size_t> new_chunk_count_strides;
    new_chunk_count_strides.push_back(1);
    for (size_t i = 1; i < rank; i++) {
      new_chunk_count_strides.push_back(new_chunk_count_strides[i-1] * new_tensor.chunks_per_dim[i-1]);
    }

    std::vector<size_t> current_new_chunk_ids;
    std::vector<size_t> current_old_element_ids;
    std::vector<size_t> current_new_element_ids;
    current_new_chunk_ids.resize(rank);
    current_old_element_ids.resize(rank);
    current_new_element_ids.resize(rank);
    for (size_t i = 0; i < new_tensor.total_chunk_count; i++) {
      size_t tmp = i;
      for (int64_t j = rank-1; j >= 0; j--) {
        current_new_chunk_ids[static_cast<size_t>(j)] = tmp / new_chunk_count_strides[static_cast<size_t>(j)];
        tmp = tmp % new_chunk_count_strides[static_cast<size_t>(j)];
      }

      for (size_t j = 0; j < new_tensor.chunk_element_count; j++) {
        tmp = j;
        for (int64_t k = rank-1; k >= 0; k--) {
          current_new_element_ids[static_cast<size_t>(k)] =
              tmp / new_tensor.intra_chunk_strides[static_cast<size_t>(k)] +
              current_new_chunk_ids[k] * new_tensor.chunk_strides[k];
          tmp = tmp % new_tensor.intra_chunk_strides[static_cast<size_t>(k)];
        }

        // Bounds check for partial chunks
        bool out_of_bounds = false;
        for (size_t k = 0; k < rank; k++) {
          if (current_new_element_ids[k] >= new_tensor.tensor_shape[k]) {
            out_of_bounds = true;
          }
        }

        if (out_of_bounds) {
          continue;
        }

        for (size_t k = 0; k < rank; k++) {
          current_old_element_ids[k] = current_new_element_ids[k] + std::get<0>(index_ranges[k]);
        }

        new_tensor.set(current_new_element_ids, get(current_old_element_ids));
      }
    }

    return new_tensor;
  }

  //Ranges are inclusive on both boundaries
  std::optional<ContiguousTensor<ValueType>> tryDiceToContiguousTensor const (
      const std::vector<std::pair<size_t, size_t>> &index_ranges) {
    if (index_ranges != rank) {
      return std::nullopt;
    }

    if (rank == 0) {
      auto tmp = ContiguousTensor<ValueType>(tensor_shape, NONE);
      tmp.data[0] = data[0];
      return tmp;
    }
    
    for (size_t i = 0; i < rank; i++) {
      if (std::get<0>(index_ranges[i]) >= tensor_shape[i] ||
          std::get<1>(index_ranges[i]) >= tensor_shape[i] ||
          std::get<0>(index_ranges[i]) > std::get<1>(index_ranges[i])) {
        return std::nullopt;  
      }
    }

    auto new_tensor = ContiguousTensor<ValueType>(tensor_shape, NONE);

    std::vector<size_t> current_new_indices;
    std::vector<size_t> current_old_indices;
    current_new_indices.resize(rank);
    current_old_indices.resize(rank);
    for (size_t i = 0; i < new_tensor.total_element_count; i++) {
      size_t tmp = i;

      for (int64_t j = rank-1; j >= 0; j--) {
        current_new_indices[static_cast<size_t>(j)] =
            (tmp / new_tensor.strides[static_cast<size_t>(j)]);
        current_old_indices[static_cast<size_t>(j)] =
            current_new_indices[static_cast<size_t>(j)] +
            std::get<0>(index_ranges[static_cast<size_t>(j)]);
        tmp = tmp % new_tensor.strides[static_cast<size_t>(j)];
      }

      new_tensor.set(current_new_indices, get(current_old_indices));
    }

    return new_tensor;
  }

  //Prints elements in logical layout
  void print(std::ostream &os) const override {
    os << "ChunkedTensor with shape: [";
    for (size_t i = 0; i < rank; i++) {
      os << tensor_shape[i];
      if (i != rank - 1) {
        os << ",";
      }
    }
    os << "]\n"
       << "Elementtype: " << ValueTypeUtils::cppNameFor<ValueType> << std::endl;
    
    if (rank == 0) {
      os << data[0] << std::endl;
      return;
    }

    std::vector<size_t> current_ids;
    current_ids.resize(rank);
    std::vector<size_t> linear_strides;
    linear_strides.resize(rank);
    linear_stides[0] = 1;
    for (size_t i = 1; i < rank; i++) {
      linear_strides[i] = linear_stride[i-1] * tensor_shape[i-1];
    }
    for (size_t i = 0; i < total_element_count; i++) {
      if (i % tensor_shape[0] == 0) {
        os << "\n";
      }

      size_t tmp = i;
      for (int64_t j = rank-1; j >= 0; j--) {
        current_ids[static_cast<size_t>(j)] = tmp / linear_strides[static_cast<size_t>(j)];
        tmp = tmp % linear_strides[static_cast<size_t>(j)];
      }

      os << get(current_ids) << " ";
    }
    os << std::endl;
  }

  //serialize();
};

bool areLogicalElementsEqual(
    const ContiguousTensor<ValueType> &contiguous_tensor,
    const ChunkedTensor<ValueType> &chunked_tensor) {
  if (contiguous_tensor.tensor_shape != chunked_tensor.tensor_shape) {
      return false;
  }
  
  std::vector<size_t> linear_strides;
  linear_strides.resize(chunked_tensor.rank);
  linear_stides[0] = 1;
  for (size_t i = 1; i < chunked_tensor.rank; i++) {
    linear_strides[i] = linear_stride[i-1] * chunked_tensor.tensor_shape[i-1];
  }
  std::vector<size_t> current_ids;
  current_ids.resize(chunked_tensor.rank);
  for (size_t i = 0; i < chuked_tensor.total_element_count; i++) {
    size_t tmp = i;
    for (int64_t j = chunked_tensor.rank-1; j >= 0; j--) {
      current_ids[static_cast<size_t>(j)] = tmp / linear_strides[static_cast<size_t>(j)];
      tmp = tmp % linear_strides[static_cast<size_t>(j)];
    }

    if (get(current_ids) != rhs.get(current_ids)) {
      return false;
    }
  }

  return true;
}