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

#include <cstring>
#include <limits>
#include <memory>
#include <ostream>
#include <cstddef>
#include <tuple>
#include <vector>
#include <cstring>
#include <optional>

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Tensor.h>

template <typename ValueType>
class ContiguousTensor : public Tensor<ValueType> {
public:
  std::vector<size_t> strides;

  std::shared_ptr<ValueType[]> data;

  ContiguousTensor<ValueType>(const std::vector<size_t> &tensor_shape,
                              InitCode init_code)
      : Tensor<ValueType>(tensor_shape), {
    strides.resize(rank);
    if (rank > 0) {
      strides[0] = 1;
    }
    
    for (size_t i = 1; i < rank; i++) {
      strides[i] = strides[i-1] * tensor_shape[i-1];
    }

    data = std::make_shared<ValueType[]>(total_element_count);

    switch (init_code) {
    case NONE:
      break;
    case ZERO:
      for (size_t i = 0; i < total_element_count; i++) {
        data[i] = 0;
      }
      break;
    case MAX:
      for (size_t i = 0; i < total_element_count; i++) {
        data[i] = std::numeric_limits<ValueType>::max();
      }
      break;
    case MIN:
      for (size_t i = 0; i < total_element_count; i++) {
        data[i] = std::numeric_limits<ValueType>::min();
      }
      break;
    case IOTA:
      for (size_t i = 0; i < total_element_count; i++) {
        data[i] = i;
      }
      break;
    default:
      // unreachable
      std::abort();
      break;
    }
  };

  ContiguousTensor<ValueType>(const ContiguousTensor<ValueType> &other)
      : Tensor<ValueType>(other.tensor_shape), strides(other.strides), data(other.data) {};

  ContiguousTensor<ValueType>(ContiguousTensor<ValueType> &&other)
      : Tensor<ValueType>(other.tensor_shape), strides(other.strides) {
    data = std::move(other.data);
  };

  ContiguousTensor<ValueType>(const DenseMatrix<ValueType> &other)
      : Tensor<ValueType>(other.numRows, other.numCols), data(other.values) {
    strides = {1, other.numCols};
  }

  ContiguousTensor<ValueType>(DenseMatrix<ValueType> &&other)
      : Tensor<ValueType>(other.numRows, other.numCols) {
    strides = {1, other.numCols};
    data = std::move(other.values);
  }

  bool operator==(const ContiguousTensor<ValueType> &rhs) {
    if (tensor_shape != rhs.tensor_shape) {
      return false;
    }

    return !static_cast<bool>(std::memcmp(
        data.get(), rhs.data.get(), total_element_count * sizeof(ValueType)));
  }

  std::optional < DenseMatrix<ValueType> tryToGetDenseMatrix() const {
    if (rank != 2) {
      return std::nullopt;
    }

    return DenseMatrix<ValueType>(numRows, numCols, data);
  }

  std::optional<ValueType> tryGet(const std::vector<size_t> &element_indices) const {
    if (element_indices.size() != rank) {
      return std::nullopt;
    }
    if (rank == 0) {
      return data[0];
    }

    for (size_t i = 0; i < rank; i++) {
      if (element_indices[i] >= tensor_shape[i]) {
        return std::nullopt;
      }
    }

    size_t linear_id = element_indices[0];
    for (size_t i = 1; i < rank; i++) {
      linear_id += element_indices[i] * strides[i];
    }

    return data[linear_id];
  }
  ValueType get(const std::vector<size_t> &element_indices) const {
    if (rank == 0) {
      return data[0];
    }
    size_t linear_id = element_indices[0];
    for (size_t i = 1; i < rank; i++) {
      linear_id += element_indices[i] * strides[i];
    }
    return data[linear_id];
  }

  bool trySet(const std::vector<size_t> &element_indices, ValueType value) {
    if (element_indices.size() != rank || rank == 0) {
      return false;
    }

    for (size_t i = 0; i < rank; i++) {
      if (element_indices[i] >= tensor_shape[i]) {
        return false;
      }
    }

    size_t linear_id = element_indices[0];
    for (size_t i = 1; i < rank; i++) {
      linear_id += element_indices[i] * strides[i];
    }
    data[linear_id] = value;

    return true;
  }
  void set(const std::vector<size_t> &element_indices, ValueType value) {
    size_t linear_id = element_indices[0];
    for (size_t i = 1; i < rank; i++) {
      linear_id += element_indices[i] * strides[i];
    }
    data[linear_id] = value;
  }

  void print(std::ostream &os) const override {
    os << "ContiguousTensor with shape: [";
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

    for (size_t i = 0; i < total_element_count; i++) {
      if (i % tensor_shape[0] == 0) {
        os << "\n";
      }
      os << data[i] << " ";
    }
    os << std::endl;
  }

  //Ranges are inclusive on both boundaries
  std::optional<ContiguousTensor<ValueType>>
  dice(const std::vector<std::pair<size_t, size_t>> &index_ranges) const {
    if (index_ranges.size() != rank) {
      return std::nullopt;
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
      new_tensor_shape[i] = std::get<1>(index_ranges[i]) - std::get<0>(index_ranges[i]) + 1;
    }

    ContiguousTensor<ValueType> result(new_tensor_shape, NONE);

    std::vector<size_t> current_indices;
    current_indices.resize(rank);
    for (size_t i = 0; i < result.total_element_count; i++) {
      size_t tmp = i;

      for (int64_t j = rank-1; j >= 0; j--) {
        current_indices[static_cast<size_t>(j)] =
            (tmp / strides[static_cast<size_t>(j)]) +
            std::get<0>(index_ranges[static_cast<size_t>(j)]);
        tmp = tmp % strides[static_cast<size_t>(j)];
      }

      result.data[i] = get(current_indices);
    }

    return result;
  }

  //Removes all dimensions with a size of 1
  void reduceRank() {
    for (size_t i = 0; i < rank; i++) {
      if (tensor_shape[i] == 1) {
        tensor_shape.erase(tensor_shape.begin() + i);
        strides.erase(strides.begin() + i);
      }
    }
    rank = tensor_shape.size();
  }

  // serialize();
};
