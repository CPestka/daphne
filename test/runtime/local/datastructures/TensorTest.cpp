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

#include <pthread.h>
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <tags.h>

#include <catch.hpp>

#include <cstdint>
#include <vector>
#include <optional>
#include <memory>
#include <tuple>

TEMPLATE_TEST_CASE("Tensor Creation", TAG_DATASTRUCTURES, ALL_VALUE_TYPES) {
  // Just check if it does not crash
  
  std::vector<size_t> rank0_shape = {};
  std::vector<size_t> rank1_shape = {42};
  std::vector<size_t> rank2_shape = {42, 3};
  std::vector<size_t> rank3_shape = {42, 3, 16};
  std::vector<size_t> rank4_shape = {3, 17, 18, 9};

  std::vector<size_t> rank0_chunk_shape = {0};
  std::vector<size_t> rank1_chunk_shape = {3};
  std::vector<size_t> rank2_chunk_shape = {2,5};
  std::vector<size_t> rank3_chunk_shape = {7,1,15};
  std::vector<size_t> rank4_chunk_shape = {5,5,10,2};

  ContiguousTensor<TestType> *ct0 =
      DataObjectFactory::create<ContiguousTensor<TestType>>(rank0_shape, IOTA);
  ContiguousTensor<TestType> *ct1 =
      DataObjectFactory::create<ContiguousTensor<TestType>>(rank1_shape, IOTA);
  ContiguousTensor<TestType> *ct2 =
      DataObjectFactory::create<ContiguousTensor<TestType>>(rank2_shape, IOTA);
  ContiguousTensor<TestType> *ct3 =
      DataObjectFactory::create<ContiguousTensor<TestType>>(rank3_shape, IOTA);

  ChunkedTensor<TestType> *cht0 =
      DataObjectFactory::create<ChunkedTensor<TestType>>(
          rank0_shape, rank0_chunk_shape, IOTA);
  ChunkedTensor<TestType> *cht1 =
      DataObjectFactory::create<ChunkedTensor<TestType>>(
          rank1_shape, rank1_chunk_shape, IOTA);
  ChunkedTensor<TestType> *cht2 =
      DataObjectFactory::create<ChunkedTensor<TestType>>(
          rank2_shape, rank2_chunk_shape, IOTA);
  ChunkedTensor<TestType> *cht3 =
      DataObjectFactory::create<ChunkedTensor<TestType>>(
          rank3_shape, rank3_chunk_shape, IOTA);

  DataObjectFactory::destroy(ct0, ct1, ct2, ct3);
  DataObjectFactory::destroy(cht0,cht1,cht2,cht3);
}

template <typename T>
T getIOTAValue(const std::vector<size_t> &tensor_shape,
               const std::vector<size_t> &ids) {
  std::vector<size_t> strides;
  strides.resize(tensor_shape.size());
  strides[0] = 1;
  size_t result = ids[0];
  for (size_t i = 1; i < tensor_shape.size(); i++) {
    strides[i] = strides[i - 1] * tensor_shape[i - 1];
    result += strides[i] * ids[i];
  }
  return static_cast<T>(result);
}

TEMPLATE_TEST_CASE("Tensor layout and accessors", TAG_DATASTRUCTURES, double, float, uint32_t, uint64_t, int32_t, int64_t) {
  std::vector<size_t> tensor_shape0 = {4, 4, 4};
  std::vector<size_t> tensor_shape1 = {3, 2, 7, 5};

  std::vector<size_t> chunk_shape0 = {2, 2, 2};
  std::vector<size_t> chunk_shape1 = {2, 4, 2, 2};

  ContiguousTensor<TestType> *ct0 =
      DataObjectFactory::create<ContiguousTensor<TestType>>(tensor_shape0,
                                                            IOTA);
  ContiguousTensor<TestType> *ct1 =
      DataObjectFactory::create<ContiguousTensor<TestType>>(tensor_shape1,
                                                            IOTA);

  ChunkedTensor<TestType> *cht0 =
      DataObjectFactory::create<ChunkedTensor<TestType>>(tensor_shape0,
                                                         chunk_shape0, IOTA);
  ChunkedTensor<TestType> *cht1 =
      DataObjectFactory::create<ChunkedTensor<TestType>>(tensor_shape1,
                                                         chunk_shape1, IOTA);

  SECTION("Manual layout check") {
    TestType expected_chunked[64] = {0,  1,  4,  5,  16, 17, 20, 21,

                                     2,  3,  6,  7,  18, 19, 22, 23,

                                     8,  9,  12, 13, 24, 25, 28, 29,

                                     10, 11, 14, 15, 26, 27, 30, 31,

                                     32, 33, 36, 37, 48, 49, 42, 43,

                                     34, 35, 38, 39, 50, 51, 54, 55,

                                     42, 43, 46, 47, 58, 59, 62, 63};

    for (int i = 0; i < 64; i++) {
      REQUIRE(cht0->data.get()[i] == expected_chunked[i]);
    }
  }

  SECTION(".get()") {
    for (size_t i = 0; i < 4; i++) {
      for (size_t j = 0; j < 4; j++) {
        for (size_t k = 0; k < 4; k++) {
          std::vector<size_t> ids = {i,j,k};
          REQUIRE(ct0->get(ids) == getIOTAValue<TestType>(tensor_shape0, ids));
          REQUIRE(cht0->get(ids) == getIOTAValue<TestType>(tensor_shape0, ids));
        }
      }
    }

    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 2; j++) {
        for (size_t k = 0; k < 7; k++) {
          for (size_t n = 0; n < 5; n++) {
            std::vector<size_t> ids = {i,j,k,n};
            REQUIRE(ct1->get(ids) == getIOTAValue<TestType>(tensor_shape1, ids));
            REQUIRE(cht1->get(ids) == getIOTAValue<TestType>(tensor_shape1, ids));
          }
        }
      }
    }
  }

  SECTION(".set()") {
    size_t test_value = 0;
    for (size_t i = 0; i < 4; i++) {
      for (size_t j = 0; j < 4; j++) {
        for (size_t k = 0; k < 4; k++) {
          std::vector<size_t> ids = {i,j,k};
          ct0->set(ids, test_value);
          REQUIRE(ct0->get(ids) == static_cast<TestType>(test_value));
          cht0->set(ids, test_value);
          REQUIRE(cht0->get(ids) == static_cast<TestType>(test_value));
    
          test_value += 42;
        }
      }
    }

    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 2; j++) {
        for (size_t k = 0; k < 7; k++) {
          for (size_t n = 0; n < 5; n++) {
            std::vector<size_t> ids = {i,j,k,n};
            ct1->set(ids, test_value);
            REQUIRE(ct1->get(ids) == static_cast<TestType>(test_value));
            cht1->set(ids, test_value);
            REQUIRE(cht1->get(ids) == static_cast<TestType>(test_value));
    
            test_value += 42;
          }
        }
      }
    }
  }

  DataObjectFactory::destroy(ct0, ct1);
  DataObjectFactory::destroy(cht0, cht1);
}

TEMPLATE_TEST_CASE("Tensor dicing, rechunking and conversions", TAG_DATASTRUCTURES,
                    double, float, uint32_t, uint64_t, int32_t, int64_t) {
  std::vector<size_t> tensor_shape = {3, 4, 2};
  std::vector<size_t> chunk_shape = {2, 2, 2};

  ContiguousTensor<TestType> *ct =
      DataObjectFactory::create<ContiguousTensor<TestType>>(tensor_shape,
                                                            IOTA);
  ChunkedTensor<TestType> *cht =
      DataObjectFactory::create<ChunkedTensor<TestType>>(tensor_shape,
                                                         chunk_shape, IOTA);

  SECTION(".dice() and variants") {
    std::vector<std::pair<size_t,size_t>> dice_range = {{1,2},{3,3},{0,1}};
    std::optional<ContiguousTensor<TestType>> dice0 = ct->tryDice(dice_range);

    REQUIRE(dice0);
    REQUIRE(dice0.value().data.get()[0] == 10);
    REQUIRE(dice0.value().data.get()[1] == 11);
    REQUIRE(dice0.value().data.get()[2] == 12);
    REQUIRE(dice0.value().data.get()[3] == 13);

    std::optional<ChunkedTensor<TestType>> dice1 = cht->tryDice(dice_range,chunk_shape);

    REQUIRE(dice1.value().get({0, 0, 0}) == 10);
    REQUIRE(dice1.value().get({1, 0, 0}) == 11);
    REQUIRE(dice1.value().get({0, 0, 1}) == 12);
    REQUIRE(dice1.value().get({1, 0, 1}) == 13);

    std::optional<ContiguousTensor<TestType>> dice3 =
        cht->tryDiceToContiguousTensor(dice_range);

    REQUIRE(dice3);
    REQUIRE(dice3.value().data.get()[0] == 10);
    REQUIRE(dice3.value().data.get()[1] == 11);
    REQUIRE(dice3.value().data.get()[2] == 12);
    REQUIRE(dice3.value().data.get()[3] == 13);

    std::vector<std::pair<size_t, size_t>> dice_chunk_range = {
        {0, 0}, {1, 1}, {0, 0}};
    std::optional<ChunkedTensor<TestType>> dice4 =
        cht->tryDice(dice_chunk_range,chunk_shape);

    REQUIRE(dice4);
    REQUIRE(dice4.value().data.get()[0] == 6);
    REQUIRE(dice4.value().data.get()[1] == 7);
    REQUIRE(dice4.value().data.get()[2] == 9);
    REQUIRE(dice4.value().data.get()[3] == 10);
    REQUIRE(dice4.value().data.get()[4] == 18);
    REQUIRE(dice4.value().data.get()[5] == 19);
    REQUIRE(dice4.value().data.get()[6] == 21);
    REQUIRE(dice4.value().data.get()[7] == 22);
  }

  SECTION(".rechunk()") {
    ChunkedTensor<TestType> *cht1 =
      DataObjectFactory::create<ChunkedTensor<TestType>>(*cht);

    REQUIRE(cht1->tryRechunk({2,4,2}));

    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 4; j++) {
        for (size_t k = 0; k < 2; k++) {
          REQUIRE(cht->get({i,j,k}) == cht1->get({i,j,k}));
        }
      }
    }

    DataObjectFactory::destroy(cht1);
  }

  SECTION("Converion: contiguous -> chunked") {
    ChunkedTensor<TestType> *cht1 =
        DataObjectFactory::create<ChunkedTensor<TestType>>(*ct);

    REQUIRE((*cht1) == (*ct));

    DataObjectFactory::destroy(cht1);
  }

  SECTION("Converion: DenseMatrix -> rank 2 ContiguousTensor") {
    std::shared_ptr<TestType[]> data(new TestType[12],
                                     std::default_delete<TestType[]>());
    for (size_t i = 0; i < 12; i++) {
      data[i] = i;
    }

    DenseMatrix<TestType> *matrix =
        DataObjectFactory::create<DenseMatrix<TestType>>(3, 4, data);
    ContiguousTensor<TestType> *ct1 =
        DataObjectFactory::create<ContiguousTensor<TestType>>(*matrix);

    for (size_t i = 0; i < 4; i++) {
      for (size_t j = 0; j < 3; j++) {
        REQUIRE(matrix->get(i,j) == static_cast<size_t>(i + 4 * j));
        REQUIRE(matrix->get(i,j) == ct1->get({i,j}));
      }
    }

    DataObjectFactory::destroy(matrix);
    DataObjectFactory::destroy(ct1);  
  }

  SECTION("Converion: rank 2 ContiguousTensor -> DenseMatrix") {
    ContiguousTensor<TestType> *ct1 =
        DataObjectFactory::create<ContiguousTensor<TestType>>(std::vector<size_t>({4, 3}), IOTA);
    ContiguousTensor<TestType> *ct2 =
        DataObjectFactory::create<ContiguousTensor<TestType>>(std::vector<size_t>({4,3,3}),IOTA);

    REQUIRE(!(ct2->tryToGetDenseMatrix()));
    std::optional<DenseMatrix<TestType>> matrix = ct1->tryToGetDenseMatrix();
    REQUIRE(matrix);

    for (size_t i = 0; i < 4; i++) {
      for (size_t j = 0; j < 3; j++) {
        REQUIRE(matrix.value().get(i,j) == ct1->get({i,j}));
      }
    }

    DataObjectFactory::destroy(ct1,ct2);
  }

  DataObjectFactory::destroy(ct);
  DataObjectFactory::destroy(cht);
}