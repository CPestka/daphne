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

#include "runtime/local/datastructures/DataObjectFactory.h"
#include "runtime/local/datastructures/Tensor.h"
#include "runtime/local/datastructures/ChunkedTensor.h"
#include "runtime/local/datastructures/ContiguousTensor.h"
#include <runtime/local/kernels/Agg.h>
#include <runtime/local/kernels/AggOpCode.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>

TEST_CASE("Agg-ContiguousTensor", TAG_KERNELS) {
    auto t1 = DataObjectFactory::create<ContiguousTensor<uint32_t>>(std::vector<size_t>({3,3,3}), InitCode::IOTA);
    auto tmod = DataObjectFactory::create<ContiguousTensor<uint64_t>>(std::vector<size_t>({3,3,2}), InitCode::IOTA);
    tmod->data[0] = 1;

    ContiguousTensor<uint32_t>* t2 = agg<ContiguousTensor<uint32_t>,ContiguousTensor<uint32_t>>(AggOpCode::SUM, std::vector<bool>({true,true,true}), t1, nullptr);
    ContiguousTensor<uint64_t>* t3 = agg<ContiguousTensor<uint64_t>,ContiguousTensor<uint64_t>>(AggOpCode::PROD, std::vector<bool>({true,true,true}), tmod, nullptr);
    ContiguousTensor<uint32_t>* t4 = agg<ContiguousTensor<uint32_t>,ContiguousTensor<uint32_t>>(AggOpCode::MIN, std::vector<bool>({true,true,true}), t1, nullptr);
    ContiguousTensor<uint32_t>* t5 = agg<ContiguousTensor<uint32_t>,ContiguousTensor<uint32_t>>(AggOpCode::MAX, std::vector<bool>({true,true,true}), t1, nullptr);
    
    REQUIRE((t2->tensor_shape == std::vector<size_t>({1,1,1})));
    REQUIRE(t3->data[0] == 355687428096000);
    REQUIRE(t4->data[0] == 0);
    REQUIRE(t5->data[0] == 26);
    
    auto tc = DataObjectFactory::create<ContiguousTensor<uint32_t>>(std::vector<size_t>({3,3,3}), InitCode::NONE);
    for(size_t i=0; i<27; i++) {
        tc->data[i] = 1;
    }

    ContiguousTensor<uint32_t>* tc1 = agg<ContiguousTensor<uint32_t>,ContiguousTensor<uint32_t>>(AggOpCode::SUM, std::vector<bool>({false,true,true}), tc, nullptr);
    REQUIRE(tc1->tensor_shape == std::vector<size_t>({3,1,1}));
    for(size_t i=0; i<3; i++) {
        REQUIRE(tc1->get({i,0,0}) == 9);
    }
}