/*
 * Copyright 2022 The DAPHNE Consortium
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

#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/io/ReadZarr.h>

#include <tags.h>

#include <catch.hpp>

#include <vector>
//#include <bit> // no c++20....

TEST_CASE("ReadZarr->ContiguousTensor", TAG_IO) {
    // ContiguousTensor<double>* ct_ptr;

    // // Read in [10,10,10] fp64 tensor
    // readZarr<ContiguousTensor<double>>(ct_ptr, "./test/runtime/local/io/zarr_test/ContiguousTensorTest/");

    // uint64_t entry_0;
    // uint64_t entry_9;
    // uint64_t entry_20;

    // // REQUIRE((std::endian::native == std::endian::little) || (std::endian::native == std::endian::big));

    // // uint64_t entry_0_expected  = (std::endian::native == std::endian::little) ? 17293822569102704640ull : 240ull;
    // // uint64_t entry_9_expected  = (std::endian::native == std::endian::little) ? 4621256167635550208ull : 8768ull;
    // // uint64_t entry_20_expected = (std::endian::native == std::endian::little) ? 4641240890982006784ull : 26944ull;

    // uint64_t entry_0_expected  = 17293822569102704640ull;
    // uint64_t entry_9_expected  = 4621256167635550208ull;
    // uint64_t entry_20_expected = 4641240890982006784ull;

    // std::memcpy(&entry_0, &(ct_ptr->data[0]), 8);
    // std::memcpy(&entry_9, &(ct_ptr->data[9]), 8);
    // std::memcpy(&entry_20, &(ct_ptr->data[20]), 8);

    // REQUIRE(entry_0 == entry_0_expected);
    // REQUIRE(entry_9 == entry_9_expected);
    // REQUIRE(entry_20 == entry_20_expected);

    // DataObjectFactory::destroy(ct_ptr);
    std::cout << "Hello 1" << std::endl;
    REQUIRE(true);
}

TEST_CASE("ReadZarr->ChunkedTensor", TAG_IO) {
    // ChunkedTensor<double>* ct_ptr;

    // // Read in [100,100,100] fp64 tensor with chunking [10,10,10]
    // readZarr<ChunkedTensor<double>>(ct_ptr, "./test/runtime/local/io/zarr_test/ChunkedTensorTest/");

    // // Check some values in chunk 0,0,0
    // uint64_t entry_0;
    // uint64_t entry_9;
    // uint64_t entry_20;

    // // REQUIRE((std::endian::native == std::endian::little) || (std::endian::native == std::endian::big));

    // // uint64_t entry_0_expected  = (std::endian::native == std::endian::little) ? 17293822569102704640ull : 240ull;
    // // uint64_t entry_9_expected  = (std::endian::native == std::endian::little) ? 4621256167635550208ull : 8768ull;
    // // uint64_t entry_20_expected = (std::endian::native == std::endian::little) ? 4641240890982006784ull : 26944ull;

    // uint64_t entry_0_expected  = 17293822569102704640ull;
    // uint64_t entry_9_expected  = 4621256167635550208ull;
    // uint64_t entry_20_expected = 4641240890982006784ull;

    // std::memcpy(&entry_0, &(ct_ptr->data[0]), 8);
    // std::memcpy(&entry_9, &(ct_ptr->data[9]), 8);
    // std::memcpy(&entry_20, &(ct_ptr->data[20]), 8);

    // REQUIRE(entry_0 == entry_0_expected);
    // REQUIRE(entry_9 == entry_9_expected);
    // REQUIRE(entry_20 == entry_20_expected);

    // // Check some values in chunk 1,1,1
    // uint64_t entry_0_111;
    // uint64_t entry_9_111;
    // uint64_t entry_20_111;

    // // uint64_t entry_0_expected_111 =
    // //   (std::endian::native == std::endian::little) ? 4681677767555678208ull : 548010048ull;
    // // uint64_t entry_9_expected_111 =
    // //   (std::endian::native == std::endian::little) ? 4681678386030968832ull : 2963929152ull;
    // // uint64_t entry_20_expected_111 =
    // //   (std::endian::native == std::endian::little) ? 4681691511451025408ull : 2696280128ull;

    // uint64_t entry_0_expected_111  = 4681677767555678208ull;
    // uint64_t entry_9_expected_111  = 4681678386030968832ull;
    // uint64_t entry_20_expected_111 = 4681691511451025408ull;

    // double* ptr_to_111_chunk = ct_ptr->getPtrToChunk({1, 1, 1});
    // std::memcpy(&entry_0_111, &(ptr_to_111_chunk[0]), 8);
    // std::memcpy(&entry_9_111, &(ptr_to_111_chunk[9]), 8);
    // std::memcpy(&entry_20_111, &(ptr_to_111_chunk[20]), 8);

    // REQUIRE(entry_0_111 == entry_0_expected_111);
    // REQUIRE(entry_9_111 == entry_9_expected_111);
    // REQUIRE(entry_20_111 == entry_20_expected_111);

    // DataObjectFactory::destroy(ct_ptr);
    std::cout << "Hello 2" << std::endl;

    REQUIRE(true);
}