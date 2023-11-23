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
#include <runtime/local/io/ZarrFileMetadata.h>
#include "runtime/local/io/io_uring/IO_Threadpool.h"

#include <chrono>

#include <tags.h>

#include <catch.hpp>

TEST_CASE("ReadZarr->ContiguousTensor", TAG_IO) {
    ContiguousTensor<double>* ct_ptr;

    // Read in [10,10,10] fp64 tensor
    readZarr<ContiguousTensor<double>>(ct_ptr, "./test/runtime/local/io/zarr_test/ContiguousTensorTest/example.zarr");

    REQUIRE(ct_ptr->data[0] == 0.0);
    REQUIRE(ct_ptr->data[1] == 1.0);
    REQUIRE(ct_ptr->data[7] == 7.0);
    REQUIRE(ct_ptr->data[10] == 100.0);
    REQUIRE(ct_ptr->data[12] == 102.0);

    DataObjectFactory::destroy(ct_ptr);
}

TEST_CASE("ReadZarr->ChunkedTensor", TAG_IO) {
    ChunkedTensor<double>* ct_ptr;

    // Read in [100,100,100] fp64 tensor with chunking [10,10,10]
    readZarr<ChunkedTensor<double>>(ct_ptr, "./test/runtime/local/io/zarr_test/ChunkedTensorTest/example.zarr");

    REQUIRE(ct_ptr->data[0] == Approx(0.0));
    REQUIRE(ct_ptr->data[1] == Approx(1.0));
    REQUIRE(ct_ptr->data[7] == Approx(7.0));
    REQUIRE(ct_ptr->data[10] == Approx(100.0));
    REQUIRE(ct_ptr->data[12] == Approx(102.0));

    double* ptr_to_111_chunk = ct_ptr->getPtrToChunk({1, 1, 1});

    REQUIRE(ptr_to_111_chunk[0] == Approx(101010.0));
    REQUIRE(ptr_to_111_chunk[1] == Approx(101011.0));
    REQUIRE(ptr_to_111_chunk[7] == Approx(101017.0));
    REQUIRE(ptr_to_111_chunk[10] == Approx(101110.0));
    REQUIRE(ptr_to_111_chunk[12] == Approx(101112.0));

    DataObjectFactory::destroy(ct_ptr);
}

TEST_CASE("PartialReadZarr->ChunkedTensor", TAG_IO) {
    ChunkedTensor<double>* ct_ptr;

    // Read in [100,100,100] fp64 tensor with chunking [10,10,10]
    readZarr<ChunkedTensor<double>>(
      ct_ptr, "./test/runtime/local/io/zarr_test/ChunkedTensorTest/example.zarr", {{0, 20}, {0, 20}, {0, 20}});

    REQUIRE(ct_ptr->data[0] == Approx(0.0));
    REQUIRE(ct_ptr->data[1] == Approx(1.0));
    REQUIRE(ct_ptr->data[7] == Approx(7.0));
    REQUIRE(ct_ptr->data[10] == Approx(100.0));
    REQUIRE(ct_ptr->data[12] == Approx(102.0));

    double* ptr_to_111_chunk = ct_ptr->getPtrToChunk({1, 1, 1});

    REQUIRE(ptr_to_111_chunk[0] == Approx(101010.0));
    REQUIRE(ptr_to_111_chunk[1] == Approx(101011.0));
    REQUIRE(ptr_to_111_chunk[7] == Approx(101017.0));
    REQUIRE(ptr_to_111_chunk[10] == Approx(101110.0));
    REQUIRE(ptr_to_111_chunk[12] == Approx(101112.0));

    DataObjectFactory::destroy(ct_ptr);
}

TEST_CASE("Async-PartialReadZarr->ChunkedTensor", TAG_IO) {
    IOThreadpool io_uring_pool(1, 4096, false, false, 1000);
    ChunkedTensor<double>* ct_ptr;

    // Read in [100,100,100] fp64 tensor with chunking [10,10,10]
    readZarr<ChunkedTensor<double>>(ct_ptr,
                                    "./test/runtime/local/io/zarr_test/ChunkedTensorTest/example.zarr",
                                    {{0, 20}, {0, 20}, {0, 20}},
                                    &io_uring_pool);

    std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();

    bool all_data_arrived = false;
    bool data_1_arrived = false;
    bool data_2_arrived = false;
    while (
      static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count()) <
      2000) {
        if (!data_1_arrived) {
            data_1_arrived = ct_ptr->PollChunkMaterializationAndIOStatus(0);
        }
        if (!data_2_arrived) {
            data_2_arrived = ct_ptr->PollChunkMaterializationAndIOStatus({1,1,1});
        }


        if (data_1_arrived && data_2_arrived) {
            all_data_arrived = true;
            break;
        }
    }

    REQUIRE(all_data_arrived);

    REQUIRE(ct_ptr->data[0] == Approx(0.0));
    REQUIRE(ct_ptr->data[1] == Approx(1.0));
    REQUIRE(ct_ptr->data[7] == Approx(7.0));
    REQUIRE(ct_ptr->data[10] == Approx(100.0));
    REQUIRE(ct_ptr->data[12] == Approx(102.0));

    double* ptr_to_111_chunk = ct_ptr->getPtrToChunk({1, 1, 1});

    REQUIRE(ptr_to_111_chunk[0] == Approx(101010.0));
    REQUIRE(ptr_to_111_chunk[1] == Approx(101011.0));
    REQUIRE(ptr_to_111_chunk[7] == Approx(101017.0));
    REQUIRE(ptr_to_111_chunk[10] == Approx(101110.0));
    REQUIRE(ptr_to_111_chunk[12] == Approx(101112.0));

    DataObjectFactory::destroy(ct_ptr);
}

TEST_CASE("Async-PartialReadZarr->ChunkedTensor+SQPoll", TAG_IO) {
    IOThreadpool io_uring_pool(1, 4096, false, true, 1000);
    ChunkedTensor<double>* ct_ptr;

    // Read in [100,100,100] fp64 tensor with chunking [10,10,10]
    readZarr<ChunkedTensor<double>>(ct_ptr,
                                    "./test/runtime/local/io/zarr_test/ChunkedTensorTest/example.zarr",
                                    {{0, 20}, {0, 20}, {0, 20}},
                                    &io_uring_pool);

    std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();

    bool all_data_arrived = false;
    bool data_1_arrived = false;
    bool data_2_arrived = false;
    while (
      static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count()) <
      2000) {
        if (!data_1_arrived) {
            data_1_arrived = ct_ptr->PollChunkMaterializationAndIOStatus(0);
        }
        if (!data_2_arrived) {
            data_2_arrived = ct_ptr->PollChunkMaterializationAndIOStatus({1,1,1});
        }


        if (data_1_arrived && data_2_arrived) {
            all_data_arrived = true;
            break;
        }
    }

    REQUIRE(all_data_arrived);

    REQUIRE(ct_ptr->data[0] == Approx(0.0));
    REQUIRE(ct_ptr->data[1] == Approx(1.0));
    REQUIRE(ct_ptr->data[7] == Approx(7.0));
    REQUIRE(ct_ptr->data[10] == Approx(100.0));
    REQUIRE(ct_ptr->data[12] == Approx(102.0));

    double* ptr_to_111_chunk = ct_ptr->getPtrToChunk({1, 1, 1});

    REQUIRE(ptr_to_111_chunk[0] == Approx(101010.0));
    REQUIRE(ptr_to_111_chunk[1] == Approx(101011.0));
    REQUIRE(ptr_to_111_chunk[7] == Approx(101017.0));
    REQUIRE(ptr_to_111_chunk[10] == Approx(101110.0));
    REQUIRE(ptr_to_111_chunk[12] == Approx(101112.0));

    DataObjectFactory::destroy(ct_ptr);
}

TEST_CASE("Async-PartialReadZarr->ChunkedTensor+SQPoll+MultipleRings", TAG_IO) {
    IOThreadpool io_uring_pool(4, 4096, false, true, 1000);
    std::cout << "1" << std::endl;
    ChunkedTensor<double>* ct_ptr;

    // Read in [100,100,100] fp64 tensor with chunking [10,10,10]
    readZarr<ChunkedTensor<double>>(ct_ptr,
                                    "./test/runtime/local/io/zarr_test/ChunkedTensorTest/example.zarr",
                                    {{0, 20}, {0, 20}, {0, 20}},
                                    &io_uring_pool);

    std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();
    std::cout << "2" << std::endl;

    bool all_data_arrived = false;
    bool data_1_arrived = false;
    bool data_2_arrived = false;
    while (
      static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t1).count()) <
      2000) {
        if (!data_1_arrived) {
            data_1_arrived = ct_ptr->PollChunkMaterializationAndIOStatus(0);
        }
        if (!data_2_arrived) {
            data_2_arrived = ct_ptr->PollChunkMaterializationAndIOStatus({1,1,1});
        }


        if (data_1_arrived && data_2_arrived) {
            all_data_arrived = true;
            break;
        }
    }
    std::cout << "3" << std::endl;

    REQUIRE(all_data_arrived);

    REQUIRE(ct_ptr->data[0] == Approx(0.0));
    REQUIRE(ct_ptr->data[1] == Approx(1.0));
    REQUIRE(ct_ptr->data[7] == Approx(7.0));
    REQUIRE(ct_ptr->data[10] == Approx(100.0));
    REQUIRE(ct_ptr->data[12] == Approx(102.0));

    double* ptr_to_111_chunk = ct_ptr->getPtrToChunk({1, 1, 1});

    REQUIRE(ptr_to_111_chunk[0] == Approx(101010.0));
    REQUIRE(ptr_to_111_chunk[1] == Approx(101011.0));
    REQUIRE(ptr_to_111_chunk[7] == Approx(101017.0));
    REQUIRE(ptr_to_111_chunk[10] == Approx(101110.0));
    REQUIRE(ptr_to_111_chunk[12] == Approx(101112.0));

    DataObjectFactory::destroy(ct_ptr);
}