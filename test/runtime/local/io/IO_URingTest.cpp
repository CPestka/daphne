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

#include <vector>
#include <chrono>
#include <iostream>

#include <fcntl.h>

#include <tags.h>

#include <catch.hpp>

#include <IO_Threadpool.h>
#include <IO_URing.h>

TEST_CASE("io_uring / io threadpool setup Tests", TAG_IO) {
    URing(64, false, false, 1000);

    { IOThreadpool(1, 64, false, false, 1000); }
    { IOThreadpool(4, 64, false, false, 1000); }
    { IOThreadpool(2, 64, false, false, 1000); }
    { IOThreadpool(8, 64, false, false, 1000); }
}

TEST_CASE("io_uring basic File R/W test", TAG_IO) {
    std::cout << "Hello" << std::endl;
    IOThreadpool io_pool(1, 64, false, false, 1000);

    int fd = open("./uring_test_tmp", O_CREAT | O_RDWR | O_DIRECT | O_TRUNC, S_IRUSR | S_IWUSR);
    REQUIRE(fd > 0);

    auto data_page   = std::make_unique<uint64_t[]>(4096 / sizeof(uint64_t));
    auto result_page = std::make_unique<uint64_t[]>(4096 / sizeof(uint64_t));
    for (size_t i = 0; i < (4096 / sizeof(uint64_t)); i++) {
        data_page[i]   = 42;
        result_page[i] = 0;
    }

    // The write
    std::vector<URingWrite> write_requests;
    write_requests.push_back({data_page.get(), 4096, 0, fd});

    std::unique_ptr<std::atomic<IO_STATUS>[]> write_future = io_pool.SubmitWrites(write_requests);

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    while (write_future[0] == IO_STATUS::IN_FLIGHT) {
        REQUIRE(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::high_resolution_clock::now() - start)
                                        .count()) < 20000);
    }

    REQUIRE(write_future[0] == IO_STATUS::SUCCESS);

    // The Read
    std::vector<URingRead> read_requests;
    write_requests.push_back({result_page.get(), 4096, 0, fd});

    std::unique_ptr<URingReadResult[]> read_future = io_pool.SubmitReads(read_requests);

    start = std::chrono::high_resolution_clock::now();

    while (read_future[0].status == IO_STATUS::IN_FLIGHT) {
        REQUIRE(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::high_resolution_clock::now() - start)
                                        .count()) < 20000);
    }

    REQUIRE(read_future[0].status == IO_STATUS::SUCCESS);

    for (size_t i = 0; i < (4096 / sizeof(uint64_t)); i++) {
        REQUIRE(result_page[i] == 42);
    }
}