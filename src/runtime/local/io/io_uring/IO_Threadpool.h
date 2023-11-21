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

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "AsyncUtil.h"
#include "IO_URing.h"

void CompletionWrapper(std::atomic<bool> *shut_down_requested, URing *ring);
void SubmissionWrapper(std::atomic<bool> *shut_down_requested, URing *ring);

struct URingRunner;

struct IOThreadpool {
    std::vector<URingRunner *> runners;
    IOThreadpool(uint32_t amount_of_io_urings,
                 uint32_t ring_size,
                 bool use_io_dev_polling,
                 bool use_sq_polling,
                 uint32_t submission_queue_idle_timeout_in_ms);
    ~IOThreadpool();
    std::unique_ptr<std::atomic<IO_STATUS>[]> SubmitReads(const std::vector<URingRead> &reads);
    std::unique_ptr<std::atomic<IO_STATUS>[]> SubmitWrites(const std::vector<URingWrite> &writes);
    void SubmitReads(const std::vector<URingRead> &reads, std::vector<std::atomic<IO_STATUS> *> &results);
    void SubmitWrites(const std::vector<URingWrite> &writes, std::vector<std::atomic<IO_STATUS> *> &results);
};
