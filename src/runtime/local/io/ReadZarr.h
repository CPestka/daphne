/*
 * Copyright 2023 The DAPHNE Consortium
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

#ifndef ZARR_IO_H
#define ZARR_IO_H

#include <iostream>
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/io/ZarrFileMetadata.h>
#include <parser/metadata/ZarrFileMetaDataParser.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template <class DTRes> struct ReadZarr {
    static void apply(DTRes *&res, const char *filename) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template <class DTRes>
void readZarr(DTRes *&res, const char *filename) {
    ReadZarr<DTRes>::apply(res, filename);
}

template <typename VT>
struct ReadZarr<ContiguousTensor<VT>> {
    static void apply(ContiguousTensor<VT> *&res, const char *filename) {
        auto fmd = ZarrFileMetaDataParser::readMetaData(filename);
        res = DataObjectFactory::create<ContiguousTensor<VT>>(fmd.shape, InitCode::NONE);

        // now do the actual reading part
        auto dimension_separator = fmd.dimension_separator;

        auto byte_order = fmd.byte_order;
    }
};

#endif // ZARR_IO_H