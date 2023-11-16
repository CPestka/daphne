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

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <optional>
#include <filesystem>
#include <vector>
#include <fstream>

#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/io/ZarrFileMetadata.h>
#include <parser/metadata/ZarrFileMetaDataParser.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<class DTRes>
struct ReadZarr {
    static void apply(DTRes *&res, const char *filename) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes>
void readZarr(DTRes *&res, const char *filename) {
    ReadZarr<DTRes>::apply(res, filename);
}

template<typename VT>
void ReverseArray(VT *data, uint64_t element_count) {
    for (uint64_t i = 0; i < element_count; i++) {
        VT tmp = data[i];
        for (uint32_t j = 0; j < sizeof(VT); j++) {
            *(reinterpret_cast<uint8_t *>(&(data[i])) + sizeof(VT) - j) = *(reinterpret_cast<uint8_t *>(&tmp) + j);
        }
    }
}

template<typename VT>
void CheckZarrMetaDataVT(ZarrDatatype read_type) {
    switch (read_type) {
        // using enum ZarrDatatype;
        case ZarrDatatype::BOOLEAN:
            if (!std::is_same<bool, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is bool != exptected VT");
            }
            break;
        case ZarrDatatype::FP64:
            if (!std::is_same<double, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is double != exptected VT");
            }
            break;
        case ZarrDatatype::FP32:
            if (!std::is_same<float, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is float != exptected VT");
            }
            break;
        case ZarrDatatype::UINT64:
            if (!std::is_same<uint64_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is uint64_t != exptected VT");
            }
            break;
        case ZarrDatatype::UINT32:
            if (!std::is_same<uint32_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is uint32_t != exptected VT");
            }
            break;
        case ZarrDatatype::UINT16:
            if (!std::is_same<uint16_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is uint16_t != exptected VT");
            }
            break;
        case ZarrDatatype::UINT8:
            if (!std::is_same<uint8_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is uint8_t != exptected VT");
            }
            break;
        case ZarrDatatype::INT64:
            if (!std::is_same<int64_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is int64_t != exptected VT");
            }
            break;
        case ZarrDatatype::INT32:
            if (!std::is_same<int32_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is int32_t != exptected VT");
            }
            break;
        case ZarrDatatype::INT16:
            if (!std::is_same<int16_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is int16_t != exptected VT");
            }
            break;
        case ZarrDatatype::INT8:
            if (!std::is_same<int8_t, VT>::value) {
                throw std::runtime_error("ReadZarr: read VT is int8_t != exptected VT");
            }
            break;
        default:
            throw std::runtime_error("ReadZarr: read VT currently not supported");
            break;
    }
}

template<typename VT>
struct ReadZarr<ContiguousTensor<VT>> {
    static void apply(ContiguousTensor<VT> *&res, const char *filename) {
        auto fmd = ZarrFileMetaDataParser::readMetaData(filename);

        if (fmd.chunks != fmd.shape) {
            throw std::runtime_error(
              "ReadZarr->ContiguousTensor: Missmatch between chunk and tensor shape. Consider using "
              "ReadZarr->ChunkedTensor instead.");
        }
        if (fmd.shape.size() == 0) {
            throw std::runtime_error(
              "ReadZarr->ContiguousTensor: Tensors of dim 0 i.e. scalars are currently not supported during reading");
        }

        CheckZarrMetaDataVT<VT>(fmd.data_type);

        res = DataObjectFactory::create<ContiguousTensor<VT>>(fmd.shape, InitCode::NONE);

        auto dimension_separator = fmd.dimension_separator;
        auto byte_order          = fmd.byte_order;
        std::vector<size_t> chunks_per_dim;
        chunks_per_dim.resize(fmd.shape.size());
        for (size_t i = 0; i < chunks_per_dim.size(); i++) {
            chunks_per_dim[i] = fmd.shape[i] / fmd.chunks[i];
        }
        uint64_t total_elements = fmd.shape[0];
        for (size_t i = 1; i < chunks_per_dim.size(); i++) {
            total_elements *= fmd.shape[i];
        }

        std::string base_file_path = filename;

        std::vector<std::pair<std::string, std::string>> chunk_keys_in_dir;    //= GetAllChunkKeys(base_file_path);

        std::vector<std::vector<size_t>> chunk_ids;
        std::vector<std::string> full_chunk_file_paths;

        for (size_t i = 0; i < chunk_keys_in_dir.size(); i++) {
            auto tmp = GetChunkIdsFromChunkKey(
              std::get<1>(chunk_keys_in_dir[i]), dimension_separator, fmd.shape, chunks_per_dim);

            if (tmp) {
                full_chunk_file_paths.push_back(std::get<0>(chunk_keys_in_dir[i]));
                chunk_ids.push_back(tmp.value());
            }
        }

        if (full_chunk_file_paths.size() != 1) {
            throw std::runtime_error("ReadZarr->ContiguousTensor: Found more than one chunk");
        }

        // No c++20....
        // if ((std::endian::native != std::endian::little) && (std::endian::native != std::endian::big)) {
        //     throw std::runtime_error(
        //       "ReadZarr->ContiguousTensor: Native endianness that is not either little or big endian is not "
        //       "supported.");
        // }

        // bool endianness_match =
        //   ((std::endian::native == std::endian::big) && (byte_order == ByteOrder::BIGENDIAN)) ||
        //   ((std::endian::native == std::endian::little) && (byte_order == ByteOrder::LITTLEENDIAN));

        for (size_t i = 0; i < chunk_keys_in_dir.size(); i++) {
            // IO via STL -> Posix ; substitue io_uring calls here
            std::ifstream f;
            f.open(full_chunk_file_paths[i], std::ios::in | std::ios::binary);

            if (!f.good()) {
                throw std::runtime_error("ReadZarr->ContiguousTensor: failed to open chunk file.");
            }

            uint64_t amount_of_bytes_to_read = sizeof(VT) * total_elements;
            f.read(reinterpret_cast<char *>(res->data.get()), amount_of_bytes_to_read);

            if (!f.good()) {
                throw std::runtime_error("ReadZarr->ContiguousTensor: failed to read chunk file.");
            }

            // if (!endianness_match) {
            //     ReverseArray(res->data.get(), total_elements);
            // }
        }
    }
};

template<typename VT>
struct ReadZarr<ChunkedTensor<VT>> {
    static void apply(ChunkedTensor<VT> *&res, const char *filename) {
        auto fmd = ZarrFileMetaDataParser::readMetaData(filename);

        if (fmd.shape.size() == 0) {
            throw std::runtime_error(
              "ReadZarr->ChunkedTensor: Tensors of dim 0 i.e. scalars are currently not supported during reading");
        }
        if (fmd.shape.size() != fmd.chunks.size()) {
            throw std::runtime_error(
              "ReadZarr->ChunkedTensor: Dimension of tensor shape and chunk shape are missmatched");
        }

        CheckZarrMetaDataVT<VT>(fmd.data_type);

        res = DataObjectFactory::create<ChunkedTensor<VT>>(fmd.shape, fmd.chunks, InitCode::NONE);

        auto dimension_separator = fmd.dimension_separator;
        auto byte_order          = fmd.byte_order;
        std::vector<size_t> chunks_per_dim;
        chunks_per_dim.resize(fmd.shape.size());
        for (size_t i = 0; i < chunks_per_dim.size(); i++) {
            chunks_per_dim[i] = fmd.shape[i] / fmd.chunks[i];
        }
        uint64_t elements_per_chunk = fmd.chunks[0];
        for (size_t i = 1; i < chunks_per_dim.size(); i++) {
            elements_per_chunk *= fmd.chunks[i];
        }

        std::string base_file_path = filename;

        std::vector<std::pair<std::string, std::string>> chunk_keys_in_dir = GetAllChunkKeys(base_file_path);

        std::vector<std::vector<size_t>> chunk_ids;
        std::vector<std::string> full_chunk_file_paths;

        for (size_t i = 0; i < chunk_keys_in_dir.size(); i++) {
            auto tmp = GetChunkIdsFromChunkKey(
              std::get<1>(chunk_keys_in_dir[i]), dimension_separator, fmd.shape, chunks_per_dim);

            if (tmp) {
                full_chunk_file_paths.push_back(std::get<0>(chunk_keys_in_dir[i]));
                chunk_ids.push_back(tmp.value());
            }
        }

        // if ((std::endian::native != std::endian::little) && (std::endian::native != std::endian::big)) {
        //     throw std::runtime_error(
        //       "ReadZarr->ChunkedTensor: Native endianness that is not either little or big endian is not "
        //       "supported.");
        // }

        // bool endianness_match =
        //   ((std::endian::native == std::endian::big) && (byte_order == ByteOrder::BIGENDIAN)) ||
        //   ((std::endian::native == std::endian::little) && (byte_order == ByteOrder::LITTLEENDIAN));

        for (size_t i = 0; i < chunk_keys_in_dir.size(); i++) {
            // IO via STL -> Posix ; substitude io_uring calls here
            std::ifstream f;
            f.open(full_chunk_file_paths[i], std::ios::in | std::ios::binary);

            if (!f.good()) {
                throw std::runtime_error("ReadZarr->ChunkedTensor: failed to open chunk file.");
            }

            uint64_t amount_of_bytes_to_read = sizeof(VT) * elements_per_chunk;
            f.read(reinterpret_cast<char *>(res->getPtrToChunk(chunk_ids[i])), amount_of_bytes_to_read);

            if (!f.good()) {
                throw std::runtime_error("ReadZarr->ChunkedTensor: failed to read chunk file.");
            }

            // if (!endianness_match) {
            //     ReverseArray(res->data.get(), elements_per_chunk);
            // }

            res->chunk_materialization_flags[res->getLinearChunkIdFromChunkIds(chunk_ids[i])] = true;
        }
    }
};

#endif    // ZARR_IO_H