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
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <optional>
#include <vector>
#include <fstream>
#include <filesystem>
#include <bit>

#include <fcntl.h>
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/io/ZarrFileMetadata.h>
#include <runtime/local/io/io_uring/IO_Threadpool.h>
#include <runtime/local/io/io_uring/IO_URing.h>
#include <parser/metadata/ZarrFileMetaDataParser.h>

enum struct IO_TYPE { POSIX, IO_URING };

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<class DTRes>
struct ReadZarr {
    static void apply(DTRes *&res, const char *filename) = delete;
};

template<class DTRes>
struct PartialReadZarr {
    static void apply(DTRes *&res, const char *filename, const std::vector<std::pair<size_t, size_t>> &element_id_ranges) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes>
void readZarr(DTRes *&res, const char *filename) {
    ReadZarr<DTRes>::apply(res, filename);
}

template<class DTRes>
void readZarr(DTRes *&res, const char *filename, const std::vector<std::pair<size_t, size_t>> &element_id_ranges) {
    PartialReadZarr<DTRes>::apply(res, filename, element_id_ranges);
}

template<class DTRes>
void readZarr(DTRes *&res,
              const char *filename,
              const std::vector<std::pair<size_t, size_t>> &element_id_ranges,
              IOThreadpool *io_uring_pool) {
    PartialReadZarr<DTRes>::apply(res, filename, element_id_ranges, io_uring_pool);
}

template<class DTRes>
void partialReadZarr(DTRes *&res, const char *filename, uint32_t lowerX, uint32_t upperX, uint32_t lowerY, uint32_t upperY, uint32_t lowerZ, uint32_t upperZ, DCTX(ctx)) {
    PartialReadZarr<DTRes>::apply(res, filename, {{lowerX, upperX},{lowerY,upperY},{lowerZ,upperZ}});
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

        // Fetch all avaiable chunk keys within the respective directory
        std::string base_file_path                                         = filename;
        std::vector<std::pair<std::string, std::string>> chunk_keys_in_dir = GetAllChunkKeys(base_file_path);

        // Check retrieved keys for vailidity and generate full canonical path and ascociated chunk ids from it
        std::vector<std::vector<size_t>> chunk_ids;
        std::vector<std::string> full_chunk_file_paths;
        for (size_t i = 0; i < chunk_keys_in_dir.size(); i++) {
            auto tmp = GetChunkIdsFromChunkKey(
              std::get<1>(chunk_keys_in_dir[i]), dimension_separator, fmd.shape, chunks_per_dim);

            if (tmp) {
                full_chunk_file_paths.push_back(
                  std::filesystem::canonical(std::filesystem::path(std::get<0>(chunk_keys_in_dir[i]))));
                chunk_ids.push_back(tmp.value());
            }
        }

        if (full_chunk_file_paths.size() > 1) {
            throw std::runtime_error("ReadZarr->ContiguousTensor: Found more than one chunk");
        }

        if ((std::endian::native != std::endian::little) && (std::endian::native != std::endian::big)) {
            throw std::runtime_error(
              "ReadZarr->ContiguousTensor: Native endianness that is not either little or big endian is not "
              "supported.");
        }

        bool endianness_match =
          ((std::endian::native == std::endian::big) && (byte_order == ByteOrder::BIGENDIAN)) ||
          ((std::endian::native == std::endian::little) && (byte_order == ByteOrder::LITTLEENDIAN));

        // For all requested chunks open the respective file, fetch the ptr to the chunks location in the chunked tensor
        // and directly read into it
        for (size_t i = 0; i < full_chunk_file_paths.size(); i++) {
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

            // Files endianness does not match the native endianness -> byte reverse every read element in the read
            // chunk
            if (!endianness_match) {
                ReverseArray(res->data.get(), total_elements);
            }
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

        // Fetch all avaiable chunk keys within the respective directory
        std::string base_file_path                                         = filename;
        std::vector<std::pair<std::string, std::string>> chunk_keys_in_dir = GetAllChunkKeys(base_file_path);

        // Check retrieved keys for vailidity and generate full canonical path and ascociated chunk ids from it
        std::vector<std::vector<size_t>> chunk_ids;
        std::vector<std::string> full_chunk_file_paths;
        for (size_t i = 0; i < chunk_keys_in_dir.size(); i++) {
            auto tmp = GetChunkIdsFromChunkKey(
              std::get<1>(chunk_keys_in_dir[i]), dimension_separator, fmd.shape, chunks_per_dim);

            if (tmp) {
                full_chunk_file_paths.push_back(
                  std::filesystem::canonical(std::filesystem::path(std::get<0>(chunk_keys_in_dir[i]))));
                chunk_ids.push_back(tmp.value());
            }
        }

        if ((std::endian::native != std::endian::little) && (std::endian::native != std::endian::big)) {
            throw std::runtime_error(
              "ReadZarr->ChunkedTensor: Native endianness that is not either little or big endian is not "
              "supported.");
        }

        bool endianness_match =
          ((std::endian::native == std::endian::big) && (byte_order == ByteOrder::BIGENDIAN)) ||
          ((std::endian::native == std::endian::little) && (byte_order == ByteOrder::LITTLEENDIAN));

        // For all requested chunks open the respective file, fetch the ptr to the chunks location in the chunked
        // tensor and directly read into it
        for (size_t i = 0; i < full_chunk_file_paths.size(); i++) {
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

            // Files endianness does not match the native endianness -> byte reverse every read element in the read
            // chunk
            if (!endianness_match) {
                ReverseArray(res->data.get(), elements_per_chunk);
            }

            res->chunk_materialization_flags[res->getLinearChunkIdFromChunkIds(chunk_ids[i])] = true;
        }
    }
};

// As in the tensor classes themselves the ranges are inclusive on both sides

template<typename VT>
struct PartialReadZarr<ContiguousTensor<VT>> {
    static void apply(ContiguousTensor<VT> *&res,
                      const char *filename,
                      const std::vector<std::pair<size_t, size_t>> &element_id_ranges) {}
};

template<typename VT>
struct PartialReadZarr<ChunkedTensor<VT>> {
    static void apply(ChunkedTensor<VT> *&res,
                      const char *filename,
                      const std::vector<std::pair<size_t, size_t>> &element_id_ranges) {
        auto fmd = ZarrFileMetaDataParser::readMetaData(filename);

        if (fmd.shape.size() == 0) {
            throw std::runtime_error(
              "PartialReadZarr->ChunkedTensor: Tensors of dim 0 i.e. scalars are currently not supported during "
              "reading");
        }
        if (fmd.shape.size() != fmd.chunks.size()) {
            throw std::runtime_error(
              "PartialReadZarr->ChunkedTensor: Dimension of tensor shape and chunk shape are missmatched");
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

        // Fetch all avaiable chunk keys within the respective directory
        std::string base_file_path                                         = filename;
        std::vector<std::pair<std::string, std::string>> chunk_keys_in_dir = GetAllChunkKeys(base_file_path);

        // Check retrieved keys for vailidity and generate full canonical path and ascociated chunk ids from it
        std::vector<std::vector<size_t>> chunk_ids;
        std::vector<std::string> full_chunk_file_paths;
        for (size_t i = 0; i < chunk_keys_in_dir.size(); i++) {
            auto tmp = GetChunkIdsFromChunkKey(
              std::get<1>(chunk_keys_in_dir[i]), dimension_separator, fmd.shape, chunks_per_dim);

            if (tmp) {
                full_chunk_file_paths.push_back(
                  std::filesystem::canonical(std::filesystem::path(std::get<0>(chunk_keys_in_dir[i]))));
                chunk_ids.push_back(tmp.value());
            }
        }

        // Convert element ranges into list of chunks required
        std::optional<std::vector<std::vector<size_t>>> requested_chunk_ids =
          res->GetChunkListFromIdRange(element_id_ranges);

        if (!requested_chunk_ids) {
            throw std::runtime_error(
              "PartialReadZarr->ChunkedTensor: Invalid element range. Range out of bounds or has missmatching "
              "dimension");
        }

        // Match requested chunks to the available chunks in the fs, discard not-requested files and throw on missing
        // file
        std::vector<std::string> full_requested_chunk_file_paths;
        full_requested_chunk_file_paths.reserve(requested_chunk_ids.value().size());
        for (size_t i = 0; i < requested_chunk_ids.value().size(); i++) {
            bool found_file_match_for_requested_chunk = false;
            for (size_t j = 0; j < chunk_ids.size(); j++) {
                if (requested_chunk_ids.value()[i] == chunk_ids[j]) {
                    found_file_match_for_requested_chunk = true;
                    full_requested_chunk_file_paths.push_back(full_chunk_file_paths[j]);

                    full_chunk_file_paths.erase(full_chunk_file_paths.begin() + static_cast<int64_t>(j));
                    chunk_ids.erase(chunk_ids.begin() + static_cast<int64_t>(j));
                    break;
                }
            }
            if (!found_file_match_for_requested_chunk) {
                throw std::runtime_error("PartialReadZarr->ChunkedTensor: Did not find all requested chunk files");
            }
        }

        if ((std::endian::native != std::endian::little) && (std::endian::native != std::endian::big)) {
            throw std::runtime_error(
              "PartialReadZarr->ChunkedTensor: Native endianness that is not either little or big endian is not "
              "supported.");
        }

        bool endianness_match =
          ((std::endian::native == std::endian::big) && (byte_order == ByteOrder::BIGENDIAN)) ||
          ((std::endian::native == std::endian::little) && (byte_order == ByteOrder::LITTLEENDIAN));

        // For all requested chunks open the respective file, fetch the ptr to the chunks location in the chunked
        // tensor and directly read into it
        for (size_t i = 0; i < full_requested_chunk_file_paths.size(); i++) {
            // IO via STL -> Posix ; substitude io_uring calls here
            std::ifstream f;
            f.open(full_requested_chunk_file_paths[i], std::ios::in | std::ios::binary);

            if (!f.good()) {
                throw std::runtime_error("PartialReadZarr->ChunkedTensor: failed to open chunk file.");
            }

            uint64_t amount_of_bytes_to_read = sizeof(VT) * elements_per_chunk;
            f.read(reinterpret_cast<char *>(res->getPtrToChunk(requested_chunk_ids.value()[i])),
                   amount_of_bytes_to_read);

            if (!f.good()) {
                throw std::runtime_error("PartialReadZarr->ChunkedTensor: failed to read chunk file.");
            }

            // Files endianness does not match the native endianness -> byte reverse every read element in the read
            // chunk
            if (!endianness_match) {
                ReverseArray(res->data.get(), elements_per_chunk);
            }

            res->chunk_materialization_flags[res->getLinearChunkIdFromChunkIds(requested_chunk_ids.value()[i])] = true;
        }
    }

    // Similar to the methods above, but instead of blocking untill IO is completed it returns upon submission and the
    // status of a given chunk can be then checked in the chunk_io_futures memeber of res
    // A kernel may make use of this and check its range of chunks it is supposed to process and for chunks that have
    // arrived and process them untill all chunks have been processed.
    static void apply(ChunkedTensor<VT> *&res,
                      const char *filename,
                      const std::vector<std::pair<size_t, size_t>> &element_id_ranges,
                      IOThreadpool *io_uring_pool) {
        auto fmd = ZarrFileMetaDataParser::readMetaData(filename);

        if (fmd.shape.size() == 0) {
            throw std::runtime_error(
              "PartialReadZarr->ChunkedTensor: Tensors of dim 0 i.e. scalars are currently not supported during "
              "reading");
        }
        if (fmd.shape.size() != fmd.chunks.size()) {
            throw std::runtime_error(
              "PartialReadZarr->ChunkedTensor: Dimension of tensor shape and chunk shape are missmatched");
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

        // Fetch all avaiable chunk keys within the respective directory
        std::string base_file_path                                         = filename;
        std::vector<std::pair<std::string, std::string>> chunk_keys_in_dir = GetAllChunkKeys(base_file_path);

        // Check retrieved keys for vailidity and generate full canonical path and ascociated chunk ids from it
        std::vector<std::vector<size_t>> chunk_ids;
        std::vector<std::string> full_chunk_file_paths;
        for (size_t i = 0; i < chunk_keys_in_dir.size(); i++) {
            auto tmp = GetChunkIdsFromChunkKey(
              std::get<1>(chunk_keys_in_dir[i]), dimension_separator, fmd.shape, chunks_per_dim);

            if (tmp) {
                full_chunk_file_paths.push_back(
                  std::filesystem::canonical(std::filesystem::path(std::get<0>(chunk_keys_in_dir[i]))));
                chunk_ids.push_back(tmp.value());
            }
        }

        // Convert element ranges into list of chunks required
        std::optional<std::vector<std::vector<size_t>>> requested_chunk_ids =
          res->GetChunkListFromIdRange(element_id_ranges);

        if (!requested_chunk_ids) {
            throw std::runtime_error(
              "PartialReadZarr->ChunkedTensor: Invalid element range. Range out of bounds or has missmatching "
              "dimension");
        }

        // Match requested chunks to the available chunks in the fs, discard not-requested files and throw on missing
        // file
        std::vector<std::string> full_requested_chunk_file_paths;
        full_requested_chunk_file_paths.reserve(requested_chunk_ids.value().size());
        for (size_t i = 0; i < requested_chunk_ids.value().size(); i++) {
            bool found_file_match_for_requested_chunk = false;
            for (size_t j = 0; j < chunk_ids.size(); j++) {
                if (requested_chunk_ids.value()[i] == chunk_ids[j]) {
                    found_file_match_for_requested_chunk = true;
                    full_requested_chunk_file_paths.push_back(full_chunk_file_paths[j]);

                    full_chunk_file_paths.erase(full_chunk_file_paths.begin() + static_cast<int64_t>(j));
                    chunk_ids.erase(chunk_ids.begin() + static_cast<int64_t>(j));
                    break;
                }
            }
            if (!found_file_match_for_requested_chunk) {
                throw std::runtime_error("PartialReadZarr->ChunkedTensor: Did not find all requested chunk files");
            }
        }

        if ((std::endian::native != std::endian::little) && (std::endian::native != std::endian::big)) {
            throw std::runtime_error(
              "PartialReadZarr->ChunkedTensor: Native endianness that is not either little or big endian is not "
              "supported.");
        }

        bool endianness_match =
          ((std::endian::native == std::endian::big) && (byte_order == ByteOrder::BIGENDIAN)) ||
          ((std::endian::native == std::endian::little) && (byte_order == ByteOrder::LITTLEENDIAN));

        std::vector<URingRead> read_requests;
        std::vector<std::atomic<IO_STATUS> *> io_futures;
        read_requests.resize(full_requested_chunk_file_paths.size());
        io_futures.resize(full_requested_chunk_file_paths.size());
        for (size_t i = 0; i < full_requested_chunk_file_paths.size(); i++) {
            uint64_t amount_of_bytes_to_read = sizeof(VT) * elements_per_chunk;

            int status;
            if (amount_of_bytes_to_read % 4096 == 0) {
                status = open(full_requested_chunk_file_paths[i].c_str(), O_RDONLY | O_DIRECT);
            } else {
                status = open(full_requested_chunk_file_paths[i].c_str(), O_RDONLY);
            }

            if (status < 0) {
                int err = errno;
                throw std::runtime_error("PartialReadZarr->ChunkedTensor: Opening chunk file failed. Errno: " +
                                         std::string(strerror(err)));
            }

            read_requests[i].dest   = res->getPtrToChunk(requested_chunk_ids.value()[i]);
            read_requests[i].size   = amount_of_bytes_to_read;
            read_requests[i].offset = 0;
            read_requests[i].fd     = status;

            AsyncIOInfo *chunk_async_io_info         = res->GetAsyncIOInfo(requested_chunk_ids.value()[i]);
            chunk_async_io_info->needs_byte_reversal = !endianness_match;
            io_futures[i]                            = &(chunk_async_io_info->status);
        }

        io_uring_pool->SubmitReads(read_requests, io_futures);
    }
};

#endif    // ZARR_IO_H