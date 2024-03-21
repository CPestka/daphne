
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <bit>
#include <memory>

#include <spdlog/spdlog.h>

#include <fcntl.h>
#include <runtime/local/datastructures/ContiguousTensor.h>
#include <runtime/local/datastructures/ChunkedTensor.h>
#include <runtime/local/io/ZarrFileMetadata.h>
#include <runtime/local/io/io_uring/IO_Threadpool.h>
#include <runtime/local/io/io_uring/IO_URing.h>
#include <parser/metadata/ZarrFileMetaDataParser.h>

bool checkEndiannessMatch(const ByteOrder bo, std::shared_ptr<spdlog::logger> log) {
    if ((std::endian::native != std::endian::little) && (std::endian::native != std::endian::big)) {
        log->error("Native endianness that is neither little nor big endian is not supported");
        throw std::runtime_error("Native endianness that is neither little nor big endian is not supported.");
    }
    return ((std::endian::native == std::endian::big) && (bo == ByteOrder::BIGENDIAN)) ||
          ((std::endian::native == std::endian::little) && (bo == ByteOrder::LITTLEENDIAN));

}

std::vector<size_t> computeChunksPerDim(const std::vector<size_t>& chunks, const std::vector<size_t>& shape) {
    std::vector<size_t> chunks_per_dim;
    chunks_per_dim.resize(shape.size());
    for (size_t i = 0; i < chunks_per_dim.size(); i++) {
        chunks_per_dim[i] = shape[i] / chunks[i];
    }
    return chunks_per_dim;
}

uint64_t computeElementsPerChunk(const std::vector<size_t>& chunks, const size_t n) {
    uint64_t elements_per_chunk = chunks[0];
    for (size_t i = 1; i < n; i++) {
        elements_per_chunk *= chunks[i];
    }
    return elements_per_chunk;
}

std::vector<std::string> computeFullFilePathsForRequestedChunks(const std::vector<std::vector<size_t>>& requested_chunk_ids, std::vector<std::string>& full_chunk_file_paths, std::vector<std::vector<size_t>>& chunk_ids) {
    std::vector<std::string> full_requested_chunk_file_paths;
    full_requested_chunk_file_paths.reserve(requested_chunk_ids.size());

    for (size_t i = 0; i < requested_chunk_ids.size(); i++) {
        bool found_file_match_for_requested_chunk = false;
        for (size_t j = 0; j < chunk_ids.size(); j++) {
            if (requested_chunk_ids[i] == chunk_ids[j]) {
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
    return full_requested_chunk_file_paths;
}