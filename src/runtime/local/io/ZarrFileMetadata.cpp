#include <runtime/local/io/ZarrFileMetadata.h>
#include <iostream>

std::ostream & operator<<(std::ostream& out, const ByteOrder& bo) {
    switch (bo) {
        case ByteOrder::LITTLEENDIAN: out << "\"little endian\""; break;
        case ByteOrder::BIGENDIAN: out << "\"big endian\""; break;
        case ByteOrder::NOT_RELEVANT: out << "\"not relevant\""; break;
    }
    return out;
}

std::ostream & operator<<(std::ostream& out, const ZarrDatatype& dt) {
    switch (dt) {
        case ZarrDatatype::BOOLEAN: out << "\"boolean\""; break;
        case ZarrDatatype::FLOATING: out << "\"floating\""; break;
        case ZarrDatatype::INTEGER: out << "\"integer\""; break;
        case ZarrDatatype::UINTEGER: out << "\"unsigned integer\""; break;
        case ZarrDatatype::COMPLEX_FLOATING: out << "\"complex floating\""; break;
        case ZarrDatatype::TIMEDELTA: out << "\"timedelta\""; break;
        case ZarrDatatype::DATETIME: out << "\"datetime\""; break;
        case ZarrDatatype::STRING: out << "\"string\""; break;
        case ZarrDatatype::UNICODE: out << "\"unicode\""; break;
        case ZarrDatatype::OTHER: out << "\"other\""; break;
    }
    return out;
}

std::ostream & operator<<(std::ostream& out, ZarrFileMetaData& zm) {
    out << "Chunks [";
    for (const auto& e : zm.chunks) {
        out << e << " ";
    }
    out << "]\n";
    out << "Shape [";
    for (const auto& e : zm.shape) {
        out << e << " ";
    }
    out << "]\n";
    out << "Zarr format: " << zm.zarr_format << "\n";
    out << "Order: " << zm.order << "\n";
    out << "Fill value: \"" << zm.fill_value << "\"\n";
    out << "Data type: " << zm.data_type << "\n";
    out << "Byte order: " << zm.byte_order << "\n";
    out << "#Bytes data type: " << zm.nBytes << "\n";
    out << "Dimension separator: " << zm.dimension_separator << "\n";
    return out;
}