// Copyright 2021-2022, MICROSOFT CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of MICROSOFT CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include "adsbrain_server.h"


namespace triton { namespace server {

TRITONSERVER_Error*
AdsBrainAPIServer::Create(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager, const int32_t port,
    const std::string address, const int thread_cnt,
    std::unique_ptr<HTTPServer>* http_server)
{
  http_server->reset(new AdsBrainAPIServer(
      server, trace_manager, shm_manager, port, address, thread_cnt));

  const std::string addr = address + ":" + std::to_string(port);
  LOG_INFO << "Started AdsBrain HTTPService at " << addr;

  return nullptr;
}

TRITONSERVER_Error*
AdsBrainAPIServer::AdsBainInferRequestClass::FinalizeResponse(
    TRITONSERVER_InferenceResponse* response)
{
  switch(request_type_) {
        case RequestType::TRITON:
          return FinalizeResponseInTritonFormat(response);
        case RequestType::ADSBRAIN_BOND:
          return FinalizeResponseInBondFormat(response);
        default:
          return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            "Does not support this kind of request format");
  }
}

TRITONSERVER_Error*
AdsBrainAPIServer::AdsBainInferRequestClass::FinalizeResponseInBondFormat(
    TRITONSERVER_InferenceResponse* response)
{
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseError(response));

  // Go through each response output and transfer information to JSON
  uint32_t output_count;
  RETURN_IF_ERR(
      TRITONSERVER_InferenceResponseOutputCount(response, &output_count));

  evbuffer* response_placeholder = evbuffer_new();
  size_t total_byte_size = 0;

  for (uint32_t idx = 0; idx < output_count; ++idx) {
    const char* cname;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint64_t dim_count;
    const void* base;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    void* userp;

    RETURN_IF_ERR(TRITONSERVER_InferenceResponseOutput(
        response, idx, &cname, &datatype, &shape, &dim_count, &base, &byte_size,
        &memory_type, &memory_type_id, &userp));

    const char* cbase = reinterpret_cast<const char*>(base);
    size_t element_count = 1;
    for (size_t j = 0; j < dim_count; ++j) {
        element_count *= shape[j];
    }

    // The current implementation may have efficiency issue when element_count
    // is large. So the model inference code should organize the output tensor
    // with a small number of elements.
    size_t offset = 0;
    for (size_t i = 0; i < element_count; ++i) {
      // Each element is in the format of a 4-byte length followed by the data
      const size_t len = *(reinterpret_cast<const uint32_t*>(cbase + offset));
      offset += sizeof(uint32_t);
      // TODO: Add the delimiters for each element and each output? Or let users
      // define how to combine them in the model inference code?
      evbuffer_add(response_placeholder, cbase + offset, len);
      offset += len;
      total_byte_size += len;
    }
  }

  evbuffer* response_body = response_placeholder;
  switch (response_compression_type_) {
    case DataCompressor::Type::DEFLATE:
    case DataCompressor::Type::GZIP: {
      auto compressed_buffer = evbuffer_new();
      auto err = DataCompressor::CompressData(
          response_compression_type_, response_placeholder, compressed_buffer);
      if (err == nullptr) {
        response_body = compressed_buffer;
        evbuffer_free(response_placeholder);
      } else {
        // just log the compression error and return the uncompressed data
        LOG_VERBOSE(1) << "unable to compress response: "
                       << TRITONSERVER_ErrorMessage(err);
        TRITONSERVER_ErrorDelete(err);
        evbuffer_free(compressed_buffer);
        response_compression_type_ = DataCompressor::Type::IDENTITY;
      }
      break;
    }
    case DataCompressor::Type::IDENTITY:
    case DataCompressor::Type::UNKNOWN:
      // Do nothing for other cases
      break;
  }
  SetResponseHeader(total_byte_size>0, total_byte_size);
  evbuffer_add_buffer(req_->buffer_out, response_body);
  // Destroy the evbuffer object as the data has been moved
  // to HTTP response buffer
  evbuffer_free(response_body);

  return nullptr;  // success
}

}}  // namespace triton::server