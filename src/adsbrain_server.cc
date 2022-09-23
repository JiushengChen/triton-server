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
    const std::string entrypoint,
    DataCompressor::Type request_compressor,
    DataCompressor::Type response_compressor,
    std::unique_ptr<HTTPServer>* http_server)
{
  http_server->reset(new AdsBrainAPIServer(
      server, trace_manager, shm_manager, port, address, thread_cnt,
      entrypoint, request_compressor, response_compressor));

  const std::string addr = address + ":" + std::to_string(port);
  LOG_INFO << "Started AdsBrain HTTPService at " << addr
           << ", entrypoint: " << entrypoint
           << ", request_compressor: " << (int)request_compressor
           << ", response_compressor: " << (int)response_compressor;

  return nullptr;
}

void
AdsBrainAPIServer::Handle(evhtp_request_t* req)
{
  LOG_VERBOSE(1) << "AdsBrain HTTP request: " << req->method << " "
                 << req->uri->path->full;

  std::string uri = std::string(req->uri->path->full);
  if (uri == "" || uri == "/") {
    // set default uri
    LOG_VERBOSE(1) << "Empty uri! Set uri to " << entrypoint_;
    uri = entrypoint_;
  }

  if (uri == "/v2/models/stats") {
    // model statistics
    HandleModelStats(req);
    return;
  }

  std::string model_name, version, kind;
  if (RE2::FullMatch(
          uri, model_regex_, &model_name,
          &version, &kind)) {
    if (kind == "ready") {
      // model ready
      HandleModelReady(req, model_name, version);
      return;
    } else if (kind == "infer") {
      // model infer
      HandleInfer(req, model_name, version);
      return;
    } else if (kind == "config") {
      // model configuration
      HandleModelConfig(req, model_name, version);
      return;
    } else if (kind == "stats") {
      // model statistics
      HandleModelStats(req, model_name, version);
      return;
    } else if (kind == "trace/setting") {
      // Trace with specific model, there is no specification on versioning
      // so fall out and return bad request error if version is specified
      if (version.empty()) {
        HandleTrace(req, model_name);
        return;
      }
    } else if (kind == "") {
      // model metadata
      HandleModelMetadata(req, model_name, version);
      return;
    }
  }

  std::string region, action, rest, repo_name;
  if (uri == "/v2") {
    // server metadata
    HandleServerMetadata(req);
    return;
  } else if (RE2::FullMatch(
                 uri, HTTPAPIServer::server_regex_, &rest)) {
    // server health
    HandleServerHealth(req, rest);
    return;
  } else if (RE2::FullMatch(
                 uri, HTTPAPIServer::systemsharedmemory_regex_,
                 &region, &action)) {
    // system shared memory
    HandleSystemSharedMemory(req, region, action);
    return;
  } else if (RE2::FullMatch(
                 uri, HTTPAPIServer::cudasharedmemory_regex_,
                 &region, &action)) {
    // cuda shared memory
    HandleCudaSharedMemory(req, region, action);
    return;
  } else if (RE2::FullMatch(
                 uri, HTTPAPIServer::modelcontrol_regex_,
                 &repo_name, &kind, &model_name, &action)) {
    // model repository
    if (kind == "index") {
      HandleRepositoryIndex(req, repo_name);
      return;
    } else if (kind.find("models", 0) == 0) {
      HandleRepositoryControl(req, repo_name, model_name, action);
      return;
    }
  } else if (RE2::FullMatch(uri, HTTPAPIServer::trace_regex_)) {
    // trace request on global settings
    HandleTrace(req);
    return;
  }

  LOG_VERBOSE(1) << "HTTP error: " << req->method << " " << uri
                 << " - " << static_cast<int>(EVHTP_RES_BADREQ);

  evhtp_send_reply(req, EVHTP_RES_BADREQ);
}

TRITONSERVER_Error*
AdsBrainAPIServer::AdsBainInferRequestClass::FinalizeResponse(
    TRITONSERVER_InferenceResponse* response)
{
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseError(response));

  triton::common::TritonJson::Value response_json(
      triton::common::TritonJson::ValueType::OBJECT);

  // Go through each response output and transfer information to JSON
  uint32_t output_count;
  RETURN_IF_ERR(
      TRITONSERVER_InferenceResponseOutputCount(response, &output_count));

  std::vector<evbuffer*> ordered_buffers;
  ordered_buffers.reserve(output_count);

  triton::common::TritonJson::Value response_outputs(
      triton::common::TritonJson::ValueType::ARRAY);

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

    // Handle data. SHM outputs will not have an info.
    auto info = reinterpret_cast<AllocPayload::OutputInfo*>(userp);

    size_t element_count = 1;

    // Add JSON data, or collect binary data.
    if (info->kind_ == AllocPayload::OutputInfo::BINARY) {
      if (byte_size > 0) {
        ordered_buffers.push_back(info->evbuffer_);
      }
    } else if (info->kind_ == AllocPayload::OutputInfo::JSON) {
      RETURN_IF_ERR(triton::server::WriteDataToJson(
          &response_outputs, cname, datatype, base, byte_size, element_count));
    }
  }

  evbuffer* response_placeholder = evbuffer_new();
  triton::common::TritonJson::WriteBuffer buffer;

  // Save JSON output
  if (response_outputs.ArraySize() > 0) {
    RETURN_IF_ERR(response_json.Add("Response", std::move(response_outputs)));
    // Write json metadata into response evbuffer
    RETURN_IF_ERR(response_json.Write(&buffer));
    evbuffer_add(response_placeholder, buffer.Base(), buffer.Size());
  }

  // If there is binary data write it next in the appropriate
  // order... also need the HTTP header when returning binary data.
  if (!ordered_buffers.empty()) {
    for (evbuffer* b : ordered_buffers) {
      evbuffer_add_buffer(response_placeholder, b);
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
  SetResponseHeader(!ordered_buffers.empty(), buffer.Size());
  evbuffer_add_buffer(req_->buffer_out, response_body);
  // Destroy the evbuffer object as the data has been moved
  // to HTTP response buffer
  evbuffer_free(response_body);

  return nullptr;  // success
}

}}  // namespace triton::server
