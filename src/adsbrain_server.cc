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
#include <algorithm>


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
    const std::string input_output_json,
    std::unique_ptr<HTTPServer>* http_server)
{
  http_server->reset(new AdsBrainAPIServer(
      server, trace_manager, shm_manager, port, address, thread_cnt,
      entrypoint, request_compressor, response_compressor, input_output_json));

  const std::string addr = address + ":" + std::to_string(port);
  LOG_INFO << "Started AdsBrain HTTPService at " << addr
           << ", entrypoint: " << entrypoint
           << ", request_compressor: " << (int)request_compressor
           << ", response_compressor: " << (int)response_compressor
           << ", input_output_json: " << input_output_json;

  return nullptr;
}

AdsBrainAPIServer::AdsBrainAPIServer(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const int32_t port, const std::string address, const int thread_cnt,
      const std::string entrypoint,
      DataCompressor::Type request_compressor,
      DataCompressor::Type response_compressor,
      const std::string input_output_json)
      : HTTPAPIServer(
            server, trace_manager, shm_manager, port, address, thread_cnt),
        entrypoint_(entrypoint),
        request_compressor_(request_compressor),
        response_compressor_(response_compressor)
  {
    global_request_json_length_ = input_output_json.length();
    if (global_request_json_length_ > 0) {
      TRITONSERVER_Error* err = 
        global_request_json_.Parse(input_output_json.c_str(), global_request_json_length_);
      if (err != nullptr) {
        LOG_ERROR << "[adsbrain] failed to parse json: " << input_output_json;
      }
    }
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
AdsBrainAPIServer::EVBufferToInput(
    const std::string& model_name, TRITONSERVER_InferenceRequest* irequest,
    evbuffer* input_buffer, InferRequestClass* infer_req, size_t header_length)
{
  // Extract individual input data from HTTP body and register in
  // 'irequest'. The HTTP body is not necessarily stored in contiguous
  // memory.
  //
  // Get the addr and size of each chunk of memory holding the HTTP
  // body.
  struct evbuffer_iovec* v = nullptr;
  int v_idx = 0;

  int n = evbuffer_peek(input_buffer, -1, NULL, NULL, 0);
  if (n > 0) {
    v = static_cast<struct evbuffer_iovec*>(
        alloca(sizeof(struct evbuffer_iovec) * n));
    if (evbuffer_peek(input_buffer, -1, NULL, v, n) != n) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "unexpected error getting input buffers");
    }
  }

  if (n != 1) {
    return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("Only support single input now! Got: ") +
           std::to_string(n)).c_str());
  }

  if (global_request_json_length_ > 0) {
    header_length = global_request_json_length_;
  }
  else {
    // get real header length
    header_length = *(unsigned int *)((char *)v[n-1].iov_base + v[n-1].iov_len - 4);
    if (header_length >= v[n-1].iov_len - 4) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("tailed inference header size should be in range (0, ") +
           std::to_string(v[n-1].iov_len - 4) + "), got: " + std::to_string(header_length))
              .c_str());
    }
    v[n-1].iov_len -= 4;
  }

  // Extract just the json header from the HTTP body. 'header_length == 0' means
  // that the entire HTTP body should be input data for a raw binary request.
  triton::common::TritonJson::Value* request_json_ptr = nullptr;
  triton::common::TritonJson::Value request_json;
  if (global_request_json_length_ > 0) {
    request_json_ptr = &global_request_json_;
  }
  else {
    RETURN_IF_ERR(EVBufferToJson(&request_json, v, &v_idx, header_length, n));
    request_json_ptr = &request_json;
  }

  // Set InferenceRequest request_id
  triton::common::TritonJson::Value id_json;
  if (request_json_ptr->Find("id", &id_json)) {
    const char* id;
    size_t id_len;
    RETURN_MSG_IF_ERR(id_json.AsString(&id, &id_len), "Unable to parse 'id'");
    RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetId(irequest, id));
  }

  // The default setting for returned outputs (JSON or BINARY). This
  // is needed for the case when outputs are not explicitly specified.
  AllocPayload::OutputInfo::Kind default_output_kind =
      AllocPayload::OutputInfo::JSON;

  // Set sequence correlation ID and flags if any
  triton::common::TritonJson::Value params_json;
  if (request_json_ptr->Find("parameters", &params_json)) {
    triton::common::TritonJson::Value seq_json;
    if (params_json.Find("sequence_id", &seq_json)) {
      // Try to parse sequence_id as uint64_t
      uint64_t seq_id;
      if (seq_json.AsUInt(&seq_id) != nullptr) {
        // On failure try to parse as a string
        std::string seq_id;
        RETURN_MSG_IF_ERR(
            seq_json.AsString(&seq_id), "Unable to parse 'sequence_id'");
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetCorrelationIdString(
            irequest, seq_id.c_str()));
      } else {
        RETURN_IF_ERR(
            TRITONSERVER_InferenceRequestSetCorrelationId(irequest, seq_id));
      }
    }

    uint32_t flags = 0;

    {
      triton::common::TritonJson::Value start_json;
      if (params_json.Find("sequence_start", &start_json)) {
        bool start = false;
        RETURN_MSG_IF_ERR(
            start_json.AsBool(&start), "Unable to parse 'sequence_start'");
        if (start) {
          flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
        }
      }

      triton::common::TritonJson::Value end_json;
      if (params_json.Find("sequence_end", &end_json)) {
        bool end = false;
        RETURN_MSG_IF_ERR(
            end_json.AsBool(&end), "Unable to parse 'sequence_end'");
        if (end) {
          flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;
        }
      }
    }

    RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetFlags(irequest, flags));

    {
      triton::common::TritonJson::Value priority_json;
      if (params_json.Find("priority", &priority_json)) {
        uint64_t p = 0;
        RETURN_MSG_IF_ERR(
            priority_json.AsUInt(&p), "Unable to parse 'priority'");
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetPriority(irequest, p));
      }
    }

    {
      triton::common::TritonJson::Value timeout_json;
      if (params_json.Find("timeout", &timeout_json)) {
        uint64_t t;
        RETURN_MSG_IF_ERR(timeout_json.AsUInt(&t), "Unable to parse 'timeout'");
        RETURN_IF_ERR(
            TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(irequest, t));
      }
    }

    {
      triton::common::TritonJson::Value bdo_json;
      if (params_json.Find("binary_data_output", &bdo_json)) {
        bool bdo = false;
        RETURN_MSG_IF_ERR(
            bdo_json.AsBool(&bdo), "Unable to parse 'binary_data_output'");
        default_output_kind = (bdo) ? AllocPayload::OutputInfo::BINARY
                                    : AllocPayload::OutputInfo::JSON;
      }
    }
  }

  // Get the byte-size for each input and from that get the blocks
  // holding the data for that input
  triton::common::TritonJson::Value inputs_json;
  RETURN_MSG_IF_ERR(
      request_json_ptr->MemberAsArray("inputs", &inputs_json),
      "Unable to parse 'inputs'");

  if (inputs_json.ArraySize() != 1) {
    return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Only support one input now! Got  ") +
            std::to_string(inputs_json.ArraySize())).c_str());
  }

  for (size_t i = 0; i < inputs_json.ArraySize(); i++) {
    triton::common::TritonJson::Value request_input;
    RETURN_IF_ERR(inputs_json.At(i, &request_input));
    RETURN_IF_ERR(ValidateInputContentType(request_input));

    const char* input_name;
    size_t input_name_len;
    RETURN_MSG_IF_ERR(
        request_input.MemberAsString("name", &input_name, &input_name_len),
        "Unable to parse 'name'");

    const char* datatype;
    size_t datatype_len;
    RETURN_MSG_IF_ERR(
        request_input.MemberAsString("datatype", &datatype, &datatype_len),
        "Unable to parse 'datatype'");
    const TRITONSERVER_DataType dtype = TRITONSERVER_StringToDataType(datatype);

    triton::common::TritonJson::Value shape_json;
    RETURN_MSG_IF_ERR(
        request_input.MemberAsArray("shape", &shape_json),
        "Unable to parse 'shape'");
    std::vector<int64_t> shape_vec;
    for (size_t i = 0; i < shape_json.ArraySize(); i++) {
      uint64_t d = 0;
      RETURN_MSG_IF_ERR(
          shape_json.IndexAsUInt(i, &d), "Unable to parse 'shape'");
      shape_vec.push_back(d);
    }

    RETURN_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
        irequest, input_name, dtype, &shape_vec[0], shape_vec.size()));

    bool binary_input;
    size_t byte_size;
    RETURN_IF_ERR(
        CheckBinaryInputData(request_input, &binary_input, &byte_size));

    if ((byte_size == 0) && binary_input) {
      RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
          irequest, input_name, nullptr, 0 /* byte_size */,
          TRITONSERVER_MEMORY_CPU, 0 /* memory_type_id */));
    } else if (binary_input) {
      if (header_length == 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "must specify valid 'Infer-Header-Content-Length' in request "
            "header and 'binary_data_size' when passing inputs in binary "
            "data format");
      }

      if (global_request_json_length_ > 0) {
        uint32_t qlen = v[v_idx].iov_len;
        byte_size = 0;
        v_idx = n;
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
              irequest, input_name, static_cast<const void*>(&qlen), sizeof(uint32_t), TRITONSERVER_MEMORY_CPU,
              0 /* memory_type_id */));
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
              irequest, input_name, v[v_idx].iov_base, qlen, TRITONSERVER_MEMORY_CPU,
              0 /* memory_type_id */));
      }
      else {
        // Process one block at a time
        while ((byte_size > 0) && (v_idx < n)) {
          char* base = static_cast<char*>(v[v_idx].iov_base);
          size_t base_size;
          if (v[v_idx].iov_len > byte_size) {
            base_size = byte_size;
            v[v_idx].iov_base = static_cast<void*>(base + byte_size);
            v[v_idx].iov_len -= byte_size;
            byte_size = 0;
          } else {
            base_size = v[v_idx].iov_len;
            byte_size -= v[v_idx].iov_len;
            v_idx++;
          }

          RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
              irequest, input_name, base, base_size, TRITONSERVER_MEMORY_CPU,
              0 /* memory_type_id */));
        }
      }

      if (byte_size != 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "unexpected size for input '" + std::string(input_name) +
                "', expecting " + std::to_string(byte_size) +
                " additional bytes for model '" + model_name + "'")
                .c_str());
      }
    } else {
      // Process input if in shared memory.
      bool use_shm;
      uint64_t shm_offset;
      const char* shm_region;
      RETURN_IF_ERR(CheckSharedMemoryData(
          request_input, &use_shm, &shm_region, &shm_offset, &byte_size));
      if (use_shm) {
        void* base;
        TRITONSERVER_MemoryType memory_type;
        int64_t memory_type_id;
        RETURN_IF_ERR(shm_manager_->GetMemoryInfo(
            shm_region, shm_offset, &base, &memory_type, &memory_type_id));
        if (memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
          cudaIpcMemHandle_t* cuda_handle;
          RETURN_IF_ERR(shm_manager_->GetCUDAHandle(shm_region, &cuda_handle));
          TRITONSERVER_BufferAttributes* buffer_attributes;
          RETURN_IF_ERR(TRITONSERVER_BufferAttributesNew(&buffer_attributes));
          auto buffer_attributes_del =
              [](TRITONSERVER_BufferAttributes* buffer_attributes) {
                TRITONSERVER_BufferAttributesDelete(buffer_attributes);
              };

          std::unique_ptr<
              TRITONSERVER_BufferAttributes, decltype(buffer_attributes_del)>
              buffer_attrsl(buffer_attributes, buffer_attributes_del);
          RETURN_IF_ERR(TRITONSERVER_BufferAttributesSetMemoryType(
              buffer_attributes, memory_type));
          RETURN_IF_ERR(TRITONSERVER_BufferAttributesSetMemoryTypeId(
              buffer_attributes, memory_type_id));
          RETURN_IF_ERR(TRITONSERVER_BufferAttributesSetCudaIpcHandle(
              buffer_attributes, reinterpret_cast<void*>(cuda_handle)));
          RETURN_IF_ERR(TRITONSERVER_BufferAttributesSetByteSize(
              buffer_attributes, byte_size));
          RETURN_IF_ERR(
              TRITONSERVER_InferenceRequestAppendInputDataWithBufferAttributes(
                  irequest, input_name, base, buffer_attributes));
#endif
        } else {
          RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
              irequest, input_name, base, byte_size, memory_type,
              memory_type_id));
        }
      } else {
        const int64_t element_cnt = GetElementCount(shape_vec);

        // FIXME, element count should never be 0 or negative so
        // shouldn't we just return an error here?
        if (element_cnt == 0) {
          RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
              irequest, input_name, nullptr, 0 /* byte_size */,
              TRITONSERVER_MEMORY_CPU, 0 /* memory_type_id */));
        } else {
          // JSON... presence of "data" already validated but still
          // checking here. Flow in this endpoint needs to be
          // reworked...
          triton::common::TritonJson::Value tensor_data;
          RETURN_MSG_IF_ERR(
              request_input.MemberAsArray("data", &tensor_data),
              "Unable to parse 'data'");

          if (dtype == TRITONSERVER_TYPE_BYTES) {
            RETURN_IF_ERR(JsonBytesArrayByteSize(tensor_data, &byte_size));
          } else {
            byte_size = element_cnt * TRITONSERVER_DataTypeByteSize(dtype);
          }

          infer_req->serialized_data_.emplace_back();
          std::vector<char>& serialized = infer_req->serialized_data_.back();
          serialized.resize(byte_size);

          RETURN_IF_ERR(ReadDataFromJson(
              input_name, tensor_data, &serialized[0], dtype,
              dtype == TRITONSERVER_TYPE_BYTES ? byte_size : element_cnt));
          RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
              irequest, input_name, &serialized[0], serialized.size(),
              TRITONSERVER_MEMORY_CPU, 0 /* memory_type_id */));
        }
      }
    }
  }

  if (v_idx != n) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected additional input data for model '" + model_name + "'")
            .c_str());
  }

  // outputs is optional
  if (request_json_ptr->Find("outputs")) {
    triton::common::TritonJson::Value outputs_json;
    RETURN_MSG_IF_ERR(
        request_json_ptr->MemberAsArray("outputs", &outputs_json),
        "Unable to parse 'outputs'");
    for (size_t i = 0; i < outputs_json.ArraySize(); i++) {
      triton::common::TritonJson::Value request_output;
      RETURN_IF_ERR(outputs_json.At(i, &request_output));
      RETURN_IF_ERR(ValidateOutputParameter(request_output));

      const char* output_name;
      size_t output_name_len;
      RETURN_MSG_IF_ERR(
          request_output.MemberAsString("name", &output_name, &output_name_len),
          "Unable to parse 'name'");
      RETURN_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(
          irequest, output_name));

      uint64_t class_size;
      RETURN_IF_ERR(CheckClassificationOutput(request_output, &class_size));

      bool use_shm;
      uint64_t offset, byte_size;
      const char* shm_region;
      RETURN_IF_ERR(CheckSharedMemoryData(
          request_output, &use_shm, &shm_region, &offset, &byte_size));

      // ValidateOutputParameter ensures that both shm and
      // classification cannot be true.
      if (use_shm) {
        void* base;
        TRITONSERVER_MemoryType memory_type;
        int64_t memory_type_id;
        RETURN_IF_ERR(shm_manager_->GetMemoryInfo(
            shm_region, offset, &base, &memory_type, &memory_type_id));

        if (memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
          cudaIpcMemHandle_t* cuda_handle;
          RETURN_IF_ERR(shm_manager_->GetCUDAHandle(shm_region, &cuda_handle));
          infer_req->alloc_payload_.output_map_.emplace(
              std::piecewise_construct, std::forward_as_tuple(output_name),
              std::forward_as_tuple(new AllocPayload::OutputInfo(
                  base, byte_size, memory_type, memory_type_id,
                  reinterpret_cast<char*>(cuda_handle))));
#endif
        } else {
          infer_req->alloc_payload_.output_map_.emplace(
              std::piecewise_construct, std::forward_as_tuple(output_name),
              std::forward_as_tuple(new AllocPayload::OutputInfo(
                  base, byte_size, memory_type, memory_type_id,
                  nullptr /* cuda ipc handle */)));
        }
      } else {
        bool use_binary;
        RETURN_IF_ERR(CheckBinaryOutputData(request_output, &use_binary));
        infer_req->alloc_payload_.output_map_.emplace(
            std::piecewise_construct, std::forward_as_tuple(output_name),
            std::forward_as_tuple(new AllocPayload::OutputInfo(
                use_binary ? AllocPayload::OutputInfo::BINARY
                           : AllocPayload::OutputInfo::JSON,
                class_size)));
      }
    }
  }

  infer_req->alloc_payload_.default_output_kind_ = default_output_kind;

  return nullptr;  // success
}

}}  // namespace triton::server
