
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
#pragma once

#include <mutex>

#include "common.h"
#include "dirent.h"
#include "http_server.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace server {

// Handle AdsBrain HTTP requests to inference server APIs
class AdsBrainAPIServer : public HTTPAPIServer {
 public:
  static TRITONSERVER_Error* Create(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& smb_manager,
      const int32_t port, const std::string address, const int thread_cnt,
      std::unique_ptr<HTTPServer>* http_server);

  class AdsBainInferRequestClass : public HTTPAPIServer::InferRequestClass {
   public:
    enum RequestType { TRITON, ADSBRAIN_BOND };

    explicit AdsBainInferRequestClass(
        TRITONSERVER_Server* server, evhtp_request_t* req,
        DataCompressor::Type response_compression_type)
        : InferRequestClass(server, req, response_compression_type)
    {
      auto request_type =
          GetEnvironmentVariableOrDefault("AB_REQUEST_TYPE", "TRITON");
      if (request_type == "TRITON") {
        request_type_ = RequestType::TRITON;
      } else if (request_type == "ADSBRAIN_BOND") {
        request_type_ = RequestType::ADSBRAIN_BOND;
      } else {
        LOG_INFO << request_type
                 << " is not supported, so use the default triton format."
                 << " Set the environment variable AB_REQUEST_TYPE"
                 << " ([TRITON, ADSBRAIN_BOND]) to switch the request format.";
        request_type_ = RequestType::TRITON;
      }
    }

    TRITONSERVER_Error* FinalizeResponse(
        TRITONSERVER_InferenceResponse* response) override;

   private:
    TRITONSERVER_Error* FinalizeResponseInTritonFormat(
        TRITONSERVER_InferenceResponse* response)
    {
      return InferRequestClass::FinalizeResponse(response);
    }

    TRITONSERVER_Error* FinalizeResponseInBondFormat(
        TRITONSERVER_InferenceResponse* response);

    RequestType request_type_ = RequestType::TRITON;
  };


 protected:
  std::unique_ptr<InferRequestClass> CreateInferRequest(
      evhtp_request_t* req) override
  {
    return std::unique_ptr<InferRequestClass>(new AdsBainInferRequestClass(
        server_.get(), req, GetResponseCompressionType(req)));
  }

 private:
  explicit AdsBrainAPIServer(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const int32_t port, const std::string address, const int thread_cnt)
      : HTTPAPIServer(
            server, trace_manager, shm_manager, port, address, thread_cnt)
  {
  }
};

}}  // namespace triton::server
