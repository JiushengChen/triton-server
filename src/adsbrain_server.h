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
      const std::string entrypoint,
      std::unique_ptr<HTTPServer>* adsbrain_server);

 private:
  explicit AdsBrainAPIServer(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const int32_t port, const std::string address, const int thread_cnt,
      const std::string entrypoint)
      : HTTPAPIServer(
            server, trace_manager, shm_manager, port, address, thread_cnt),
        entrypoint_(entrypoint)
  {}

  void Handle(evhtp_request_t* req) override;
  std::string entrypoint_;
};

}}  // namespace triton::server
