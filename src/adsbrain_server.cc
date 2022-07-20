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

}}  // namespace triton::server
