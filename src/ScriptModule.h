#pragma once

#include <memory>

#include <napi.h>
#include <torch/script.h>

namespace torchjs
{
  class ScriptModule : public Napi::ObjectWrap<ScriptModule>
  {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    ScriptModule(const Napi::CallbackInfo &info);

  private:
    static Napi::FunctionReference constructor;
    torch::jit::script::Module module_;
    std::string path_;

    Napi::Value forward(const Napi::CallbackInfo &info);
    Napi::Value call_scripted_function(const Napi::CallbackInfo &info);
    Napi::Value toString(const Napi::CallbackInfo &info);
    Napi::Value cpu(const Napi::CallbackInfo &info);
    Napi::Value cuda(const Napi::CallbackInfo &info);
    static Napi::Value isCudaAvailable(const Napi::CallbackInfo &info);
    static Napi::Value IValueToJSType(Napi::Env env, const c10::IValue &iValue);
    static std::pair<c10::IValue, bool> JSTypeToIValue(Napi::Env env, const Napi::Value &jsValue);
  };

} // namespace torchjs