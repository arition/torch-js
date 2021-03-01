#include "ScriptModule.h"

#include <math.h>
#include <exception>
#include "Tensor.h"
#include "utils.h"
#include "FunctionWorker.h"

namespace torchjs
{
  Napi::Object ScriptModule::Init(Napi::Env env, Napi::Object exports)
  {
    Napi::Function func = DefineClass(env, "ScriptModule",
                                      {InstanceMethod("forward", &ScriptModule::forward),
                                       InstanceMethod("toString", &ScriptModule::toString),
                                       InstanceMethod("cpu", &ScriptModule::cpu),
                                       InstanceMethod("cuda", &ScriptModule::cuda),
                                       StaticMethod("isCudaAvailable", &ScriptModule::isCudaAvailable)});

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    exports.Set("ScriptModule", func);
    return exports;
  }

  ScriptModule::ScriptModule(const Napi::CallbackInfo &info) : Napi::ObjectWrap<ScriptModule>(info)
  {
    try
    {
      torch::NoGradGuard no_grad;
      Napi::HandleScope scope(info.Env());
      Napi::String value = info[0].As<Napi::String>();
      path_ = value;
      module_ = torch::jit::load(value);
    }
    catch (const std::exception &e)
    {
      throw Napi::Error::New(info.Env(), e.what());
    }
  }

  Napi::FunctionReference ScriptModule::constructor;

  Napi::Value ScriptModule::forward(const Napi::CallbackInfo &info)
  {
    try
    {
      torch::NoGradGuard no_grad;
      module_.eval();

      auto len = info.Length();
      auto inputs = std::vector<torch::jit::IValue>();
      for (size_t i = 0; i < len; ++i)
      {
        inputs.push_back(JSTypeToIValue(info.Env(), info[i]));
      }
      
      auto worker = new FunctionWorker<c10::IValue>(
          info.Env(),
          [=]() -> c10::IValue {
            torch::NoGradGuard no_grad;
            return module_.forward(inputs);
          },
          [=](Napi::Env env, c10::IValue value) -> Napi::Value {
            return IValueToJSType(env, value);
          });

      worker->Queue();
      return worker->GetPromise();
    }
    catch (const std::exception &e)
    {
      throw Napi::Error::New(info.Env(), e.what());
    }
  }

  c10::IValue ScriptModule::JSTypeToIValue(Napi::Env env, const Napi::Value &jsValue)
  {
    Napi::HandleScope scope(env);
    if (jsValue.IsArray())
    {
      auto jsList = jsValue.As<Napi::Array>();
      auto len = jsList.Length();
      if (len == 0)
      {
        throw Napi::Error::New(env, "Empty array is not supported");
      }
      auto firstElement = JSTypeToIValue(env, jsList[(uint32_t)0]);
      c10::List<c10::IValue> list(firstElement.type());
      for (uint32_t i = 1; i < len; ++i)
      {
        list.push_back(JSTypeToIValue(env, jsList[i]));
      }
      return list;
    }
    else if (jsValue.IsObject())
    {
      auto jsObject = jsValue.As<Napi::Object>();
      if (Tensor::IsInstance(jsObject))
      {
        auto tensor = Napi::ObjectWrap<Tensor>::Unwrap(jsObject);
        return c10::IValue(tensor->tensor());
      }
      throw Napi::Error::New(env, "Object/Dict input is not implemented");
    }
    else if (jsValue.IsNumber())
    {
      auto _jsNumber = jsValue.As<Napi::Number>().DoubleValue();
      if (fmod(_jsNumber, 1) == 0){
        int jsNumber = int(_jsNumber);
        return c10::IValue(jsNumber);
      }
      else{
        return c10::IValue(_jsNumber);
      }
    }
    else if (jsValue.IsBoolean())
    {
      auto jsBool = jsValue.As<Napi::Boolean>().Value();
      return c10::IValue(jsBool);
    }
    else if (jsValue.IsString())
    {
      auto jsString = jsValue.As<Napi::String>().Utf8Value();
      return c10::IValue(jsString);
    }
    throw Napi::Error::New(env, "Unsupported javascript input type");
  }

  Napi::Value ScriptModule::IValueToJSType(Napi::Env env, const c10::IValue &iValue)
  {
    Napi::EscapableHandleScope scope(env);
    if (iValue.isTensor())
    {
      return scope.Escape(Tensor::FromTensor(env, iValue.toTensor()));
    }
    else if (iValue.isList())
    {
      auto list = iValue.toList();
      auto jsList = Napi::Array::New(env);
      for (auto i = 0; i < list.size(); i++)
      {
        jsList[i] = IValueToJSType(env, list[i]);
      }
      return scope.Escape(jsList);
    }
    else if (iValue.isGenericDict())
    {
      auto dict = iValue.toGenericDict();
      auto jsDict = Napi::Object::New(env);
      for (auto iter = dict.begin(); iter != dict.end(); iter++)
      {
        auto key = IValueToJSType(env, iter->key());
        auto value = IValueToJSType(env, iter->value());
        jsDict.Set(key, value);
      }
      return scope.Escape(jsDict);
    }
    else if (iValue.isInt())
    {
      return scope.Escape(Napi::Number::New(env, iValue.toInt()));
    }
    else if (iValue.isDouble())
    {
      return scope.Escape(Napi::Number::New(env, iValue.toDouble()));
    }
    else if (iValue.isBool())
    {
      return scope.Escape(Napi::Boolean::New(env, iValue.toBool()));
    }
    else if (iValue.isString())
    {
      return scope.Escape(Napi::String::New(env, iValue.toString().get()->string()));
    }
    else if (iValue.isTuple())
    {
      auto list = iValue.toTuple()->elements();
      auto jsList = Napi::Array::New(env);
      for (auto i = 0; i < list.size(); i++)
      {
        jsList[i] = IValueToJSType(env, list[i]);
      }
      return scope.Escape(jsList);
    }
    throw Napi::Error::New(env, "Unsupported output type from ScriptModule");
  }

  Napi::Value ScriptModule::toString(const Napi::CallbackInfo &info)
  {
    return Napi::String::New(info.Env(), "ScriptModule(\"" + path_ + "\")");
  }

  Napi::Value ScriptModule::cpu(const Napi::CallbackInfo &info)
  {
    try
    {
      module_.to(at::kCPU);
      return info.This();
    }
    catch (const std::exception &e)
    {
      throw Napi::Error::New(info.Env(), e.what());
    }
  }

  Napi::Value ScriptModule::cuda(const Napi::CallbackInfo &info)
  {
    try
    {
      if (torch::cuda::is_available())
      {
        module_.to(at::kCUDA);
      }
      else
      {
        throw Napi::Error::New(info.Env(), "CUDA is not aviliable");
      }
      return info.This();
    }
    catch (const std::exception &e)
    {
      throw Napi::Error::New(info.Env(), e.what());
    }
  }

  Napi::Value ScriptModule::isCudaAvailable(const Napi::CallbackInfo &info)
  {
    try
    {
      return Napi::Boolean::New(info.Env(), torch::cuda::is_available());
    }
    catch (const std::exception &e)
    {
      throw Napi::Error::New(info.Env(), e.what());
    }
  }

} // namespace torchjs