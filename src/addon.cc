#include <napi.h>

#include "ATen.h"
#include "ScriptModule.h"
#include "Tensor.h"
#include "constants.h"
#include <torchvision/vision.h>

using namespace torchjs;

// void InitTorchVisionOps()
// {
//   // Create a useless model to ask compiler to link torchvision libs
//   auto model = vision::models::ResNet18();
//   model->eval();
// }

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
  // InitTorchVisionOps();
  aten::Init(env, exports);
  constants::Init(env, exports);
  ScriptModule::Init(env, exports);
  Tensor::Init(env, exports);
  return exports;
}

NODE_API_MODULE(torchjs, Init);