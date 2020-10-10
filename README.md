# TorchJS

[![npm version](https://badge.fury.io/js/%40arition%2Ftorch-js.svg)](https://badge.fury.io/js/%40arition%2Ftorch-js)

TorchJS is a JS binding for PyTorch. Its primary objective is to allow running [Torch Script](https://pytorch.org/docs/master/jit.html) inside Node.js program. Complete binding of libtorch is possible but is out-of-scope at the moment.

## Changes after fork

- Support `List` (Javascript `Array`), `Dict` (Javascript `Object`), `String`, `float` (Javascript `number`) as inputs and outputs.

- Add CUDA support.

- Add ops from torchvision.

- Add async support for `forward` function.

## Install

To install the forked version, you can install it from npm:

```bash
yarn add torch-js@npm:@arition/torch-js
```

## Example

In `tests/resources/torch_module.py`, you will find the defination of our test module and the code to generate the trace file.

```python
class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()

    def forward(self, input1, input2):
        return input1 + input2
```

Once you have the trace file, it may be loaded into NodeJS like this

```javascript

```

```javascript
const torch = require("torch-js");
const modelPath = `test_model.pt`;
const model = new torch.ScriptModule(testModelPath);
const inputA = torch.rand([1, 5]);
const inputB = torch.rand([1, 5]);
const res = await model.forward(inputA, inputB);
```

More examples regarding tensor creation, ScriptModule operations, and loading models can be found in our [examples](./examples) folder.

## Installation

This project uses `cmake-js` to build Node extension. Once you have that, you can run

```bash
yarn install
```

And, to test, run:

```bash
node tests/runTorch.js
```

If it failed to run because `libmklml` is missing, you can download it from conda.

```bash
conda install libmklml
```

If conda's `lib` directory is in your path, then you should be able to run the command above. Otherwise, you can set environment variable to point to the directory.

On macOS, it would be:

```bash
DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib/ node tests/runTorch.js
```

On Linux, it should be:

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib/ node tests/runTorch.js
```

If there is any error loading the trace file, you might have to resolve it by installing the matching versions of PyTorch and libtorch and regenerate the file.
