"use strict";
const torch = require("../dist");

var test_model_path = __dirname + "/resources/test_model.pt";

var script_module = new torch.ScriptModule(test_model_path);
console.log(script_module.toString());

console.log(`CUDA Avaliable: ${torch.ScriptModule.isCudaAvailable()}`)

async function test() {
  // Testing rand() with variable number of arguments
  var a = torch.rand(1, 5);

  // Testing rand() with shape array
  var b = torch.rand([1, 5]);
  console.log(typeof b.toObject().shape);
  console.log(b.toObject().shape.constructor.name);
  console.log(typeof [1,2,3,4,5]);

  var c = await script_module.forward(a, b);
  console.log(c.toObject());

  // Testing initialization from object
  var d = torch.tensor(
    new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]), {
      shape: [1, 5],
    });
  console.log(d.toObject());

  var e = await script_module.forward(c, d);
  console.log(e.toObject());

  // Testing option parsing
  var f = torch.rand([1, 5], {
    dtype: torch.float64
  });
  console.log(f.toObject());

  var g = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
  ]);
  console.log(g.toObject());

  var h = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
  ], {
    dtype: torch.float64
  });
  console.log(h.toObject());
}

test();
