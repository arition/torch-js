# Sample Operation Usage

## forward
The script_module module allows for a forward call, that recieves two tensors and returns a Promise as a result.
```js
const a = torch.rand(1, 5);
const b = torch.rand(1, 5);
const res = await script_module.forward(a, b);
```