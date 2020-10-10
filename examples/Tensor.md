# Sample Tensor Creation

## Rand

The rand function may be used to generate a tensor of a given shape and configuration, populated with random values

### Calling rand with variable number of arguments

```js
const a = torch.rand(1, 5);
```

### Calling rand with shape array

```js
const a = torch.rand([1, 5]);
```

### Calling rand using option parsing

```js
const a = torch.rand([1, 5], {
  dtype: torch.float64,
});
```

### Calling rand using option parsing

```js
const a = torch.rand([1, 5], {
  dtype: torch.float64,
});
```

## Tensor

The tensor function may be used to create a tensor given an array of values that the tensor would consist of.

### Calling tensor given a nested array of tensor values

```js
const a = torch.tensor([
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6],
]);
```

### Calling tensor given a nested array of tensor values and a specified option type

```js
const a = torch.tensor(
  [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
  ],
  {
    dtype: torch.float64,
  }
);
```
