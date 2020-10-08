import { describe, expect, test } from "@jest/globals";

const torch = require("../dist");

describe("Tensor creation", () => {
  test("Tensor creation using valid array (1-D)", () => {
    const a = torch.tensor([[1, 2, 3, 4, 5, 6]]).toObject();
    expect(a.data.length).toBe(6);
    expect(a.shape).toMatchObject([1, 6]);

    const b = torch.tensor([[1, 2, 3]]).toObject();
    expect(b.data.length).toBe(3);
    expect(b.shape).toMatchObject([1, 3]);
  });

  test("Tensor creation using valid array (multi-dimensional)", () => {
    const a = torch
      .tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
      ])
      .toObject();
    expect(a.data.length).toBe(6);
    expect(a.shape).toMatchObject([2, 3]);

    const b = torch
      .tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
      ])
      .toObject();
    expect(b.data.length).toBe(9);
    expect(b.shape).toMatchObject([3, 3]);
  });

  test("Tensor creation using valid object", () => {
    const a = torch
      .tensor(new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]), {
        shape: [1, 5],
      })
      .toObject();
    expect(a.data.length).toBe(5);
    expect(a.shape).toMatchObject([1, 5]);

    const b = torch
      .tensor(new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), {
        shape: [3, 3],
      })
      .toObject();
    expect(b.data.length).toBe(9);
    expect(b.shape).toMatchObject([3, 3]);
  });

  test("Tensor creation using invalid params", () => {
    const t = () => torch.tensor();
    const t2 = () => torch.tensor(123);
    const t3 = () => torch.tensor(true);

    expect(t).toThrow("Cannot read property 'length' of undefined");
    expect(t2).toThrow("Invalid argument");
    expect(t3).toThrow("Invalid argument");
  });
});
