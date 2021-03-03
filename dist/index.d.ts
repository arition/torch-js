import torch = require("./torch");
export * from "./torch";
declare type Matrix = number[] | NestedArray;
declare type NestedArray = Array<Matrix>;
declare type TensorDataCompatible = Float32Array | Float64Array | Int32Array | Matrix;
export declare function tensor(data: TensorDataCompatible, { dtype, shape }?: {
    dtype?: number;
    shape?: number[];
}): torch.Tensor;
