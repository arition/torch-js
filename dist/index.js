"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    Object.defineProperty(o, k2, { enumerable: true, get: function() { return m[k]; } });
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.tensor = void 0;
const assert = require("assert");
const torch = require("./torch");
__exportStar(require("./torch"), exports);
function flattenArray(arr) {
    const shape = [arr.length];
    while (arr.length > 0 && Array.isArray(arr[0])) {
        shape.push(arr[0].length);
        // Forcing Array<any> is required to make TS happy; otherwise, TS2349
        arr = arr.reduce((acc, val) => acc.concat(val), []);
        // Running assert in every step to make sure that shape is regular
        const numel = shape.reduce((acc, cur) => acc * cur);
        assert.strictEqual(arr.length, numel);
    }
    return {
        data: arr,
        shape,
    };
}
function typedArrayType(dtype) {
    switch (dtype) {
        case torch.float32:
            return Float32Array;
        case torch.float64:
            return Float64Array;
        case torch.int32:
            return Int32Array;
        default:
            throw new TypeError("Unsupported dtype");
    }
}
function tensor(data, { dtype = torch.float32, shape } = {}) {
    if (Array.isArray(data)) {
        const arrayAndShape = flattenArray(data);
        const typedArray = new (typedArrayType(dtype))(arrayAndShape.data);
        return torch.Tensor.fromObject({
            data: typedArray,
            shape: arrayAndShape.shape,
        });
    }
    if (shape === undefined) {
        shape = [data.length];
    }
    return torch.Tensor.fromObject({
        data,
        shape,
    });
}
exports.tensor = tensor;
