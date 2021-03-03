// Possible values for dtype
export const float32: number;
export const float64: number;
export const int32: number;

export interface ObjectTensor {
  data: Float32Array | Float64Array | Int32Array;
  shape: number[];
}

export class Tensor {
  static fromObject({ data, shape }: ObjectTensor): Tensor;

  toObject(): ObjectTensor;

  toString(): string;

  cpu(): Tensor;

  cuda(): Tensor;

  clone(): Tensor;

  /**
   * Free the underlying tensor resources.
   *
   * DO NOT use the object again after calling the method.
   */
  free(): void;
}

// raghavmecheri - FIXTHIS: Get rid of circular dependancies if possible

// eslint-disable-next-line @typescript-eslint/no-empty-interface, no-use-before-define
interface TorchTypesArray extends Array<TorchTypes> {}
interface TorchTypesRecord // eslint-disable-line @typescript-eslint/no-empty-interface
  extends Record<string | number | symbol, TorchTypes> {} // eslint-disable-line no-use-before-define

type TorchTypes =
  | TorchTypesArray
  | TorchTypesRecord
  | string
  | number
  | boolean
  | Tensor;

export class ScriptModule {
  constructor(path: string);

  forward(input: TorchTypes): Promise<TorchTypes>;

  forward(...inputs: TorchTypes[]): Promise<TorchTypes>;

  toString(): string;

  cpu(): ScriptModule;

  cuda(): ScriptModule;

  static isCudaAvailable(): boolean;
}

export function rand(shape: number[], options?: { dtype?: number }): Tensor;
// The function actually support options at the end but TS can't express that
export function rand(
  ...shape: number[] /** , options?: {dtype?: number} */
): Tensor;
