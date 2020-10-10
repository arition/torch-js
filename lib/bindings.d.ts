type bindingFunction = (mod: string) => any;
interface bindingInterface extends bindingFunction {
  getRoot: (file: string) => string;
  getFileName: (callingFile?: string) => string;
}
declare let bindings: bindingInterface;

declare module "bindings" {
  export = bindings;
}
