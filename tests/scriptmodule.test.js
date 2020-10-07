import { describe, expect, test } from "@jest/globals";

const torch = require("../dist");

const testModelPath = `${__dirname}/resources/test_model.pt`;

describe("Constructor", () => {
  test("Call constructor using valid model path", () => {
    const scriptModule = new torch.ScriptModule(testModelPath);
    expect(scriptModule.toString()).toMatch(/ScriptModule.*\.pt/);
  });

  // TODO -- Fine tune error message to remove stacktrace for error thrown on invalid model file
  test("Call constructor from invalid model path", () => {
    const t = () => new torch.ScriptModule("/resources/no_model.pt");
    expect(t).toThrow(
      new RegExp("open file failed, file path: /resources/no_model.pt.*")
    );
    expect(true).toEqual(true);
  });

  test("Call constructor with missing params", () => {
    const t = () => new torch.ScriptModule();
    expect(t).toThrow(new Error("A string was expected"));
  });

  test("Call constructor with invalid params", () => {
    const t = () => new torch.ScriptModule(true);
    const t2 = () => new torch.ScriptModule(123);
    const t3 = () => new torch.ScriptModule(122.3);
    expect(t).toThrow(new Error("A string was expected"));
    expect(t2).toThrow(new Error("A string was expected"));
    expect(t3).toThrow(new Error("A string was expected"));
  });
});

describe("toString", () => {
  test("Call toString using valid model path", () => {
    const scriptModule = new torch.ScriptModule(testModelPath);
    expect(scriptModule.toString()).toMatch(/ScriptModule.*\.pt/);
  });

  test("Call toString using valid model path with variable params (shouldn't make a difference)", () => {
    const scriptModule = new torch.ScriptModule(testModelPath);
    expect(scriptModule.toString(3)).toMatch(/ScriptModule.*\.pt/);
    expect(scriptModule.toString(false)).toMatch(/ScriptModule.*\.pt/);
    expect(scriptModule.toString(3.233)).toMatch(/ScriptModule.*\.pt/);
    expect(scriptModule.toString(["Hello world"])).toMatch(
      /ScriptModule.*\.pt/
    );
  });
});

describe("Forward function", () => {
  test("Call to forward using valid tensor params", async () => {
    const scriptModule = new torch.ScriptModule(testModelPath);
    const a = torch.tensor([
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ]);
    const b = torch.tensor([
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ]);
    const res = await scriptModule.forward(a, b);
    expect(res.toObject().data.length).toEqual(6);
    expect(res.toObject().shape).toMatchObject([2, 3]);
  });

  // TODO -- Fine tune error message to remove stacktrace for missing arguement value -- at the moment, the multiple lines make it difficult to test for an error message
  test("Call to forward using missing second tensor param", async () => {
    const scriptModule = new torch.ScriptModule(testModelPath);
    const a = torch.tensor([
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ]);
    expect(scriptModule.forward(a)).rejects.toThrow(new RegExp(/[\s\S]/));
  });
});
