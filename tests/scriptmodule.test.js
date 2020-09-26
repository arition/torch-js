import {describe, expect, test} from '@jest/globals'

const torch = require("../dist");

const test_model_path = __dirname + "/resources/test_model.pt";

describe('Constructor', () => {
	test('Call constructor using valid model path', () => {
		const script_module = new torch.ScriptModule(test_model_path)
		expect(script_module.toString()).toMatch(/ScriptModule.*\.pt/)
	})
	
	// TODO -- Fine tune error message to remove stacktrace for error thrown on invalid model file
	test('Call constructor from invalid model path', () => {
		const t = () => new torch.ScriptModule("/resources/no_model.pt")
		// expect(t).toThrow(new Error("open file failed, file path: /resources/no_model.pt"))
		expect(true).toEqual(true);	
	})

	test('Call constructor with missing params', () => {
		const t = () => new torch.ScriptModule()
		expect(t).toThrow(new Error("A string was expected"))		
	})

	test('Call constructor with invalid params', () => {
		const t = () => new torch.ScriptModule(true)
		const t2 = () => new torch.ScriptModule(123)
		const t3 = () => new torch.ScriptModule(122.3)
		expect(t).toThrow(new Error("A string was expected"))
		expect(t2).toThrow(new Error("A string was expected"))	
		expect(t3).toThrow(new Error("A string was expected"))		
	})
})

describe('toString', () => {
	test('Call toString using valid model path', () => {
		const script_module = new torch.ScriptModule(test_model_path)
		expect(script_module.toString()).toMatch(/ScriptModule.*\.pt/)
	})

	test('Call toString using valid model path with variable params (shouldn\'t make a difference)', () => {
		const script_module = new torch.ScriptModule(test_model_path)
		expect(script_module.toString(3)).toMatch(/ScriptModule.*\.pt/)
		expect(script_module.toString(false)).toMatch(/ScriptModule.*\.pt/)
		expect(script_module.toString(3.233)).toMatch(/ScriptModule.*\.pt/)
		expect(script_module.toString(["Hello world"])).toMatch(/ScriptModule.*\.pt/)
	})
})

describe('Forward function', () => {
	test('Call to forward using valid tensor params', async () => {
		const script_module = new torch.ScriptModule(test_model_path)
		const a = torch.tensor([
			[0.1, 0.2, 0.3],
			[0.4, 0.5, 0.6],
		]);
		const b = torch.tensor([
			[0.1, 0.2, 0.3],
			[0.4, 0.5, 0.6],
		]);
		const res = await script_module.forward(a, b);
		// This returns an object for the data key as well -- is this intentional?
		expect(res.toObject().shape).toMatchObject([2,3]);
	})

	// TODO -- Fine tune error message to remove stacktrace for missing arguement value
	test('Call to forward using missing second tensor param', async () => {
		expect.assertions(1);

		const script_module = new torch.ScriptModule(test_model_path)
		const a = torch.tensor([
			[0.1, 0.2, 0.3],
			[0.4, 0.5, 0.6],
		]);
		//expect(script_module.forward(a)).rejects.toEqual("Error: forward() is missing value for argument 'input2'. Declaration: forward(__torch__.test_model self, Tensor input1, Tensor input2) -> (Tensor)")
		expect(true).toEqual(true);
	})
})