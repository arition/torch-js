import {describe, expect, test} from '@jest/globals'

const torch = require("../dist");

const test_model_path = __dirname + "/resources/test_model.pt";
const script_module = new torch.ScriptModule(test_model_path);

describe('Random tensor creation', () => {
	test('Random tensor creation using variable number of arguements', () => {
		// TODO
		expect(true).toBe(true);
	})

	test('Random tensor creation using shape array', () => {
		// TODO
		expect(true).toBe(true);
	})

	test('Random tensor creation using option parsing', () => {
		// TODO
		expect(true).toBe(true);
	})

	test('Random tensor creation using invalid params', () => {
		// TODO
		expect(true).toBe(true);
	})
})