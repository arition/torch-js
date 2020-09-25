import {describe, expect, test} from '@jest/globals'

const torch = require("../dist");

const test_model_path = __dirname + "/resources/test_model.pt";
const script_module = new torch.ScriptModule(test_model_path);

describe('Tensor creation', () => {
	test('Tensor creation using valid array (1-D)', () => {
		// TODO
		expect(true).toBe(true);
	})

	test('Tensor creation using valid array (multi-dimensional)', () => {
		// TODO
		expect(true).toBe(true);
	})

	test('Tensor creation using valid object', () => {
		// TODO
		expect(true).toBe(true);
	})

	test('Tensor creation using invalid params', () => {
		// TODO
		expect(true).toBe(true);
	})
})