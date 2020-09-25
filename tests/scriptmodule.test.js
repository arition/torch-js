import {describe, expect, test} from '@jest/globals'

const torch = require("../dist");

const test_model_path = __dirname + "/resources/test_model.pt";
const script_module = new torch.ScriptModule(test_model_path);

describe('Forward function', () => {
	test('Call to forward using valid tensor params', () => {
		expect(script_module.toString()).toMatch(/ScriptModule.*\.pt/)
	})
})