const torch = require("../dist");
const chai = require('chai');
const expect = chai.expect;

const test_model_path = __dirname + "/resources/test_model.pt";
const script_module = new torch.ScriptModule(test_model_path);

describe('Tensor creation', () => {
	it('Tensor creation using valid array (1-D)', () => {
		// TODO
		return true
	})

	it('Tensor creation using valid array (multi-dimensional)', () => {
		// TODO
		return true
	})

	it('Tensor creation using valid object', () => {
		// TODO
		return true
	})

	it('Tensor creation using invalid params', () => {
		// TODO
		return true
	})
})