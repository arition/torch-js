const torch = require("../dist");
const chai = require('chai');
const expect = chai.expect;

const test_model_path = __dirname + "/resources/test_model.pt";
const script_module = new torch.ScriptModule(test_model_path);

describe('Random tensor creation', () => {
	it('Random tensor creation using variable number of arguements', () => {
		// TODO
		return true
	})

	it('Random tensor creation using shape array', () => {
		// TODO
		return true
	})

	it('Random tensor creation using option parsing', () => {
		// TODO
		return true
	})

	it('Random tensor creation using invalid params', () => {
		// TODO
		return true
	})
})