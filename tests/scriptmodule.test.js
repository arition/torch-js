const torch = require("../dist");
const chai = require('chai');
const expect = chai.expect;

const test_model_path = __dirname + "/resources/test_model.pt";
const script_module = new torch.ScriptModule(test_model_path);

describe('toString function', () => {
	it('toString for valid test_model_path', () => {
		expect(script_module.toString()).to.match(/ScriptModule.*\.pt/)
	})
})