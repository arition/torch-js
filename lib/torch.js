const torch = require("bindings")("torch-js");

const bindings = require("bindings");
const path = require("path");

const moduleRoot = bindings.getRoot(bindings.getFileName());
const buildFolder = path.join(moduleRoot, "build", "Release");
torch.initenv(`${buildFolder};${process.env.path}`);

module.exports = torch;
