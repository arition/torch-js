const torch = require("bindings")("torch-js");
const bindings = require("bindings");
const path = require("path");
const os = require('os'); 

const moduleRoot = bindings.getRoot(bindings.getFileName());
const buildFolder = path.join(moduleRoot, "build", "Release");
const type = os.type();
switch(type) { 
case 'Darwin':
  torch.initenv('DYLD_LIBRARY_PATH', `${buildFolder};${process.env.DYLD_LIBRARY_PATH}`);
  break;
case 'Linux':
  torch.initenv('LD_LIBRARY_PATH', `${buildFolder};${process.env.LD_LIBRARY_PATH}`);
  break;
case 'Windows_NT':
  torch.initenv('PATH', `${buildFolder};${process.env.PATH}`);
  break;
default:
  torch.initenv('LD_LIBRARY_PATH', `${buildFolder};${process.env.LD_LIBRARY_PATH}`);
}

module.exports = torch;
