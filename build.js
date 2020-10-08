const fs = require("fs");

if (fs.existsSync("dist")) {
  fs.rmdirSync("dist", { recursive: true });
}
fs.mkdirSync("dist");
fs.copyFileSync("lib/torch.js", "dist/torch.js");
fs.copyFileSync("lib/torch.d.ts", "dist/torch.d.ts");
require("typescript/bin/tsc");
