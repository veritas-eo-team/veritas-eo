import { spawn } from "node:child_process";
import path from "node:path";
import process from "node:process";

import { startApi } from "./api.mjs";

const here = import.meta.dirname;
const api = startApi();
const next = spawn(
  process.execPath,
  [path.join(here, "..", "node_modules", "next", "dist", "bin", "next"), "dev", "-p", "3000", "-H", "127.0.0.1"],
  {
    cwd: path.join(here, ".."),
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, VERITAS_API: `http://127.0.0.1:${process.env.VERITAS_API_PORT || "8077"}` },
  },
);

next.stdout.on("data", (chunk) => process.stdout.write(String(chunk)));
next.stderr.on("data", (chunk) => process.stderr.write(String(chunk)));

const stop = () => {
  next.kill();
  api.kill();
};

process.on("SIGINT", stop);
process.on("SIGTERM", stop);
next.on("exit", (code) => {
  api.kill();
  process.exit(code ?? 0);
});
api.on("exit", (code) => {
  if (code && code !== 0) {
    console.error(`[api] exited with code ${code}`);
    next.kill();
    process.exit(code);
  }
});
