import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import process from "node:process";

const root = path.resolve(import.meta.dirname, "..", "..");
const venv = process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python";
const local = path.join(root, venv);
const python = process.env.VERITAS_PYTHON || (existsSync(local) ? local : "python");

export function startApi(port = process.env.VERITAS_API_PORT || "8077") {
  const child = spawn(python, ["scripts/api_server.py", "--port", String(port)], {
    cwd: root,
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
  child.stdout.on("data", (chunk) => process.stdout.write(`[api] ${chunk}`));
  child.stderr.on("data", (chunk) => process.stderr.write(`[api] ${chunk}`));
  return child;
}

if (import.meta.url === `file://${process.argv[1]?.replace(/\\/g, "/")}`) {
  const child = startApi();
  const stop = () => child.kill();
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
  child.on("exit", (code) => process.exit(code ?? 0));
}
