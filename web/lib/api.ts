import type { Explanation, IntegrityPayload, Meta, RunPayload } from "./types";

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(
      "The local VERITAS API is not answering. Start it with `python scripts/api_server.py` (or `npm run dev` in web/).",
    );
  }
  const text = await response.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    throw new ApiError(`The API returned a non-JSON response (${response.status}).`);
  }
  if (!response.ok) {
    const message =
      data && typeof data === "object" && "error" in data
        ? String((data as { error: unknown }).error)
        : `Request failed (${response.status}).`;
    throw new ApiError(message);
  }
  return data as T;
}

export function getMeta(): Promise<Meta> {
  return request<Meta>("/api/meta");
}

export interface ScenarioRequest {
  location: string;
  attack: string;
  size?: number | null;
  duration?: number | null;
  noise?: number | null;
  start?: string | null;
  seed?: number;
}

export function runScenario(body: ScenarioRequest): Promise<RunPayload> {
  return request<RunPayload>("/api/scenario", { method: "POST", body: JSON.stringify(body) });
}

export function explainDay(runId: string, day: string): Promise<Explanation> {
  const query = new URLSearchParams({ run: runId, day });
  return request<Explanation>(`/api/explain?${query.toString()}`);
}

export function getIntegrity(): Promise<IntegrityPayload> {
  return request<IntegrityPayload>("/api/integrity");
}

export type IntegrityAction = "verify" | "tamper" | "gateway" | "overhead";

export function postIntegrity<T>(action: IntegrityAction): Promise<T> {
  return request<T>(`/api/integrity/${action}`, { method: "POST", body: "{}" });
}

export interface ScoreRequest {
  csv: string;
  name: string;
  lat: number;
  lon: number;
  units: "mj" | "kwh" | "wm2";
  dateCol?: string;
  valueCol?: string;
  knownStart?: string | null;
}

export function scoreCsv(body: ScoreRequest): Promise<RunPayload> {
  return request<RunPayload>("/api/score", { method: "POST", body: JSON.stringify(body) });
}

export interface DemoCsv {
  csv: string;
  name: string;
  lat: number;
  lon: number;
  units: "mj" | "kwh" | "wm2";
  knownStart: string;
}

export function getDemoCsv(): Promise<DemoCsv> {
  return request<DemoCsv>("/api/demo-csv");
}
