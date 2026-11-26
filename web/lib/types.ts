export type Action = "accept" | "deweight" | "verify" | "quarantine";

export interface SeriesPayload {
  dates: string[];
  observed: (number | null)[];
  expected: (number | null)[];
  clearSky: (number | null)[];
  gapPct: (number | null)[];
  trust: (number | null)[];
  action: Action[];
  attackType: string[];
  alert: boolean[];
  scores: Record<"step" | "drift" | "replay" | "consistency", (number | null)[]>;
}

export interface RunSummary {
  trueStart: string | null;
  trueEnd: string | null;
  firstAlarm: string | null;
  delay: number | null;
  type: string | null;
  estStart: string | null;
  lowestTrust: number | null;
  alarmDays: number;
}

export interface FlaggedDay {
  date: string;
  action: Action;
  trust: number | null;
  gapPct: number | null;
  attackType: string;
  estStart: string | null;
}

export interface RunPayload {
  runId: string;
  attack: string;
  locationLabel?: string;
  demo?: boolean;
  name?: string;
  labels: { start: string | null; end: string | null; type: string | null };
  summary: RunSummary;
  series: SeriesPayload;
  flagged: FlaggedDay[];
  days: number;
  warning?: string | null;
  info?: Record<string, unknown> | null;
  neighbours?: string[];
  episodes?: {
    from: string | null;
    to: string | null;
    days: number;
    type: string;
    estStart: string | null;
  }[];
  firstAlarm?: string | null;
  trustMin?: number | null;
  quarantined?: number;
  csv?: string;
}

export interface BenchmarkRow {
  group: string;
  attack: string;
  n_scenarios: number;
  catch_rate: number;
  median_delay_days: number;
  start_error_median_days: number;
  fa_per_location_year: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  brier: number;
  [key: string]: string | number;
}

export interface Meta {
  testYear: number;
  cleanLocations: number;
  train: string[];
  test: string[];
  model: boolean;
  modelError: string | null;
  tuned: boolean;
  provenance: { ok: boolean; records: number; reason: string | null; head: string | null };
  benchmark: {
    rows: BenchmarkRow[];
    summary: string | null;
    images: { file: string; title: string }[];
  };
  units: string;
  warmupDays: number;
  attackWindow: [string, string];
}

export interface Explanation {
  date: string;
  action: Action;
  trust: number;
  p: number;
  type: string;
  gapPct: number;
  reasons: [string, string][];
  contrib: { detector: string; score: number; p: number }[];
}

export interface LogRecord {
  seq: number;
  kind: string;
  path: string;
  created: string;
  fileHash: string;
  recordHash: string;
  keyId: string;
  ok: boolean;
}

export interface IntegrityPayload {
  records: LogRecord[];
  keys: string[];
  verified: { ok: boolean; records: number; reason: string | null; head: string | null };
  error?: string;
}

export const ACTION_LABEL: Record<Action, string> = {
  accept: "Accept",
  deweight: "De-weight",
  verify: "Verify",
  quarantine: "Quarantine",
};

export const ACTION_COLOR: Record<Action, string> = {
  accept: "#3F5F4B",
  deweight: "#A98B3E",
  verify: "#C07A2A",
  quarantine: "#C8452A",
};
