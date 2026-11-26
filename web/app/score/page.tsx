"use client";

import { useRef, useState } from "react";

import PageHeader from "@/components/PageHeader";
import RunReport from "@/components/RunReport";
import { getDemoCsv, scoreCsv } from "@/lib/api";
import { longDay } from "@/lib/format";
import type { RunPayload } from "@/lib/types";

const UNITS: { value: "mj" | "kwh" | "wm2"; label: string }[] = [
  { value: "mj", label: "MJ/m²/day · NASA POWER" },
  { value: "kwh", label: "kWh/m²/day" },
  { value: "wm2", label: "daily mean W/m²" },
];

export default function ScorePage() {
  const [file, setFile] = useState<File | null>(null);
  const [csv, setCsv] = useState<string | null>(null);
  const [name, setName] = useState("uploaded_series");
  const [lat, setLat] = useState(35.68);
  const [lon, setLon] = useState(10.1);
  const [units, setUnits] = useState<"mj" | "kwh" | "wm2">("mj");
  const [dateCol, setDateCol] = useState("");
  const [valueCol, setValueCol] = useState("");
  const [known, setKnown] = useState(false);
  const [knownStart, setKnownStart] = useState("2024-05-01");
  const [run, setRun] = useState<RunPayload | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement | null>(null);

  const accept = async (picked: File) => {
    setError(null);
    setRun(null);
    const text = await picked.text();
    setFile(picked);
    setCsv(text);
    setName(picked.name.replace(/\.[^.]+$/, ""));
  };

  const submit = async () => {
    if (!csv) return;
    setBusy(true);
    setError(null);
    try {
      const payload = await scoreCsv({
        csv,
        name,
        lat,
        lon,
        units,
        dateCol: dateCol.trim() || undefined,
        valueCol: valueCol.trim() || undefined,
        knownStart: known ? knownStart : null,
      });
      setRun(payload);
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const demo = async () => {
    setBusy(true);
    setError(null);
    try {
      const payload = await getDemoCsv();
      setCsv(payload.csv);
      setName(payload.name);
      setLat(payload.lat);
      setLon(payload.lon);
      setUnits(payload.units);
      setKnown(true);
      setKnownStart(payload.knownStart);
      setFile(null);
      const result = await scoreCsv({
        csv: payload.csv,
        name: `${payload.name} (Kairouan, +12% from 1 May 2024)`,
        lat: payload.lat,
        lon: payload.lon,
        units: payload.units,
        knownStart: payload.knownStart,
      });
      setRun(result);
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader
        index="06"
        section="Assess your data"
        title="Bring a series from anywhere."
        description="A CSV with one date column and one daily value is enough. VERITAS learns the site's normal behaviour from the first year, then checks every day after it. Include nb_* and ctx_* columns to work offline."
        aside={<span className="font-mono text-[0.6rem] uppercase tracking-widest text-ash">No retraining required</span>}
      />

      <div className="page grid gap-10 pb-10 lg:grid-cols-12">
        <div className="space-y-6 lg:col-span-5">
          <div
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              const picked = event.dataTransfer.files?.[0];
              if (picked) void accept(picked);
            }}
            className={`border border-dashed p-8 transition-colors ${
              dragging ? "border-ink bg-ink/[0.03]" : "border-[color:var(--edge)] bg-bone"
            }`}
          >
            <p className="eyebrow">Daily CSV</p>
            <p className="display mt-3 text-3xl">{file ? file.name : "Drop a file, or choose one."}</p>
            <p className="mt-2 text-[0.8rem] leading-relaxed text-graphite">
              One row per day. Common column names such as date, value, ghi or radiation are detected
              automatically.
            </p>
            <div className="mt-5 flex flex-wrap gap-3">
              <button type="button" className="btn btn-ghost !py-2.5" onClick={() => input.current?.click()}>
                Choose file
              </button>
              <button type="button" className="btn btn-ghost !py-2.5" onClick={demo} disabled={busy}>
                Try the demo
              </button>
            </div>
            <input
              ref={input}
              type="file"
              accept=".csv,.txt"
              className="hidden"
              onChange={(event) => {
                const picked = event.target.files?.[0];
                if (picked) void accept(picked);
              }}
            />
            {file && (
              <p className="mt-4 font-mono text-[0.6rem] uppercase tracking-wider text-ash">
                {(file.size / 1024).toFixed(0)} KB · kept in this browser tab only
              </p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label" htmlFor="lat">
                Latitude
              </label>
              <input
                id="lat"
                type="number"
                step="0.01"
                min={-90}
                max={90}
                className="field num"
                value={lat}
                onChange={(event) => setLat(Number(event.target.value))}
              />
            </div>
            <div>
              <label className="label" htmlFor="lon">
                Longitude
              </label>
              <input
                id="lon"
                type="number"
                step="0.01"
                min={-180}
                max={180}
                className="field num"
                value={lon}
                onChange={(event) => setLon(Number(event.target.value))}
              />
            </div>
          </div>

          <div>
            <label className="label" htmlFor="units">
              Units of your values
            </label>
            <select id="units" className="field" value={units} onChange={(event) => setUnits(event.target.value as typeof units)}>
              {UNITS.map((unit) => (
                <option key={unit.value} value={unit.value}>
                  {unit.label}
                </option>
              ))}
            </select>
          </div>

          <details className="border border-[color:var(--edge)] p-4">
            <summary className="cursor-pointer font-mono text-[0.62rem] uppercase tracking-widest text-graphite">
              Advanced
            </summary>
            <div className="mt-4 space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="label" htmlFor="dateCol">
                    Date column
                  </label>
                  <input
                    id="dateCol"
                    className="field"
                    placeholder="auto"
                    value={dateCol}
                    onChange={(event) => setDateCol(event.target.value)}
                  />
                </div>
                <div>
                  <label className="label" htmlFor="valueCol">
                    Value column
                  </label>
                  <input
                    id="valueCol"
                    className="field"
                    placeholder="auto"
                    value={valueCol}
                    onChange={(event) => setValueCol(event.target.value)}
                  />
                </div>
              </div>
              <label className="flex items-center gap-2 text-[0.8rem] text-graphite">
                <input
                  type="checkbox"
                  checked={known}
                  onChange={(event) => setKnown(event.target.checked)}
                  className="h-3.5 w-3.5 accent-ink"
                />
                I know when an attack starts in this file
              </label>
              {known && (
                <div>
                  <label className="label" htmlFor="knownStart">
                    True attack start
                  </label>
                  <input
                    id="knownStart"
                    type="date"
                    className="field"
                    value={knownStart}
                    onChange={(event) => setKnownStart(event.target.value)}
                  />
                </div>
              )}
            </div>
          </details>

          <button type="button" className="btn w-full justify-between disabled:opacity-40" onClick={submit} disabled={!csv || busy}>
            <span>{busy ? "Scoring…" : "Score this file"}</span>
            <span aria-hidden="true">→</span>
          </button>
          <p className="text-[0.72rem] leading-relaxed text-ash">
            The first run may download context from NASA POWER. Coordinates are only used for that context; the file
            never leaves this machine.
          </p>
        </div>

        <div className="lg:col-span-7">
          {error && (
            <p className="border-l-2 border-l-signal bg-signal/[0.06] px-4 py-3 text-[0.85rem] leading-relaxed text-graphite">
              {error}
            </p>
          )}

          {!run && !busy && !error && (
            <div className="panel flex min-h-[440px] flex-col justify-between p-8">
              <div>
                <p className="eyebrow">No verdict yet</p>
                <p className="display mt-4 max-w-md text-4xl">A hidden +12% step is a fair first test.</p>
                <p className="mt-4 max-w-md text-[0.88rem] leading-relaxed text-graphite">
                  The demo file is two years of a test site with a bias applied from 1 May, including context
                  columns so it works without an internet connection.
                </p>
              </div>
              <div className="mt-10 grid gap-4 border-t border-[color:var(--edge)] pt-6 sm:grid-cols-3">
                {[
                  ["Input", "date + daily value"],
                  ["Reference", "first year, assumed clean"],
                  ["Output", "trust and action per day"],
                ].map(([title, body]) => (
                  <div key={title}>
                    <p className="font-mono text-[0.6rem] uppercase tracking-widest text-ash">{title}</p>
                    <p className="mt-1 text-[0.82rem] text-graphite">{body}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {busy && (
            <div className="panel flex min-h-[440px] items-center justify-center">
              <div className="text-center">
                <p className="font-mono text-[0.62rem] uppercase tracking-widest text-ash">Scoring</p>
                <p className="display mt-3 text-3xl">Context, calibration, detectors…</p>
                <div className="tick mx-auto mt-6 w-40 opacity-40" />
              </div>
            </div>
          )}

          {run && !busy && (
            <div className="space-y-6">
              <div className="border-y border-[color:var(--edge)] py-4 text-[0.82rem] leading-relaxed text-graphite">
                <p className="num">
                  {run.name} · {run.days} days
                  {typeof run.info?.valid_days === "number" ? ` (${run.info.valid_days} with values)` : ""} · context{" "}
                  {String(run.info?.mode ?? "—")} · {run.neighbours?.length ?? 0} neighbours
                </p>
                {known && knownStart && (
                  <p className="mt-1 font-mono text-[0.6rem] uppercase tracking-wider text-ash">
                    True start {longDay(knownStart)} · reference period at the head of the file is assumed clean
                  </p>
                )}
              </div>
              <RunReport run={run} />
            </div>
          )}
        </div>
      </div>
    </>
  );
}
