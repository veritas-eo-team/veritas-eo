"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import PageHeader from "@/components/PageHeader";
import ScenarioControls, { defaults, type ScenarioValues } from "@/components/ScenarioControls";
import SeriesChart from "@/components/SeriesChart";
import { getMeta, runScenario } from "@/lib/api";
import { day, longDay, pct } from "@/lib/format";
import {
  ACTION_COLOR,
  ACTION_LABEL,
  type Action,
  type Meta,
  type RunPayload,
  type SeriesPayload,
} from "@/lib/types";

export default function ReplayPage() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [values, setValues] = useState<ScenarioValues>(defaults(null));
  const [randomStart, setRandomStart] = useState(false);
  const [speed, setSpeed] = useState(18);
  const [run, setRun] = useState<RunPayload | null>(null);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    getMeta()
      .then((data) => {
        setMeta(data);
        setValues((current) => ({ ...current, location: data.test[0] ?? data.train[0] ?? current.location }));
      })
      .catch((exc: Error) => setError(exc.message));
  }, []);

  useEffect(() => {
    if (timer.current) clearInterval(timer.current);
    if (!playing || !run) return;
    timer.current = setInterval(() => {
      setCursor((value) => {
        if (!run) return value;
        if (value >= run.series.dates.length - 1) {
          setPlaying(false);
          return value;
        }
        return value + 1;
      });
    }, Math.max(16, 1000 / speed));
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [playing, speed, run]);

  const start = async () => {
    setBusy(true);
    setError(null);
    setPlaying(false);
    setRun(null);
    try {
      const payload = await runScenario({
        location: values.location,
        attack: values.attack,
        size: values.size === null ? null : values.size / 100,
        duration: values.duration,
        noise: values.noise === null ? null : values.noise / 100,
        start: values.attack === "none" || randomStart ? null : values.start,
        seed: values.seed,
      });
      const first = payload.labels.start ? payload.series.dates.indexOf(payload.labels.start) : -1;
      const from = first > 0 ? Math.max(0, first - 40) : Math.max(0, payload.series.dates.length - 190);
      setRun(payload);
      setCursor(from);
      setPlaying(true);
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const startIndex = run?.labels.start ? run.series.dates.indexOf(run.labels.start) : -1;
  const viewFrom = startIndex > 0 ? Math.max(0, startIndex - 40) : Math.max(0, (run?.series.dates.length ?? 0) - 190);
  const visible = useMemo(() => {
    if (!run) return null;
    return sliceSeries(run.series, viewFrom, Math.max(cursor, viewFrom));
  }, [run, viewFrom, cursor]);

  const currentDay = run?.series.dates[cursor] ?? null;
  const badge: Action = (run?.series.action[cursor] as Action | undefined) ?? "accept";
  const trust = run?.series.trust[cursor] ?? null;

  const alerts = useMemo(() => {
    if (!run) return [];
    const out: { date: string; type: string }[] = [];
    for (let i = viewFrom; i <= cursor; i += 1) {
      if (run.series.alert[i]) {
        out.push({ date: run.series.dates[i] ?? "", type: run.series.attackType[i] ?? "none" });
      }
    }
    return out.slice(-4).reverse();
  }, [run, cursor, viewFrom]);

  const finished = run ? cursor >= run.series.dates.length - 1 : false;

  return (
    <>
      <PageHeader
        index="03"
        section="Live replay"
        title="Every morning, one decision."
        description="The pipeline is causal: each day's verdict uses only data up to that day. Replaying a run one day at a time is therefore identical to running it in operation, which is what this page does."
        aside={<span className="font-mono text-[0.6rem] uppercase tracking-widest text-ash">Causality is tested, not assumed</span>}
      />

      <div className="page grid gap-10 pb-10 lg:grid-cols-12">
        <aside className="lg:col-span-4">
          <div className="lg:sticky lg:top-24">
            <ScenarioControls
              meta={meta}
              values={values}
              onChange={setValues}
              onRun={start}
              busy={busy}
              actionLabel="Start replay"
              randomStart={randomStart}
              onRandomStart={setRandomStart}
            >
              <div>
                <div className="mb-1 flex items-baseline justify-between">
                  <span className="label !mb-0">Speed</span>
                  <span className="num text-sm">{speed} days / s</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={120}
                  step={1}
                  value={speed}
                  onChange={(event) => setSpeed(Number(event.target.value))}
                  className="h-1 w-full cursor-pointer appearance-none bg-[color:var(--edge)]"
                  style={{ accentColor: "#0B0B0C" }}
                />
              </div>
            </ScenarioControls>
          </div>
        </aside>

        <div className="lg:col-span-8">
          {error && (
            <p className="border-l-2 border-l-signal bg-signal/[0.06] px-4 py-3 text-[0.85rem] leading-relaxed text-graphite">
              {error}
            </p>
          )}

          {!run && (
            <div className="panel flex min-h-[440px] flex-col justify-between p-8">
              <div>
                <p className="eyebrow">Standby</p>
                <p className="display mt-4 max-w-md text-4xl">Choose a scenario, then press start.</p>
                <p className="mt-4 max-w-md text-[0.88rem] leading-relaxed text-graphite">
                  The badge below the controls shows the action the operator would take each morning, and the feed
                  records the first time quarantine starts.
                </p>
              </div>
              <div className="tick mt-10 w-full opacity-30" />
            </div>
          )}

          {run && visible && (
            <div className="space-y-8">
              <div className="grid gap-6 md:grid-cols-12">
                <div className="md:col-span-5">
                  <div
                    className="flex min-h-[140px] flex-col justify-center border px-6 py-5"
                    style={{ borderColor: ACTION_COLOR[badge], background: `${ACTION_COLOR[badge]}0D` }}
                  >
                    <p className="font-mono text-[0.6rem] uppercase tracking-widest text-ash">
                      {currentDay ? longDay(currentDay) : "—"}
                    </p>
                    <p className="display mt-2 text-4xl" style={{ color: ACTION_COLOR[badge] }}>
                      {ACTION_LABEL[badge]}
                    </p>
                    <p className="num mt-2 text-[0.78rem] text-graphite">
                      trust {trust === null ? "—" : trust.toFixed(2)} · gap{" "}
                      {run.series.gapPct[cursor] === null ? "—" : `${(run.series.gapPct[cursor] ?? 0).toFixed(1)}%`}
                    </p>
                  </div>
                </div>

                <div className="md:col-span-7">
                  <div className="flex items-center justify-between">
                    <p className="eyebrow">Alert feed</p>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        className="border border-[color:var(--edge)] px-3 py-1 font-mono text-[0.58rem] uppercase tracking-wider hover:border-ink"
                        onClick={() => setPlaying((value) => !value)}
                        disabled={finished}
                      >
                        {playing ? "Pause" : "Resume"}
                      </button>
                      <button
                        type="button"
                        className="border border-[color:var(--edge)] px-3 py-1 font-mono text-[0.58rem] uppercase tracking-wider hover:border-ink"
                        onClick={() => {
                          setCursor(run.series.dates.length - 1);
                          setPlaying(false);
                        }}
                      >
                        Skip to end
                      </button>
                    </div>
                  </div>
                  <ul className="mt-3 space-y-2">
                    {alerts.length === 0 ? (
                      <li className="border border-[color:var(--edge)] px-4 py-3 text-[0.82rem] text-graphite">
                        Monitoring. No alert so far{cursor > 0 ? ` in ${cursor + 1} days` : ""}.
                      </li>
                    ) : (
                      alerts.map((alert) => (
                        <li
                          key={alert.date}
                          className="border-l-2 border-l-signal bg-signal/[0.05] px-4 py-2.5 text-[0.82rem]"
                        >
                          <span className="font-mono text-[0.62rem] uppercase tracking-wider text-signal">Alert</span>{" "}
                          {day(alert.date)} · quarantine starts · detected type “{alert.type}”
                        </li>
                      ))
                    )}
                  </ul>
                  <div className="mt-4 h-[2px] w-full bg-[color:var(--edge)]">
                    <div
                      className="h-full bg-ink transition-all duration-200"
                      style={{ width: `${((cursor + 1) / run.series.dates.length) * 100}%` }}
                    />
                  </div>
                  <p className="mt-2 font-mono text-[0.6rem] uppercase tracking-wider text-ash">
                    {cursor + 1} / {run.series.dates.length} days
                    {run.summary.trueStart ? ` · attack starts ${day(run.summary.trueStart)} (hidden)` : " · clean run"}
                  </p>
                </div>
              </div>

              <div className="panel p-4">
                <SeriesChart series={visible} labels={run.labels} showControls={false} />
              </div>

              {finished && (
                <p className="border-t border-[color:var(--edge)] pt-4 text-[0.86rem] text-graphite">
                  {run.summary.firstAlarm
                    ? `Detected ${run.summary.delay} day${run.summary.delay === 1 ? "" : "s"} after the attack began, labelled “${run.summary.type}”. Lowest trust after the start: ${pct(run.summary.lowestTrust, 0)}.`
                    : run.summary.trueStart
                      ? `No alarm after the attack start: ${run.summary.alarmDays} day(s) flagged in total, all before it.`
                      : `Clean run finished with ${run.summary.alarmDays} flagged day(s) out of ${run.days}.`}
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

function sliceSeries(series: SeriesPayload, from: number, to: number): SeriesPayload {
  const cut = <T,>(arr: T[]): T[] => arr.slice(from, to + 1);
  return {
    dates: cut(series.dates),
    observed: cut(series.observed),
    expected: cut(series.expected),
    clearSky: cut(series.clearSky),
    gapPct: cut(series.gapPct),
    trust: cut(series.trust),
    action: cut(series.action),
    attackType: cut(series.attackType),
    alert: cut(series.alert),
    scores: {
      step: cut(series.scores.step),
      drift: cut(series.scores.drift),
      replay: cut(series.scores.replay),
      consistency: cut(series.scores.consistency),
    },
  };
}
