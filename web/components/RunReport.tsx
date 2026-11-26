"use client";

import { useEffect, useState } from "react";

import SeriesChart from "@/components/SeriesChart";
import { explainDay } from "@/lib/api";
import { day, longDay, pct, signedPct } from "@/lib/format";
import {
  ACTION_COLOR,
  ACTION_LABEL,
  type Action,
  type Explanation,
  type RunPayload,
} from "@/lib/types";

interface Props {
  run: RunPayload;
  onPickDay?: (date: string) => void;
  compact?: boolean;
}

export default function RunReport({ run, onPickDay }: Props) {
  const [selected, setSelected] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const flagged = run.flagged;
  const firstAlarm = run.summary.firstAlarm;
  const defaultDay = firstAlarm && flagged.some((row) => row.date === firstAlarm)
    ? firstAlarm
    : flagged[0]?.date ?? null;

  useEffect(() => {
    setSelected(defaultDay);
    setExplanation(null);
    setError(null);
  }, [run.runId, defaultDay]);

  useEffect(() => {
    if (!selected) return;
    let alive = true;
    setLoading(true);
    explainDay(run.runId, selected)
      .then((data) => {
        if (alive) {
          setExplanation(data);
          setError(null);
        }
      })
      .catch((exc: Error) => {
        if (alive) setError(exc.message);
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [run.runId, selected]);

  const severity: Record<Action, number> = { accept: 0, deweight: 1, verify: 2, quarantine: 3 };
  const verdict: Action = flagged.reduce<Action>(
    (worst, row) => (severity[row.action] > severity[worst] ? row.action : worst),
    "accept",
  );
  const caught = run.summary.firstAlarm !== null;

  return (
    <div className="space-y-8">
      {run.demo && (
        <p className="border-l-2 border-l-[#A98B3E] bg-[#A98B3E]/[0.07] px-4 py-3 text-[0.82rem] leading-relaxed text-graphite">
          Demo mode: real clean data or the trained model is missing, so this run uses a synthetic series.
        </p>
      )}
      {run.warning && (
        <p className="border-l-2 border-l-[#A98B3E] bg-[#A98B3E]/[0.07] px-4 py-3 text-[0.82rem] leading-relaxed text-graphite">
          {run.warning}
        </p>
      )}

      <div className="grid gap-6 border-b border-[color:var(--edge)] pb-8 lg:grid-cols-12">
        <div className="lg:col-span-4">
          <p className="eyebrow mb-3">Verdict</p>
          <p
            className="display text-5xl"
            style={{ color: run.summary.trueStart && !caught ? "#6B6C70" : ACTION_COLOR[verdict] }}
          >
            {run.summary.trueStart && !caught ? "Missed" : ACTION_LABEL[verdict]}
          </p>
          <p className="mt-3 max-w-sm text-[0.84rem] leading-relaxed text-graphite">
            {run.summary.trueStart
              ? caught
                ? `Attack placed on ${longDay(run.summary.trueStart)}; first alarm after ${run.summary.delay} day${run.summary.delay === 1 ? "" : "s"}, labelled “${run.summary.type}”.`
                : `Attack placed on ${longDay(run.summary.trueStart)} and never flagged in this run.`
              : `Control run with no attack: ${run.summary.alarmDays} day${run.summary.alarmDays === 1 ? "" : "s"} flagged out of ${run.days}.`}
          </p>
          <p className="mt-4 font-mono text-[0.62rem] uppercase tracking-widest text-ash">
            {run.locationLabel ?? run.name} · {run.attack}
          </p>
        </div>

        <dl className="grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-3 lg:col-span-8 lg:grid-cols-4">
          <Stat label="True start" value={day(run.summary.trueStart)} note="hidden from the detector" />
          <Stat label="First alarm" value={day(run.summary.firstAlarm)} note="verify or quarantine" accent={caught} />
          <Stat label="Delay" value={run.summary.delay === null ? "—" : `${run.summary.delay} d`} note={run.summary.type ?? "no type"} />
          <Stat label="Lowest trust" value={run.summary.lowestTrust?.toFixed(2) ?? "—"} note={`${run.summary.alarmDays} alarm days`} />
        </dl>
      </div>

      {run.episodes && run.episodes.length > 0 && (
        <div className="border border-[color:var(--edge)]">
          <p className="eyebrow border-b border-[color:var(--edge)] px-5 py-3">Alarm episodes</p>
          <table className="w-full text-left text-[0.8rem]">
            <thead>
              <tr className="border-b border-[color:var(--edge)] font-mono text-[0.6rem] uppercase tracking-widest text-ash">
                <th className="px-5 py-2 font-normal">From</th>
                <th className="px-5 py-2 font-normal">To</th>
                <th className="px-5 py-2 font-normal">Days</th>
                <th className="px-5 py-2 font-normal">Type</th>
                <th className="px-5 py-2 font-normal">Likely start</th>
              </tr>
            </thead>
            <tbody>
              {run.episodes.map((episode, i) => (
                <tr key={i} className="border-b border-[color:var(--edge-soft)] last:border-0">
                  <td className="num px-5 py-2">{day(episode.from)}</td>
                  <td className="num px-5 py-2">{day(episode.to)}</td>
                  <td className="num px-5 py-2">{episode.days}</td>
                  <td className="px-5 py-2">{episode.type}</td>
                  <td className="num px-5 py-2">{day(episode.estStart)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div>
        <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
          <p className="eyebrow">Signals</p>
          <p className="text-[0.74rem] text-ash">
            Hover for values · click a day to explain it · shaded band is the true attack period
          </p>
        </div>
        <SeriesChart
          series={run.series}
          labels={run.labels}
          highlight={selected}
          onPick={(date) => {
            setSelected(date);
            onPickDay?.(date);
          }}
        />
      </div>

      {flagged.length === 0 ? (
        <p className="border-l-2 border-l-[#3F5F4B] bg-[#3F5F4B]/[0.06] px-4 py-3 text-[0.85rem] text-graphite">
          No day was flagged: every value was accepted. That is the correct answer for a clean run.
        </p>
      ) : (
        <div className="grid gap-8 lg:grid-cols-12">
          <div className="lg:col-span-4">
            <p className="eyebrow mb-3">Flagged days · {flagged.length}</p>
            <div className="max-h-[320px] overflow-y-auto border border-[color:var(--edge)]">
              {flagged.map((row) => (
                <button
                  key={row.date}
                  type="button"
                  onClick={() => setSelected(row.date)}
                  className={`flex w-full items-baseline justify-between gap-3 border-b border-[color:var(--edge-soft)] px-4 py-2.5 text-left text-[0.78rem] transition-colors last:border-0 ${
                    selected === row.date ? "bg-ink text-paper" : "hover:bg-ink/[0.04]"
                  }`}
                >
                  <span className="num">{day(row.date)}</span>
                  <span
                    className="font-mono text-[0.6rem] uppercase tracking-wider"
                    style={{ color: selected === row.date ? undefined : ACTION_COLOR[row.action] }}
                  >
                    {ACTION_LABEL[row.action]}
                  </span>
                  <span className="num text-ash">{row.trust?.toFixed(2)}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="lg:col-span-8">
            <p className="eyebrow mb-3">Why this day</p>
            {error && <p className="text-[0.85rem] text-signal">{error}</p>}
            {loading && !explanation && <p className="text-[0.85rem] text-ash">Reading the run…</p>}
            {explanation && (
              <div className="space-y-5">
                <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
                  <span
                    className="border px-3 py-1.5 font-mono text-[0.62rem] uppercase tracking-widest"
                    style={{ borderColor: ACTION_COLOR[explanation.action], color: ACTION_COLOR[explanation.action] }}
                  >
                    {ACTION_LABEL[explanation.action]}
                  </span>
                  <span className="num text-[0.78rem] text-graphite">{longDay(explanation.date)}</span>
                  <span className="num text-[0.78rem] text-graphite">trust {explanation.trust.toFixed(2)}</span>
                  <span className="num text-[0.78rem] text-graphite">p(compromise) {explanation.p.toFixed(2)}</span>
                  <span className="num text-[0.78rem] text-graphite">gap {signedPct(explanation.gapPct)}</span>
                </div>

                <div className="grid gap-6 md:grid-cols-2">
                  <div>
                    <p className="eyebrow mb-3">Detector contribution</p>
                    <ul className="space-y-2.5">
                      {explanation.contrib.map((item) => (
                        <li key={item.detector}>
                          <div className="flex items-baseline justify-between text-[0.76rem]">
                            <span>{item.detector}</span>
                            <span className="num text-ash">
                              {item.score.toFixed(2)} → {pct(item.p, 0)}
                            </span>
                          </div>
                          <div className="mt-1 h-[3px] w-full bg-[color:var(--edge-soft)]">
                            <div
                              className="h-full"
                              style={{
                                width: `${Math.min(100, item.p * 100)}%`,
                                background: item.p > 0.6 ? "#C8452A" : item.p > 0.35 ? "#C07A2A" : "#3F5F4B",
                              }}
                            />
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <ol className="space-y-2.5">
                    {explanation.reasons.map(([level, text], i) => (
                      <li
                        key={i}
                        className="border-l-2 pl-3 text-[0.82rem] leading-relaxed text-graphite"
                        style={{
                          borderColor:
                            level === "strong" ? "#C8452A" : level === "medium" ? "#C07A2A" : "rgba(11,11,12,0.18)",
                        }}
                      >
                        {text}
                      </li>
                    ))}
                  </ol>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {run.csv && (
        <div>
          <a
            className="btn btn-ghost"
            href={`data:text/csv;charset=utf-8,${encodeURIComponent(run.csv)}`}
            download={`veritas_${run.name ?? "score"}.csv`}
          >
            Download the verdict for every day
          </a>
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  note,
  accent,
}: {
  label: string;
  value: string;
  note?: string;
  accent?: boolean;
}) {
  return (
    <div>
      <dt className="font-mono text-[0.6rem] uppercase tracking-widest text-ash">{label}</dt>
      <dd className={`num mt-1 text-xl ${accent ? "text-signal" : ""}`}>{value}</dd>
      {note && <p className="mt-0.5 text-[0.68rem] text-ash">{note}</p>}
    </div>
  );
}
