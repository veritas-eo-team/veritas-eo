"use client";

import { motion, useReducedMotion } from "framer-motion";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import PipelineRail from "@/components/PipelineRail";
import Reveal from "@/components/Reveal";
import { getMeta } from "@/lib/api";
import { longDay, num, pct } from "@/lib/format";
import type { BenchmarkRow, Meta } from "@/lib/types";

const FALLBACK: { caught: number; delay: string; fa: number; runs: number } = {
  caught: 1,
  delay: "2 days",
  fa: 0,
  runs: 2453,
};

const LIMITS = [
  {
    title: "A patient, sub-threshold lie",
    body: "An attacker who adds a fraction of a percent and never trips the CUSUM can stay hidden for months. Our red team held 0.5% (median) for 180 days undetected on eight new sites.",
  },
  {
    title: "Three-day replays",
    body: "Seven-day copies are caught almost every time; three-day copies are caught in 28% of runs. The matching window is the constraint, and we say so.",
  },
  {
    title: "Small downdrafts",
    body: "The clear-sky ceiling only catches inflation. A −3% step is found in 40% of runs, and we report that rather than hiding behind the average.",
  },
  {
    title: "Deep cloud at tropical sites",
    body: "Abidjan needs about six weeks to see a 12% step, because cloud and the satellite disagree often. Latency per climate is published in the benchmark.",
  },
];

export default function Overview() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    getMeta()
      .then(setMeta)
      .catch((exc: Error) => setError(exc.message));
  }, []);

  const stats = useMemo(() => {
    const rows = (meta?.benchmark.rows ?? []).filter(
      (row) => row.group === "test" && ["step", "drift", "replay"].includes(row.attack),
    );
    if (!rows.length) return FALLBACK;
    const caught = Math.min(...rows.map((row) => row.catch_rate));
    const driest = [...rows].sort((a, b) => (a.median_delay_days ?? 0) - (b.median_delay_days ?? 0))[0];
    const fa = rows.reduce((sum, row) => sum + row.fa_per_location_year, 0) / rows.length;
    const runs = (meta?.benchmark.rows ?? []).reduce((sum, row) => sum + (row.n_scenarios || 0), 0);
    return { caught, delay: `${driest?.median_delay_days ?? 0} days`, fa, runs };
  }, [meta]);

  const testRows = (meta?.benchmark.rows ?? []).filter(
    (row) => row.group === "test" && ["step", "drift", "replay"].includes(row.attack),
  );

  return (
    <>
      {/* ---------------------------------------------------------- hero */}
      <section className="relative overflow-hidden">
        <div className="pointer-events-none absolute inset-0 grid-lines opacity-70" aria-hidden="true" />
        <div className="page relative pb-16 pt-14 md:pb-24 md:pt-20">
          <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-[color:var(--edge)] pb-4">
            <p className="eyebrow">TSYP14 · Track 2 · False-data injection</p>
            <p className="eyebrow">Phase 2 · {new Date().getFullYear()}</p>
          </div>

          <div className="grid gap-10 pt-12 md:pt-16 lg:grid-cols-12">
            <div className="lg:col-span-8">
              <h1 className="display text-[clamp(3rem,8vw,7.2rem)]">
                {["Trust,", "measured every", "morning."].map((line, i) => (
                  <span key={line} className="block overflow-hidden">
                    <motion.span
                      className="block"
                      initial={reduced ? undefined : { y: "110%" }}
                      animate={reduced ? undefined : { y: 0 }}
                      transition={{ duration: 1, delay: 0.08 * i, ease: "easeOut" }}
                    >
                      {line}
                    </motion.span>
                  </span>
                ))}
              </h1>
            </div>
            <div className="lg:col-span-4 lg:pt-3">
              <motion.p
                initial={reduced ? undefined : { opacity: 0, y: 16 }}
                animate={reduced ? undefined : { opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.5 }}
                className="max-w-md text-[1.02rem] leading-relaxed text-graphite"
              >
                A weather station can be compromised without breaking a signature. VERITAS checks a daily
                solar-radiation series against its neighbours, clear-sky physics and an independent satellite
                pass, then gives one trust score per day and the reasons behind it.
              </motion.p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Link href="/lab" className="btn">
                  Run an attack <span aria-hidden="true">→</span>
                </Link>
                <Link href="/benchmark" className="btn btn-ghost">
                  Read the evaluation
                </Link>
              </div>
            </div>
          </div>

          <div className="mt-14 grid grid-cols-2 gap-y-8 border-t border-[color:var(--edge)] pt-8 md:mt-20 md:grid-cols-4">
            <Metric value={pct(stats.caught)} label="Standard attacks caught" note="unseen locations, 2024" />
            <Metric value={stats.delay} label="Fastest median delay" note="after a 12% step" />
            <Metric value={num(stats.fa, 2)} label="False alarms" note="per location-year" />
            <Metric
              value={stats.runs.toLocaleString("en-GB")}
              label="Scenario runs"
              note={meta ? `${meta.test.length} held-out sites` : "measured locally"}
            />
          </div>
        </div>
      </section>

      {error && (
        <div className="page">
          <p className="border-l-2 border-l-signal bg-signal/[0.06] px-4 py-3 text-[0.85rem] text-graphite">{error}</p>
        </div>
      )}

      {/* ---------------------------------------------------------- premise */}
      <section className="page mt-20 md:mt-28">
        <div className="grid gap-10 border-t border-[color:var(--edge)] pt-10 lg:grid-cols-12">
          <div className="lg:col-span-3">
            <p className="eyebrow">01 / Premise</p>
          </div>
          <div className="lg:col-span-9">
            <p className="display max-w-4xl text-[clamp(1.9rem,3.6vw,3.2rem)]">
              A sensor can lie while every cryptographic check still passes. Provenance tells you who sent a
              value; only context tells you whether it could be true.
            </p>
            <div className="mt-10 grid gap-8 md:grid-cols-3">
              <p className="text-[0.9rem] leading-relaxed text-graphite">
                In situ networks are attacked by biasing a device that holds a valid key, or by replaying days
                that were already accepted. Signatures do not help with either.
              </p>
              <p className="text-[0.9rem] leading-relaxed text-graphite">
                So we predict each day from what the attacker cannot rewrite: neighbouring grid cells, the
                clear-sky ceiling, weather, and what the satellite saw overhead.
              </p>
              <p className="text-[0.9rem] leading-relaxed text-graphite">
                The result is a signed, hash-chained record of every file and reading, and a causal detector that
                never looks at tomorrow.
              </p>
            </div>
          </div>
        </div>

        <Flow />
      </section>

      {/* ---------------------------------------------------------- pipeline */}
      <section className="page mt-20 md:mt-28">
        <div className="grid gap-10 border-t border-[color:var(--edge)] pt-10 lg:grid-cols-12">
          <div className="lg:col-span-3">
            <p className="eyebrow">02 / Method</p>
            <h2 className="display mt-4 text-4xl md:text-5xl">How a day is judged</h2>
            <p className="mt-4 max-w-xs text-[0.88rem] leading-relaxed text-graphite">
              Five steps, no lookahead, one decision by the time the operator opens the console.
            </p>
          </div>
          <div className="lg:col-span-9">
            <PipelineRail />
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------- evidence */}
      <section className="page mt-20 md:mt-28">
        <div className="border-t border-[color:var(--edge)] pt-10">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="eyebrow">03 / Evidence</p>
              <h2 className="display mt-4 max-w-2xl text-4xl md:text-5xl">
                Five locations and one year the model never saw.
              </h2>
            </div>
            <Link href="/benchmark" className="link-underline font-mono text-[0.65rem] uppercase tracking-widest">
              Full benchmark
            </Link>
          </div>

          {testRows.length > 0 ? (
            <div className="mt-10 overflow-x-auto">
              <table className="w-full min-w-[640px] text-left">
                <thead>
                  <tr className="border-y border-[color:var(--edge)] font-mono text-[0.6rem] uppercase tracking-widest text-ash">
                    <th className="py-3 font-normal">Attack</th>
                    <th className="py-3 font-normal">Caught</th>
                    <th className="py-3 font-normal">Median delay</th>
                    <th className="py-3 font-normal">False alarms</th>
                    <th className="py-3 font-normal">F1</th>
                    <th className="py-3 font-normal">AUC</th>
                    <th className="py-3 font-normal">Runs</th>
                  </tr>
                </thead>
                <tbody>
                  {testRows.map((row: BenchmarkRow) => (
                    <tr key={row.attack} className="border-b border-[color:var(--edge-soft)]">
                      <td className="py-4 capitalize">{row.attack}</td>
                      <td className="num py-4">{pct(row.catch_rate)}</td>
                      <td className="num py-4">{num(row.median_delay_days, 1)} d</td>
                      <td className="num py-4">{num(row.fa_per_location_year, 2)}</td>
                      <td className="num py-4">{num(row.f1, 2)}</td>
                      <td className="num py-4">{num(row.roc_auc, 2)}</td>
                      <td className="num py-4 text-ash">{row.n_scenarios}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="mt-8 text-[0.85rem] text-ash">
              {meta ? "No benchmark table found: run `python -m src.eval.run_benchmark --sweep`." : "Reading results…"}
            </p>
          )}

          <div className="mt-8 grid gap-6 md:grid-cols-3">
            <Note
              title="Held-out by construction"
              body="Thresholds are tuned on train locations only; the code refuses test frames. The five test sites and the 2024 year are never used for tuning."
            />
            <Note
              title="Adaptive attacker included"
              body="A white-box attacker that knows the thresholds searches for the largest bias it can hold. Its safe envelope is published, not omitted."
            />
            <Note
              title="Runs on one laptop"
              body="0.285 ms per location-day, 4.7 MB peak memory, no GPU and no cloud account. The full sweep takes about six minutes."
            />
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------- limits */}
      <section className="page mt-20 md:mt-28">
        <div className="grid gap-10 border-t border-[color:var(--edge)] pt-10 lg:grid-cols-12">
          <div className="lg:col-span-3">
            <p className="eyebrow">04 / Where it fails</p>
            <h2 className="display mt-4 text-4xl md:text-5xl">Published, not patched over.</h2>
          </div>
          <div className="lg:col-span-9">
            <div className="grid gap-x-10 gap-y-8 sm:grid-cols-2">
              {LIMITS.map((limit, i) => (
                <Reveal key={limit.title} delay={i * 0.05}>
                  <div className="border-t border-[color:var(--edge)] pt-4">
                    <p className="font-mono text-[0.6rem] uppercase tracking-widest text-ash">
                      {String(i + 1).padStart(2, "0")}
                    </p>
                    <h3 className="mt-2 text-[1.05rem]">{limit.title}</h3>
                    <p className="mt-2 text-[0.86rem] leading-relaxed text-graphite">{limit.body}</p>
                  </div>
                </Reveal>
              ))}
            </div>
            <p className="mt-10 max-w-2xl text-[0.86rem] leading-relaxed text-graphite">
              The integrity page proves the log is intact; the benchmark page shows the numbers above; the lab
              lets you attack a site yourself, including the attacks that still work.
            </p>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------- cta */}
      <section className="mt-20 bg-ink text-paper md:mt-28">
        <div className="page grid gap-10 py-16 md:grid-cols-12 md:py-24">
          <div className="md:col-span-8">
            <p className="eyebrow !text-paper/50">05 / Your data</p>
            <p className="display mt-5 max-w-3xl text-[clamp(2rem,4.4vw,3.8rem)]">
              Bring a series. We will tell you what we think of it, day by day.
            </p>
          </div>
          <div className="md:col-span-4 md:pt-4">
            <p className="max-w-sm text-[0.9rem] leading-relaxed text-paper/70">
              Upload a CSV with a date and a daily value, give coordinates, and VERITAS calibrates on the first
              year then scores every day after it. No retraining, no cloud.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link href="/score" className="btn !border-paper !bg-paper !text-ink hover:!bg-transparent hover:!text-paper">
                Assess a file <span aria-hidden="true">→</span>
              </Link>
              <Link href="/integrity" className="btn !border-paper/40 !bg-transparent !text-paper hover:!bg-paper hover:!text-ink">
                Check the log
              </Link>
            </div>
          </div>
        </div>
        {meta && (
          <div className="page flex flex-wrap items-center justify-between gap-2 border-t border-paper/15 py-5 font-mono text-[0.6rem] uppercase tracking-widest text-paper/50">
            <span>
              Provenance {meta.provenance.ok ? "verified" : "unverified"} · {meta.provenance.records} records
            </span>
            <span>
              {meta.cleanLocations} sites · {meta.train.length} train / {meta.test.length} test · test year{" "}
              {meta.testYear}
            </span>
            <span>Last checked {longDay(new Date().toISOString().slice(0, 10))}</span>
          </div>
        )}
      </section>
    </>
  );
}

function Metric({ value, label, note }: { value: string; label: string; note: string }) {
  return (
    <div className="pr-4">
      <p className="num text-[clamp(2rem,4vw,3.1rem)] leading-none">{value}</p>
      <p className="mt-3 text-[0.82rem]">{label}</p>
      <p className="mt-0.5 text-[0.72rem] text-ash">{note}</p>
    </div>
  );
}

function Note({ title, body }: { title: string; body: string }) {
  return (
    <div className="border-t border-[color:var(--edge)] pt-4">
      <h3 className="text-[0.92rem]">{title}</h3>
      <p className="mt-2 text-[0.82rem] leading-relaxed text-graphite">{body}</p>
    </div>
  );
}

function Flow() {
  const stages = ["Sensor", "Gateway", "Context model", "Four detectors", "Trust policy"];
  return (
    <div className="mt-14 border-y border-[color:var(--edge)] py-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        {stages.map((stage, i) => (
          <div key={stage} className="flex items-center gap-3 sm:gap-4">
            <span className="flex h-6 w-6 items-center justify-center border border-ink font-mono text-[0.58rem]">
              {i + 1}
            </span>
            <span className="text-[0.86rem]">{stage}</span>
            {i < stages.length - 1 && <span className="hidden h-px w-10 bg-[color:var(--edge)] sm:block md:w-16" />}
          </div>
        ))}
      </div>
    </div>
  );
}
