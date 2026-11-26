"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import PageHeader from "@/components/PageHeader";
import Reveal from "@/components/Reveal";
import { getMeta } from "@/lib/api";
import { num, pct } from "@/lib/format";
import type { BenchmarkRow, Meta } from "@/lib/types";

const GROUPS: [string, string, string][] = [
  ["test", "Unseen test locations · 2024", "Five sites never used for training or tuning."],
  ["unseen_year", "Unseen year · train locations", "The same sites, a year the model never saw."],
];

export default function BenchmarkPage() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMeta()
      .then(setMeta)
      .catch((exc: Error) => setError(exc.message));
  }, []);

  const rows = meta?.benchmark.rows ?? [];
  const totals = useMemo(() => {
    const main = rows.filter((row) => row.group === "test");
    if (!main.length) return null;
    return {
      runs: main.reduce((sum, row) => sum + row.n_scenarios, 0),
      best: Math.max(...main.map((row) => row.f1)),
      fa: Math.max(...main.map((row) => row.fa_per_location_year)),
    };
  }, [rows]);

  return (
    <>
      <PageHeader
        index="05"
        section="Benchmark"
        title="What was measured, and where it stops."
        description="Standard attacks over a sweep of sizes, the adaptive attacker, an unseen year and leave-one-region-out. Thresholds are tuned on train locations only; the code refuses test frames."
        aside={
          <span className="font-mono text-[0.6rem] uppercase tracking-widest text-ash">
            {totals ? `${totals.runs} runs in the table` : "results/benchmark_table.csv"}
          </span>
        }
      />

      <div className="page space-y-14 pb-10">
        {error && (
          <p className="border-l-2 border-l-signal bg-signal/[0.06] px-4 py-3 text-[0.85rem] text-graphite">{error}</p>
        )}

        {!meta && !error && <p className="text-[0.85rem] text-ash">Reading results…</p>}

        {meta && rows.length === 0 && (
          <p className="border border-[color:var(--edge)] p-6 text-[0.9rem] text-graphite">
            No results yet. Run <span className="num">python -m src.eval.run_benchmark --sweep</span> to produce them.
          </p>
        )}

        {totals && (
          <div className="grid gap-8 border-y border-[color:var(--edge)] py-8 sm:grid-cols-3">
            <div>
              <p className="num text-4xl">{totals.runs}</p>
              <p className="mt-2 text-[0.82rem]">Held-out runs in this table</p>
              <p className="text-[0.72rem] text-ash">sites × seeds × sizes</p>
            </div>
            <div>
              <p className="num text-4xl">{num(totals.best, 2)}</p>
              <p className="mt-2 text-[0.82rem]">Best F1 on a held-out attack</p>
              <p className="text-[0.72rem] text-ash">day-level, alarm = verify or quarantine</p>
            </div>
            <div>
              <p className="num text-4xl">{num(totals.fa, 2)}</p>
              <p className="mt-2 text-[0.82rem]">Worst false-alarm rate</p>
              <p className="text-[0.72rem] text-ash">per location-year across attacks</p>
            </div>
          </div>
        )}

        {GROUPS.map(([group, title, note]) => {
          const part = rows.filter((row) => row.group === group);
          if (!part.length) return null;
          return (
            <section key={group}>
              <div className="flex flex-wrap items-end justify-between gap-3 border-b border-[color:var(--edge)] pb-3">
                <h2 className="display text-3xl md:text-4xl">{title}</h2>
                <p className="text-[0.78rem] text-ash">{note}</p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[880px] text-left text-[0.82rem]">
                  <thead>
                    <tr className="border-b border-[color:var(--edge)] font-mono text-[0.58rem] uppercase tracking-widest text-ash">
                      <th className="py-2.5 font-normal">Attack</th>
                      <th className="py-2.5 font-normal">Caught</th>
                      <th className="py-2.5 font-normal">Delay</th>
                      <th className="py-2.5 font-normal">Start error</th>
                      <th className="py-2.5 font-normal">False alarms</th>
                      <th className="py-2.5 font-normal">Precision</th>
                      <th className="py-2.5 font-normal">Recall</th>
                      <th className="py-2.5 font-normal">F1</th>
                      <th className="py-2.5 font-normal">AUC</th>
                      <th className="py-2.5 font-normal">Brier</th>
                      <th className="py-2.5 font-normal">Runs</th>
                    </tr>
                  </thead>
                  <tbody>
                    {part.map((row: BenchmarkRow) => (
                      <tr key={`${group}-${row.attack}`} className="border-b border-[color:var(--edge-soft)]">
                        <td className="py-2.5 capitalize">{row.attack}</td>
                        <td className="num py-2.5">{pct(row.catch_rate)}</td>
                        <td className="num py-2.5">{num(row.median_delay_days, 1)} d</td>
                        <td className="num py-2.5">{num(row.start_error_median_days, 1)} d</td>
                        <td className="num py-2.5">{num(row.fa_per_location_year, 2)}</td>
                        <td className="num py-2.5">{num(row.precision, 2)}</td>
                        <td className="num py-2.5">{num(row.recall, 2)}</td>
                        <td className="num py-2.5">{num(row.f1, 2)}</td>
                        <td className="num py-2.5">{num(row.roc_auc, 2)}</td>
                        <td className="num py-2.5">{num(row.brier, 3)}</td>
                        <td className="num py-2.5 text-ash">{row.n_scenarios}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          );
        })}

        {meta && meta.benchmark.images.length > 0 && (
          <section>
            <div className="border-b border-[color:var(--edge)] pb-3">
              <h2 className="display text-3xl md:text-4xl">Figures</h2>
            </div>
            <div className="mt-6 grid gap-x-8 gap-y-10 md:grid-cols-2">
              {meta.benchmark.images.map((image, index) => (
                <Reveal key={image.file} delay={index * 0.04}>
                  <figure>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`/api/results/${image.file}`}
                      alt={image.title}
                      className="w-full border border-[color:var(--edge)] bg-bone"
                      loading="lazy"
                    />
                    <figcaption className="mt-2 font-mono text-[0.6rem] uppercase tracking-widest text-ash">
                      {image.title}
                    </figcaption>
                  </figure>
                </Reveal>
              ))}
            </div>
          </section>
        )}

        {meta?.benchmark.summary && (
          <section>
            <details className="border border-[color:var(--edge)]">
              <summary className="cursor-pointer list-none px-5 py-4 font-mono text-[0.62rem] uppercase tracking-widest">
                Full summary · results/summary.md
              </summary>
              <pre className="max-h-[520px] overflow-auto border-t border-[color:var(--edge)] bg-bone p-5 font-mono text-[0.72rem] leading-relaxed text-graphite">
                {meta.benchmark.summary}
              </pre>
            </details>
          </section>
        )}

        <section className="grid gap-8 border-t border-[color:var(--edge)] pt-8 md:grid-cols-3">
          {[
            [
              "How to read the delay",
              "Delay is days from the true attack start to the first verify or quarantine. A missed attack counts against catch rate, never against delay.",
            ],
            [
              "Why false alarms matter",
              "An operator who sees one false alarm per site-year still trusts the console. Above that, alarms are ignored, so the rate is a product requirement, not a footnote.",
            ],
            [
              "Reproduce it",
              "Every number above comes from results/benchmark_table.csv, produced by one command on a laptop. Nothing is hand-entered.",
            ],
          ].map(([title, body]) => (
            <div key={title} className="border-t border-[color:var(--edge)] pt-4">
              <h3 className="text-[0.92rem]">{title}</h3>
              <p className="mt-2 text-[0.82rem] leading-relaxed text-graphite">{body}</p>
            </div>
          ))}
        </section>

        <section className="flex flex-wrap items-center justify-between gap-4 border-t border-[color:var(--edge)] pt-8">
          <p className="max-w-xl text-[0.9rem] leading-relaxed text-graphite">
            The benchmark is a claim about averages. The lab is where you check it against a single site, with the
            attack you choose.
          </p>
          <Link href="/lab" className="btn">
            Open the attack lab <span aria-hidden="true">→</span>
          </Link>
        </section>
      </div>
    </>
  );
}
