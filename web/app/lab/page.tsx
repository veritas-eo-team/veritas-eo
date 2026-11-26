"use client";

import { useEffect, useState } from "react";

import PageHeader from "@/components/PageHeader";
import RunReport from "@/components/RunReport";
import ScenarioControls, { defaults, type ScenarioValues } from "@/components/ScenarioControls";
import { getMeta, runScenario } from "@/lib/api";
import type { Meta, RunPayload } from "@/lib/types";

export default function LabPage() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [values, setValues] = useState<ScenarioValues>(defaults(null));
  const [randomStart, setRandomStart] = useState(false);
  const [run, setRun] = useState<RunPayload | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMeta()
      .then((data) => {
        setMeta(data);
        setValues((current) => ({ ...current, location: data.test[0] ?? data.train[0] ?? current.location }));
      })
      .catch((exc: Error) => setError(exc.message));
  }, []);

  const start = async () => {
    setBusy(true);
    setError(null);
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
      setRun(payload);
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader
        index="02"
        section="Attack lab"
        title="Inject an attack, watch it be judged."
        description="Choose a site, an attack and a size. The detector sees only the series: it is never told whether an attack exists, when it starts, or what kind it is. Every flagged day can be opened and explained."
        aside={<span className="font-mono text-[0.6rem] uppercase tracking-widest text-ash">Detector is blind to the labels</span>}
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
              randomStart={randomStart}
              onRandomStart={setRandomStart}
            />
            {meta && (
              <p className="mt-4 font-mono text-[0.6rem] uppercase leading-relaxed tracking-wider text-ash">
                {meta.cleanLocations} sites on disk · context model {meta.model ? "loaded" : "missing"} · thresholds{" "}
                {meta.tuned ? "tuned" : "defaults"} · test year {meta.testYear}
              </p>
            )}
          </div>
        </aside>

        <div className="lg:col-span-8">
          {error && (
            <p className="border-l-2 border-l-signal bg-signal/[0.06] px-4 py-3 text-[0.85rem] leading-relaxed text-graphite">
              {error}
            </p>
          )}

          {!run && !busy && (
            <EmptyState />
          )}

          {busy && (
            <div className="panel flex min-h-[420px] items-center justify-center">
              <div className="text-center">
                <p className="font-mono text-[0.62rem] uppercase tracking-widest text-ash">Running the pipeline</p>
                <p className="display mt-3 text-3xl">Context, detectors, fusion…</p>
                <div className="tick mx-auto mt-6 w-40 opacity-40" />
              </div>
            </div>
          )}

          {run && !busy && <RunReport run={run} />}
        </div>
      </div>
    </>
  );
}

function EmptyState() {
  return (
    <div className="panel flex min-h-[420px] flex-col justify-between p-8">
      <div>
        <p className="eyebrow">Nothing running</p>
        <p className="display mt-4 max-w-md text-4xl">
          A 12% step on a held-out site is a good place to start.
        </p>
        <p className="mt-4 max-w-md text-[0.88rem] leading-relaxed text-graphite">
          The default scenario hides a +12% bias on one of the five test locations from May first. Press run and
          you will see how many days it takes to notice, what it blames, and why.
        </p>
      </div>
      <div className="mt-10 grid gap-4 border-t border-[color:var(--edge)] pt-6 sm:grid-cols-3">
        {[
          ["Step", "constant bias from one day"],
          ["Drift", "a ramp under the threshold"],
          ["Replay", "days copied from earlier"],
        ].map(([name, note]) => (
          <div key={name}>
            <p className="font-mono text-[0.6rem] uppercase tracking-widest text-ash">Attack</p>
            <p className="mt-1 text-[0.95rem]">{name}</p>
            <p className="text-[0.75rem] text-ash">{note}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
