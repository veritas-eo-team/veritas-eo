"use client";

import type { ReactNode } from "react";

import type { Meta } from "@/lib/types";
import { locationLabel } from "@/lib/format";

export interface ScenarioValues {
  location: string;
  attack: string;
  size: number | null;
  duration: number | null;
  noise: number | null;
  start: string | null;
  seed: number;
}

const ATTACKS = ["step", "drift", "replay", "spike", "stuck", "none"] as const;

const HINTS: Record<string, string> = {
  step: "A constant additive bias from one day onward, hidden from the detector.",
  drift: "A slow ramp the attacker hopes looks like weather.",
  replay: "Recent days are replaced by an older window of real observations.",
  spike: "A short, large excursion, then back to normal.",
  stuck: "A frozen value repeated until it looks like a broken sensor.",
  none: "No attack: a control run, used to measure false alarms.",
};

export function defaults(meta: Meta | null): ScenarioValues {
  const year = meta?.testYear ?? 2024;
  return {
    location: meta?.test[0] ?? meta?.train[0] ?? "kairouan",
    attack: "step",
    size: 12,
    duration: null,
    noise: 0,
    start: `${year}-05-01`,
    seed: 1,
  };
}

interface Props {
  meta: Meta | null;
  values: ScenarioValues;
  onChange: (next: ScenarioValues) => void;
  onRun: () => void;
  busy?: boolean;
  actionLabel?: string;
  randomStart: boolean;
  onRandomStart: (value: boolean) => void;
  children?: ReactNode;
}

export default function ScenarioControls({
  meta,
  values,
  onChange,
  onRun,
  busy = false,
  actionLabel = "Run the detector",
  randomStart,
  onRandomStart,
  children,
}: Props) {
  const year = meta?.testYear ?? 2024;
  const patch = (next: Partial<ScenarioValues>) => onChange({ ...values, ...next });

  return (
    <div className="panel">
      <div className="flex items-baseline justify-between border-b border-[color:var(--edge)] px-5 py-4">
        <p className="eyebrow">Scenario</p>
        <span className="font-mono text-[0.6rem] uppercase tracking-widest text-ash">Judge challenge mode</span>
      </div>

      <div className="space-y-5 p-5">
        <div>
          <label className="label" htmlFor="location">
            Location
          </label>
          <select
            id="location"
            className="field"
            value={values.location}
            onChange={(event) => patch({ location: event.target.value })}
          >
            {meta && meta.test.length > 0 && (
              <optgroup label="Unseen test locations">
                {meta.test.map((id) => (
                  <option key={id} value={id}>
                    {locationLabel(id)} · held out
                  </option>
                ))}
              </optgroup>
            )}
            {meta && meta.train.length > 0 && (
              <optgroup label="Train locations">
                {meta.train.map((id) => (
                  <option key={id} value={id}>
                    {locationLabel(id)}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
        </div>

        <div>
          <label className="label" htmlFor="attack">
            Attack
          </label>
          <div className="grid grid-cols-3 gap-1">
            {ATTACKS.map((attack) => (
              <button
                key={attack}
                type="button"
                onClick={() =>
                  patch({
                    attack,
                    size: attack === "replay" || attack === "stuck" || attack === "none" ? null
                      : attack === "drift" ? 15
                      : attack === "spike" ? 50
                      : 12,
                    duration: attack === "drift" ? 60 : attack === "replay" ? 7 : attack === "stuck" ? 7 : attack === "spike" ? 1 : null,
                  })
                }
                className={`border px-2 py-2 font-mono text-[0.62rem] uppercase tracking-wider transition-colors ${
                  values.attack === attack
                    ? "border-ink bg-ink text-paper"
                    : "border-[color:var(--edge)] text-graphite hover:border-ink hover:text-ink"
                }`}
                aria-pressed={values.attack === attack}
              >
                {attack}
              </button>
            ))}
          </div>
          <p className="mt-2 text-[0.78rem] leading-relaxed text-graphite">{HINTS[values.attack]}</p>
        </div>

        {values.attack === "step" && (
          <Slider
            id="size"
            label="Step size"
            value={values.size ?? 12}
            min={-30}
            max={30}
            step={1}
            format={(v) => `${v > 0 ? "+" : ""}${v}%`}
            onChange={(size) => patch({ size })}
          />
        )}
        {values.attack === "drift" && (
          <>
            <Slider
              id="size"
              label="Final drift size"
              value={values.size ?? 15}
              min={1}
              max={30}
              step={1}
              format={(v) => `+${v}%`}
              onChange={(size) => patch({ size })}
            />
            <Slider
              id="duration"
              label="Ramp duration"
              value={values.duration ?? 60}
              min={7}
              max={180}
              step={1}
              format={(v) => `${v} days`}
              onChange={(duration) => patch({ duration })}
            />
          </>
        )}
        {values.attack === "replay" && (
          <>
            <Slider
              id="duration"
              label="Replay length"
              value={values.duration ?? 7}
              min={3}
              max={21}
              step={1}
              format={(v) => `${v} days`}
              onChange={(duration) => patch({ duration })}
            />
            <Slider
              id="noise"
              label="Noise on copied values"
              value={values.noise ?? 0}
              min={0}
              max={2}
              step={0.1}
              format={(v) => `${v.toFixed(1)}%`}
              onChange={(noise) => patch({ noise })}
            />
          </>
        )}
        {values.attack === "spike" && (
          <>
            <Slider
              id="size"
              label="Spike size"
              value={values.size ?? 50}
              min={10}
              max={100}
              step={5}
              format={(v) => `+${v}%`}
              onChange={(size) => patch({ size })}
            />
            <Slider
              id="duration"
              label="Spike duration"
              value={values.duration ?? 1}
              min={1}
              max={7}
              step={1}
              format={(v) => `${v} days`}
              onChange={(duration) => patch({ duration })}
            />
          </>
        )}
        {values.attack === "stuck" && (
          <Slider
            id="duration"
            label="Stuck duration"
            value={values.duration ?? 7}
            min={3}
            max={21}
            step={1}
            format={(v) => `${v} days`}
            onChange={(duration) => patch({ duration })}
          />
        )}

        <div>
          <div className="flex items-center justify-between">
            <label className="label !mb-0" htmlFor="start">
              Attack start
            </label>
            <label className="flex cursor-pointer items-center gap-2 font-mono text-[0.6rem] uppercase tracking-wider text-graphite">
              <input
                type="checkbox"
                checked={randomStart}
                onChange={(event) => onRandomStart(event.target.checked)}
                className="h-3 w-3 accent-ink"
              />
              seeded
            </label>
          </div>
          <input
            id="start"
            type="date"
            className="field mt-2 disabled:opacity-40"
            min={`${year}-02-01`}
            max={`${year}-09-30`}
            value={values.start ?? `${year}-05-01`}
            disabled={randomStart || values.attack === "none"}
            onChange={(event) => patch({ start: event.target.value })}
          />
        </div>

        <div className="flex items-end gap-3">
          <div className="w-24">
            <label className="label" htmlFor="seed">
              Seed
            </label>
            <input
              id="seed"
              type="number"
              min={0}
              max={999}
              className="field num"
              value={values.seed}
              onChange={(event) => patch({ seed: Number(event.target.value) })}
            />
          </div>
          <p className="pb-2 text-[0.7rem] leading-snug text-ash">Same seed, same run.</p>
        </div>

        {children}

        <button type="button" className="btn w-full justify-between disabled:opacity-40" onClick={onRun} disabled={busy}>
          <span>{busy ? "Running…" : actionLabel}</span>
          <span aria-hidden="true">→</span>
        </button>
        <p className="text-[0.7rem] leading-relaxed text-ash">
          The detector is never told when, where or whether an attack happens. Only the series is passed in.
        </p>
      </div>
    </div>
  );
}

function Slider({
  id,
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (value: number) => string;
  onChange: (value: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <label className="label !mb-0" htmlFor={id}>
          {label}
        </label>
        <span className="num text-sm">{format(value)}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1 w-full cursor-pointer appearance-none bg-[color:var(--edge)] accent-ink"
        style={{ accentColor: "#0B0B0C" }}
      />
    </div>
  );
}
