"use client";

import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";

import { pct, signedPct } from "@/lib/format";
import { ACTION_COLOR, ACTION_LABEL, type Action, type SeriesPayload } from "@/lib/types";

interface Props {
  series: SeriesPayload;
  labels: { start: string | null; end: string | null };
  highlight?: string | null;
  onPick?: (date: string) => void;
  showControls?: boolean;
}

const PAD = { left: 52, right: 16, top: 10, bottom: 24 };
const PANEL = { values: 190, gap: 74, scores: 96, trust: 64 };
const GAP = 16;

const COLORS = {
  observed: "#0B0B0C",
  expected: "#8A8B8E",
  clearSky: "#A98B3E",
  gap: "#3A3B3E",
  threshold: "#C8452A",
  grid: "rgba(11,11,12,0.09)",
  scores: {
    step: "#0B0B0C",
    drift: "#3F5F4B",
    replay: "#6B6C70",
    consistency: "#C07A2A",
  } as Record<string, string>,
};

type Range = "full" | "attack" | "ninety";

export default function SeriesChart({ series, labels, highlight, onPick, showControls = true }: Props) {
  const wrap = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(920);
  const [range, setRange] = useState<Range>("full");
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);

  useEffect(() => {
    const node = wrap.current;
    if (!node) return;
    const observer = new ResizeObserver((entries) => {
      const next = entries[0]?.contentRect.width;
      if (next) setWidth(Math.max(320, next));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const [from, to] = useMemo<[number, number]>(() => {
    const dates = series.dates;
    if (!dates.length) return [0, 0];
    if (range === "ninety" || !labels.start) {
      return range === "ninety" ? [Math.max(0, dates.length - 90), dates.length - 1] : [0, dates.length - 1];
    }
    const start = dates.indexOf(labels.start);
    const end = labels.end ? dates.indexOf(labels.end) : start;
    if (start < 0) return [0, dates.length - 1];
    const left = Math.max(0, start - 30);
    const right = Math.min(dates.length - 1, Math.max(end, start) + 30);
    return [left, right];
  }, [series.dates, labels.start, labels.end, range]);

  const count = Math.max(1, to - from + 1);
  const innerW = Math.max(160, width - PAD.left - PAD.right);
  const x = (i: number) => PAD.left + ((i - from) / Math.max(1, count - 1)) * innerW;

  const panels = useMemo(() => {
    const valuesTop = PAD.top;
    const gapTop = valuesTop + PANEL.values + GAP;
    const scoresTop = gapTop + PANEL.gap + GAP;
    const trustTop = scoresTop + PANEL.scores + GAP;
    return {
      values: valuesTop,
      gap: gapTop,
      scores: scoresTop,
      trust: trustTop,
      height: trustTop + PANEL.trust + PAD.bottom,
    };
  }, []);

  const slice = <T,>(arr: T[]): T[] => arr.slice(from, to + 1);

  const valueScale = useMemo(() => {
    const pool: number[] = [];
    for (const key of [series.observed, series.expected, series.clearSky] as (number | null)[][]) {
      for (const v of slice(key)) if (v !== null && Number.isFinite(v)) pool.push(v);
    }
    if (!pool.length) return { min: 0, max: 1 };
    const min = Math.min(...pool);
    const max = Math.max(...pool);
    const pad = Math.max(0.4, (max - min) * 0.08);
    return { min: Math.max(0, min - pad), max: max + pad };
  }, [series, from, to]);

  const scoreMax = useMemo(() => {
    let top = 1.4;
    for (const key of Object.values(series.scores)) {
      for (const v of slice(key)) if (v !== null && v > top) top = v;
    }
    return Math.min(3, top);
  }, [series.scores, from, to]);

  const gapMax = useMemo(() => {
    let top = 5;
    for (const v of slice(series.gapPct)) if (v !== null && Math.abs(v) > top) top = Math.abs(v);
    return Math.min(120, top * 1.1);
  }, [series.gapPct, from, to]);

  const yValues = (v: number) =>
    panels.values + PANEL.values - ((v - valueScale.min) / Math.max(1e-6, valueScale.max - valueScale.min)) * PANEL.values;
  const yGap = (v: number) => panels.gap + PANEL.gap / 2 - (v / gapMax) * (PANEL.gap / 2);
  const yScore = (v: number) => panels.scores + PANEL.scores - (Math.min(v, scoreMax) / scoreMax) * PANEL.scores;
  const yTrust = (v: number) => panels.trust + PANEL.trust - v * PANEL.trust;

  const path = (values: (number | null)[], scaleY: (v: number) => number): string => {
    let out = "";
    let open = false;
    for (let i = from; i <= to; i += 1) {
      const value = values[i];
      if (value === null || value === undefined || !Number.isFinite(value)) {
        open = false;
        continue;
      }
      out += `${open ? "L" : "M"}${x(i).toFixed(2)},${scaleY(value).toFixed(2)}`;
      open = true;
    }
    return out;
  };

  const tickIdx = useMemo(() => {
    const total = count;
    const step = Math.max(1, Math.floor(total / 5));
    const ticks: number[] = [];
    for (let i = from; i <= to; i += step) ticks.push(i);
    if (ticks[ticks.length - 1] !== to) ticks.push(to);
    return ticks;
  }, [from, to, count]);

  const band = useMemo(() => {
    if (!labels.start) return null;
    const startIdx = series.dates.indexOf(labels.start);
    const endIdx = labels.end ? series.dates.indexOf(labels.end) : startIdx;
    if (startIdx < 0) return null;
    const left = Math.max(x(Math.max(from, startIdx)), PAD.left);
    const right = Math.min(x(Math.min(to, endIdx)), width - PAD.right);
    return right >= left ? { left, right: Math.max(right, left + 1) } : null;
  }, [labels, series.dates, from, to, width]);

  const hoverRow = hover ? hover.i : null;

  const onMove = (event: MouseEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const relative = event.clientX - box.left;
    const ratio = (relative - PAD.left) / innerW;
    const index = Math.round(from + ratio * (count - 1));
    const clamped = Math.min(to, Math.max(from, index));
    setHover({ i: clamped, x: relative, y: event.clientY - box.top });
  };

  const date = hoverRow !== null ? series.dates[hoverRow] : null;

  return (
    <div className="relative" ref={wrap}>
      {showControls && (
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <Legend />
          <div className="flex gap-1">
            {(
              [
                ["full", "Full run"],
                ["attack", "Attack window"],
                ["ninety", "Last 90 days"],
              ] as [Range, string][]
            ).map(([key, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => setRange(key)}
                className={`border px-3 py-1.5 font-mono text-[0.6rem] uppercase tracking-wider transition-colors ${
                  range === key
                    ? "border-ink bg-ink text-paper"
                    : "border-[color:var(--edge)] text-graphite hover:border-ink hover:text-ink"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

      <svg
        width="100%"
        height={panels.height}
        viewBox={`0 0 ${width} ${panels.height}`}
        role="img"
        aria-label="Observed values, gap, detector scores and trust over time"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        onClick={() => {
          if (onPick && date) onPick(date);
        }}
        className={onPick ? "cursor-crosshair" : undefined}
      >
        {band && (
          <rect
            x={band.left}
            y={PAD.top}
            width={band.right - band.left}
            height={panels.trust + PANEL.trust - PAD.top}
            fill="#C8452A"
            opacity={0.07}
          />
        )}

        <line x1={PAD.left} x2={width - PAD.right} y1={panels.gap + PANEL.gap / 2} y2={panels.gap + PANEL.gap / 2} stroke={COLORS.grid} />

        <path d={path(series.expected, yValues)} fill="none" stroke={COLORS.expected} strokeWidth={1} strokeDasharray="3 3" />
        <path d={path(series.clearSky, yValues)} fill="none" stroke={COLORS.clearSky} strokeWidth={0.9} strokeDasharray="1 3" opacity={0.8} />
        <path d={path(series.observed, yValues)} fill="none" stroke={COLORS.observed} strokeWidth={1.15} />

        <path d={path(series.gapPct, yGap)} fill="none" stroke={COLORS.gap} strokeWidth={1} />

        <line
          x1={PAD.left}
          x2={width - PAD.right}
          y1={yScore(1)}
          y2={yScore(1)}
          stroke={COLORS.threshold}
          strokeWidth={0.9}
          strokeDasharray="4 3"
        />
        {(["step", "drift", "replay", "consistency"] as const).map((key) => (
          <path
            key={key}
            d={path(series.scores[key], yScore)}
            fill="none"
            stroke={COLORS.scores[key]}
            strokeWidth={1.1}
            opacity={0.9}
          />
        ))}

        {series.trust.map((value, i) => {
          if (i < from || i > to || value === null) return null;
          const action = (series.action[i] || "accept") as Action;
          const barWidth = Math.max(0.7, innerW / count - 0.4);
          const top = yTrust(value);
          return (
            <rect
              key={i}
              x={x(i) - barWidth / 2}
              y={top}
              width={barWidth}
              height={Math.max(0, panels.trust + PANEL.trust - top)}
              fill={ACTION_COLOR[action] ?? "#6B6C70"}
              opacity={0.9}
            />
          );
        })}

        {[0, 0.5, 1].map((level) => (
          <line
            key={level}
            x1={PAD.left}
            x2={width - PAD.right}
            y1={yTrust(level)}
            y2={yTrust(level)}
            stroke={COLORS.grid}
            strokeWidth={0.7}
          />
        ))}

        {[
          { top: panels.values, label: `${valueScale.max.toFixed(1)}` },
          { top: panels.values + PANEL.values, label: `${valueScale.min.toFixed(1)}` },
        ].map((tick) => (
          <text key={tick.label + tick.top} x={PAD.left - 8} y={tick.top + 4} textAnchor="end" className="fill-ash font-mono text-[9px]">
            {tick.label}
          </text>
        ))}
        <text x={PAD.left - 8} y={panels.gap + PANEL.gap / 2 + 3} textAnchor="end" className="fill-ash font-mono text-[9px]">
          0%
        </text>
        <text x={PAD.left - 8} y={panels.gap + 10} textAnchor="end" className="fill-ash font-mono text-[9px]">
          {`+${gapMax.toFixed(0)}`}
        </text>
        <text x={PAD.left - 8} y={panels.gap + PANEL.gap - 2} textAnchor="end" className="fill-ash font-mono text-[9px]">
          {`-${gapMax.toFixed(0)}`}
        </text>
        <text x={PAD.left - 8} y={panels.scores + 10} textAnchor="end" className="fill-ash font-mono text-[9px]">
          1.0
        </text>
        <text x={PAD.left - 8} y={panels.trust + 9} textAnchor="end" className="fill-ash font-mono text-[9px]">
          1
        </text>
        <text x={PAD.left - 8} y={panels.trust + PANEL.trust} textAnchor="end" className="fill-ash font-mono text-[9px]">
          0
        </text>

        {tickIdx.map((index) => (
          <text
            key={index}
            x={x(index)}
            y={panels.height - 6}
            textAnchor={index === from ? "start" : index === to ? "end" : "middle"}
            className="fill-ash font-mono text-[9px]"
          >
            {series.dates[index]?.slice(0, 7)}
          </text>
        ))}

        {highlight && series.dates.includes(highlight) && (
          <line
            x1={x(series.dates.indexOf(highlight))}
            x2={x(series.dates.indexOf(highlight))}
            y1={PAD.top}
            y2={panels.trust + PANEL.trust}
            stroke="#0B0B0C"
            strokeWidth={1}
            strokeDasharray="2 2"
          />
        )}

        {hoverRow !== null && hoverRow >= from && hoverRow <= to && (
          <g pointerEvents="none">
            <line x1={x(hoverRow)} x2={x(hoverRow)} y1={PAD.top} y2={panels.trust + PANEL.trust} stroke="#0B0B0C" strokeWidth={0.7} opacity={0.5} />
            {series.observed[hoverRow] !== null && (
              <circle cx={x(hoverRow)} cy={yValues(series.observed[hoverRow] as number)} r={2.4} fill={COLORS.observed} />
            )}
            {series.expected[hoverRow] !== null && (
              <circle cx={x(hoverRow)} cy={yValues(series.expected[hoverRow] as number)} r={2} fill={COLORS.expected} />
            )}
          </g>
        )}
      </svg>

      {hover && date && (
        <div
          className="pointer-events-none absolute top-0 z-10 hidden min-w-[196px] border border-ink bg-bone p-3 sm:block"
          style={{
            left: Math.min(Math.max(hover.x + 16, 8), width - 210),
            top: Math.max(8, hover.y - 60),
          }}
        >
          <p className="font-mono text-[0.62rem] uppercase tracking-widest text-ash">{date}</p>
          <dl className="mt-2 space-y-1 text-[0.72rem]">
            <Row label="observed" value={`${series.observed[hover.i]?.toFixed(2) ?? "—"} MJ/m²/day`} />
            <Row label="expected" value={`${series.expected[hover.i]?.toFixed(2) ?? "—"}`} />
            <Row label="gap" value={signedPct(series.gapPct[hover.i])} />
            <Row label="trust" value={pct(series.trust[hover.i], 0)} />
            <Row
              label="action"
              value={ACTION_LABEL[(series.action[hover.i] || "accept") as Action]}
              color={ACTION_COLOR[(series.action[hover.i] || "accept") as Action]}
            />
          </dl>
        </div>
      )}
    </div>
  );
}

function Row({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <dt className="text-ash">{label}</dt>
      <dd className="num" style={color ? { color } : undefined}>
        {value}
      </dd>
    </div>
  );
}

function Legend() {
  const items: [string, string, string?][] = [
    ["observed", COLORS.observed],
    ["expected", COLORS.expected, "3 3"],
    ["clear-sky limit", COLORS.clearSky, "1 3"],
    ["step", COLORS.scores.step!],
    ["drift", COLORS.scores.drift!],
    ["replay", COLORS.scores.replay!],
    ["physics", COLORS.scores.consistency!],
  ];
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
      {items.map(([label, color, dash]) => (
        <span key={label} className="flex items-center gap-2 font-mono text-[0.58rem] uppercase tracking-wider text-graphite">
          <svg width="14" height="6" aria-hidden="true">
            <line x1="0" y1="3" x2="14" y2="3" stroke={color} strokeWidth="1.4" strokeDasharray={dash} />
          </svg>
          {label}
        </span>
      ))}
    </div>
  );
}
