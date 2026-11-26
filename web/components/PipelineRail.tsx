"use client";

import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useEffect, useRef } from "react";

const STEPS = [
  {
    index: "01",
    title: "Signed at the sensor",
    body: "Every reading carries a device key, a sequence number, a timestamp and a nonce. The gateway rejects replays and forged frames before any analysis runs.",
    meta: "Ed25519 · 300 s window · 0.089 ms per check",
  },
  {
    index: "02",
    title: "Expected from context",
    body: "Four neighbouring grid cells, clear-sky radiation, temperature, humidity and season predict what this site should have measured today. A ridge model, calibrated per site.",
    meta: "gap = log(observed) − log(expected) · 0.285 ms per day",
  },
  {
    index: "03",
    title: "Four detectors, causal",
    body: "Step (CUSUM), drift (EWMA with a slope test), replay or stuck values, and physical contradictions. Each one sees only days up to today.",
    meta: "no lookahead · NaN never raises an alarm",
  },
  {
    index: "04",
    title: "Fused into trust",
    body: "Detector scores become a compromise probability; the shape of the gap decides whether this is a step, a drift or a copy.",
    meta: "trust = 1 − p(compromise)",
  },
  {
    index: "05",
    title: "One action per day",
    body: "Accept, de-weight, verify or quarantine. Every decision arrives with the evidence that produced it, in plain language.",
    meta: "accept 0.8 · de-weight 0.5 · verify 0.2",
  },
];

export default function PipelineRail() {
  const root = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    gsap.registerPlugin(ScrollTrigger);
    const context = gsap.context(() => {
      const line = root.current?.querySelector("[data-rail-line]");
      if (line) {
        gsap.fromTo(
          line,
          { scaleY: 0 },
          {
            scaleY: 1,
            transformOrigin: "top center",
            ease: "none",
            scrollTrigger: {
              trigger: root.current,
              start: "top 70%",
              end: "bottom 60%",
              scrub: 0.6,
            },
          },
        );
      }
      gsap.utils.toArray<HTMLElement>("[data-rail-step]").forEach((step) => {
        gsap.from(step, {
          opacity: 0,
          y: 26,
          duration: 0.8,
          ease: "power2.out",
          scrollTrigger: { trigger: step, start: "top 88%", once: true },
        });
      });
    }, root);
    return () => context.revert();
  }, []);

  return (
    <div ref={root} className="relative">
      <div className="absolute left-[7px] top-2 hidden h-[calc(100%-2rem)] w-px bg-[color:var(--edge)] md:block">
        <div data-rail-line className="h-full w-px bg-ink" />
      </div>
      <ol className="space-y-10 md:space-y-14">
        {STEPS.map((step) => (
          <li key={step.index} data-rail-step className="relative grid gap-3 md:grid-cols-12 md:gap-8 md:pl-12">
            <span className="absolute left-0 top-2 hidden h-[15px] w-[15px] rounded-full border border-ink bg-paper md:block" />
            <div className="md:col-span-4">
              <p className="eyebrow">{step.index}</p>
              <h3 className="display mt-2 text-3xl md:text-[2.1rem]">{step.title}</h3>
            </div>
            <div className="md:col-span-6">
              <p className="max-w-xl text-[0.95rem] leading-relaxed text-graphite">{step.body}</p>
            </div>
            <div className="md:col-span-2">
              <p className="font-mono text-[0.6rem] uppercase leading-relaxed tracking-wider text-ash">{step.meta}</p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
