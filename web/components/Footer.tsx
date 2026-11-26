import Link from "next/link";

export default function Footer() {
  return (
    <footer className="mt-24 border-t border-[color:var(--edge)]">
      <div className="page grid gap-10 py-14 md:grid-cols-12">
        <div className="md:col-span-5">
          <p className="display text-3xl">VERITAS</p>
          <p className="mt-3 max-w-xs text-sm leading-relaxed text-graphite">
            A weather-data trust monitor for a daily solar-radiation series. Built on NASA POWER and
            Sentinel-2, runs on one laptop.
          </p>
        </div>
        <div className="md:col-span-3">
          <p className="eyebrow mb-4">Console</p>
          <ul className="space-y-2 text-sm text-graphite">
            <li><Link className="link-underline" href="/lab">Attack lab</Link></li>
            <li><Link className="link-underline" href="/replay">Live replay</Link></li>
            <li><Link className="link-underline" href="/integrity">Integrity</Link></li>
            <li><Link className="link-underline" href="/benchmark">Benchmark</Link></li>
            <li><Link className="link-underline" href="/score">Assess your data</Link></li>
          </ul>
        </div>
        <div className="md:col-span-4">
          <p className="eyebrow mb-4">Reference</p>
          <ul className="space-y-2 text-sm text-graphite">
            <li>TSYP14 · Ground Truth Under Attack</li>
            <li>Track 2 · False-data injection</li>
            <li>Standard attacks: step, drift, replay</li>
            <li>Local execution, no cloud dependency</li>
          </ul>
        </div>
      </div>
      <div className="page flex flex-col gap-2 border-t border-[color:var(--edge-soft)] py-6 pb-10 font-mono text-[0.62rem] uppercase tracking-widest text-ash sm:flex-row sm:items-center sm:justify-between">
        <span>Detector thresholds tuned on train locations only</span>
        <span>Every alarm carries its evidence</span>
      </div>
    </footer>
  );
}
