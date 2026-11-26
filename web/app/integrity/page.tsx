"use client";

import { useEffect, useState } from "react";

import PageHeader from "@/components/PageHeader";
import Reveal from "@/components/Reveal";
import { getIntegrity, postIntegrity, type IntegrityAction } from "@/lib/api";
import type { IntegrityPayload } from "@/lib/types";

interface VerifyResult {
  ok: boolean;
  records: number;
  reason: string;
  firstBad: number | null;
  head: string;
}
interface TamperResult {
  caught: boolean;
  reason: string;
  firstBad: number | null;
}
interface GatewayResult {
  cases: { case: string; expected: string; accepted: boolean; reason: string }[];
}
interface OverheadResult {
  rows: { operation: string; value: string }[];
}
type Result =
  | { action: "verify"; data: VerifyResult }
  | { action: "tamper"; data: TamperResult }
  | { action: "gateway"; data: GatewayResult }
  | { action: "overhead"; data: OverheadResult };

export default function IntegrityPage() {
  const [log, setLog] = useState<IntegrityPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<IntegrityAction | null>(null);
  const [result, setResult] = useState<Result | null>(null);

  useEffect(() => {
    getIntegrity()
      .then(setLog)
      .catch((exc: Error) => setError(exc.message));
  }, []);

  const run = async (action: IntegrityAction) => {
    setBusy(action);
    setError(null);
    try {
      if (action === "verify") setResult({ action, data: await postIntegrity<VerifyResult>(action) });
      if (action === "tamper") setResult({ action, data: await postIntegrity<TamperResult>(action) });
      if (action === "gateway") setResult({ action, data: await postIntegrity<GatewayResult>(action) });
      if (action === "overhead") setResult({ action, data: await postIntegrity<OverheadResult>(action) });
      if (action !== "gateway" && action !== "overhead") {
        setLog(await getIntegrity());
      }
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <>
      <PageHeader
        index="04"
        section="Integrity"
        title="Proof that nothing moved."
        description="Every downloaded and derived file is hashed, chained to the record before it, and signed. Editing, deleting or reordering a record breaks the chain, and the check is reproducible on a fresh copy of the repository."
        aside={<span className="font-mono text-[0.6rem] uppercase tracking-widest text-ash">Ed25519 · SHA-256 chain</span>}
      />

      <div className="page grid gap-10 pb-10 lg:grid-cols-12">
        <div className="space-y-10 lg:col-span-8">
          <section>
            <div className="flex flex-wrap items-end justify-between gap-3 border-b border-[color:var(--edge)] pb-3">
              <p className="eyebrow">Provenance log</p>
              {log && (
                <p className="font-mono text-[0.6rem] uppercase tracking-wider text-ash">
                  {log.records.length} records · key {log.keys.join(", ") || "none"}
                </p>
              )}
            </div>

            {error && <p className="mt-4 text-[0.85rem] text-signal">{error}</p>}

            {!log && !error && <p className="mt-4 text-[0.85rem] text-ash">Reading the log…</p>}

            {log && log.records.length === 0 && (
              <p className="mt-4 text-[0.85rem] text-graphite">
                No provenance records yet. {log.error ? `(${log.error})` : ""}
              </p>
            )}

            {log && log.records.length > 0 && (
              <div className="mt-4 max-h-[520px] overflow-auto border border-[color:var(--edge)]">
                <table className="w-full min-w-[720px] text-left text-[0.74rem]">
                  <thead className="sticky top-0 bg-bone">
                    <tr className="border-b border-[color:var(--edge)] font-mono text-[0.56rem] uppercase tracking-widest text-ash">
                      <th className="px-3 py-2 font-normal">Seq</th>
                      <th className="px-3 py-2 font-normal">Kind</th>
                      <th className="px-3 py-2 font-normal">Path</th>
                      <th className="px-3 py-2 font-normal">Created (UTC)</th>
                      <th className="px-3 py-2 font-normal">File hash</th>
                      <th className="px-3 py-2 font-normal">Record hash</th>
                      <th className="px-3 py-2 font-normal">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {log.records.map((record) => (
                      <tr key={record.seq} className="border-b border-[color:var(--edge-soft)] last:border-0">
                        <td className="num px-3 py-1.5 text-ash">{record.seq}</td>
                        <td className="px-3 py-1.5">{record.kind}</td>
                        <td className="max-w-[220px] truncate px-3 py-1.5 text-graphite" title={record.path}>
                          {record.path}
                        </td>
                        <td className="num px-3 py-1.5 text-ash">{record.created?.slice(0, 19)}</td>
                        <td className="num px-3 py-1.5 text-ash">{record.fileHash}</td>
                        <td className="num px-3 py-1.5 text-ash">{record.recordHash}</td>
                        <td className="px-3 py-1.5">
                          <span
                            className="font-mono text-[0.58rem] uppercase tracking-wider"
                            style={{ color: record.ok ? "#3F5F4B" : "#C8452A" }}
                          >
                            {record.ok ? "valid" : "broken"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <p className="mt-3 text-[0.75rem] leading-relaxed text-ash">
              The private signing key never leaves the log machine; only the public key is committed. Verification
              below re-hashes every file on disk and walks the chain.
            </p>
          </section>

          <section>
            <p className="eyebrow border-b border-[color:var(--edge)] pb-3">Reading level</p>
            <p className="mt-4 max-w-2xl text-[0.9rem] leading-relaxed text-graphite">
              Signatures protect files in transit and at rest, but a sensor that holds a valid key can still send
              biased values. The gateway checks device, signature, timestamp window, sequence number and nonce,
              then passes the reading to the detectors. Both halves are shown here.
            </p>
          </section>
        </div>

        <aside className="lg:col-span-4">
          <div className="panel lg:sticky lg:top-24">
            <div className="border-b border-[color:var(--edge)] px-5 py-4">
              <p className="eyebrow">Run a check</p>
              {log && (
                <p className="mt-2 text-[0.78rem] leading-relaxed text-graphite">
                  Chain currently{" "}
                  <span style={{ color: log.verified.ok ? "#3F5F4B" : "#C8452A" }}>
                    {log.verified.ok ? "intact" : "failing"}
                  </span>
                  {log.verified.head ? ` · head ${log.verified.head}` : ""}
                </p>
              )}
            </div>
            <div className="grid grid-cols-2 gap-px bg-[color:var(--edge-soft)]">
              {(
                [
                  ["verify", "Verify chain"],
                  ["tamper", "Tamper demo"],
                  ["gateway", "Gateway cases"],
                  ["overhead", "Measure cost"],
                ] as [IntegrityAction, string][]
              ).map(([action, label]) => (
                <button
                  key={action}
                  type="button"
                  onClick={() => run(action)}
                  disabled={busy !== null}
                  className="bg-bone px-4 py-4 text-left text-[0.8rem] transition-colors hover:bg-ink hover:text-paper disabled:opacity-50"
                >
                  {busy === action ? "Working…" : label}
                </button>
              ))}
            </div>
            <div className="min-h-[220px] border-t border-[color:var(--edge)] p-5">
              {!result && <p className="text-[0.8rem] leading-relaxed text-ash">Pick a check to see its output.</p>}
              {result?.action === "verify" && (
                <div>
                  <p className="text-[1.05rem]" style={{ color: result.data.ok ? "#3F5F4B" : "#C8452A" }}>
                    {result.data.ok ? "Chain, hashes and signatures OK" : "Verification failed"}
                  </p>
                  <p className="mt-2 text-[0.8rem] leading-relaxed text-graphite">
                    {result.data.ok
                      ? `${result.data.records} records verified. Head ${result.data.head?.slice(0, 24)}…`
                      : `Record ${result.data.firstBad}: ${result.data.reason}`}
                  </p>
                </div>
              )}
              {result?.action === "tamper" && (
                <div>
                  <p className="text-[1.05rem]" style={{ color: result.data.caught ? "#C8452A" : "#3F5F4B" }}>
                    {result.data.caught ? "Tampering detected" : "Tampering was not caught"}
                  </p>
                  <p className="mt-2 text-[0.8rem] leading-relaxed text-graphite">
                    {result.data.caught
                      ? `Broken at record ${result.data.firstBad}: ${result.data.reason}`
                      : result.data.reason}
                  </p>
                  <p className="mt-3 text-[0.72rem] leading-relaxed text-ash">
                    The edit is made on a temporary copy; the real data and log stay untouched.
                  </p>
                </div>
              )}
              {result?.action === "gateway" && (
                <div>
                  <p className="mb-2 text-[0.95rem]">Gateway cases</p>
                  <ul className="space-y-1.5">
                    {result.data.cases.map((item) => (
                      <li key={item.case} className="text-[0.74rem] leading-snug">
                        <span
                          className="font-mono text-[0.58rem] uppercase tracking-wider"
                          style={{ color: item.accepted ? "#3F5F4B" : "#C8452A" }}
                        >
                          {item.accepted ? "accept" : "reject"}
                        </span>{" "}
                        {item.case}
                      </li>
                    ))}
                  </ul>
                  <p className="mt-3 text-[0.72rem] leading-relaxed text-ash">
                    A compromised sensor with a valid key passes every cryptographic check. Only the statistical
                    detectors can catch it.
                  </p>
                </div>
              )}
              {result?.action === "overhead" && (
                <div>
                  <p className="mb-2 text-[0.95rem]">Security overhead</p>
                  <ul className="space-y-1">
                    {result.data.rows.map((row) => (
                      <li key={row.operation} className="flex items-baseline justify-between gap-3 text-[0.72rem]">
                        <span className="text-graphite">{row.operation}</span>
                        <span className="num whitespace-nowrap">{row.value}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>

          <Reveal className="mt-6" delay={0.05}>
            <p className="font-mono text-[0.6rem] uppercase leading-relaxed tracking-wider text-ash">
              Signed readings are appended to a second chained log at the gateway, so an operator can prove what was
              accepted and when.
            </p>
          </Reveal>
        </aside>
      </div>
    </>
  );
}
