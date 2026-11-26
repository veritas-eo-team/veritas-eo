"use client";

import { AnimatePresence, motion } from "framer-motion";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const LINKS = [
  { href: "/", label: "Overview", index: "01" },
  { href: "/lab", label: "Attack lab", index: "02" },
  { href: "/replay", label: "Live replay", index: "03" },
  { href: "/integrity", label: "Integrity", index: "04" },
  { href: "/benchmark", label: "Benchmark", index: "05" },
];

export default function Nav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  useEffect(() => {
    const wide = window.matchMedia("(min-width: 1024px)");
    const onChange = () => wide.matches && setOpen(false);
    wide.addEventListener("change", onChange);
    return () => wide.removeEventListener("change", onChange);
  }, []);

  return (
    <>
      <header className="sticky top-0 z-50 border-b border-[color:var(--edge)] bg-paper">
        <div className="page flex h-[72px] items-center justify-between gap-6">
          <Link href="/" className="group flex items-center gap-3" aria-label="VERITAS overview">
            <Mark />
            <span className="font-mono text-[0.78rem] tracking-widest">VERITAS</span>
          </Link>

          <nav className="hidden items-center gap-8 lg:flex" aria-label="Sections">
            {LINKS.map((link) => {
              const active = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className="group relative flex items-baseline gap-2 py-2 text-[0.82rem] tracking-tight"
                >
                  <span className="font-mono text-[0.6rem] text-ash">{link.index}</span>
                  <span className={active ? "text-ink" : "text-graphite group-hover:text-ink"}>{link.label}</span>
                  <span
                    className={`absolute -bottom-px left-0 h-px w-full origin-left bg-ink transition-transform duration-300 ${
                      active ? "scale-x-100" : "scale-x-0 group-hover:scale-x-100"
                    }`}
                  />
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-3">
            <Link href="/score" className="btn hidden !py-2.5 !text-[0.64rem] sm:inline-flex">
              Assess your data
            </Link>
            <button
              type="button"
              onClick={() => setOpen((value) => !value)}
              className="flex h-10 w-10 flex-col items-center justify-center gap-[5px] border border-[color:var(--edge)] lg:hidden"
              aria-expanded={open}
              aria-label="Menu"
            >
              <span className={`h-px w-4 bg-ink transition-transform ${open ? "translate-y-[3px] rotate-45" : ""}`} />
              <span className={`h-px w-4 bg-ink transition-transform ${open ? "-translate-y-[3px] -rotate-45" : ""}`} />
            </button>
          </div>
        </div>
      </header>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="fixed inset-0 z-40 bg-paper pt-[72px] lg:hidden"
          >
            <div className="page flex h-full flex-col justify-between pb-12 pt-8">
              <nav className="flex flex-col">
                {[...LINKS, { href: "/score", label: "Assess your data", index: "06" }].map((link, i) => (
                  <motion.div
                    key={link.href}
                    initial={{ opacity: 0, y: 18 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.05 + i * 0.05, duration: 0.4, ease: "easeOut" }}
                  >
                    <Link
                      href={link.href}
                      className="flex items-baseline justify-between border-b border-[color:var(--edge)] py-5"
                    >
                      <span className="display text-4xl">{link.label}</span>
                      <span className="font-mono text-[0.6rem] text-ash">{link.index}</span>
                    </Link>
                  </motion.div>
                ))}
              </nav>
              <p className="eyebrow">TSYP14 · Track 2 · False-data injection</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

function Mark() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      <rect x="0.5" y="0.5" width="17" height="17" fill="none" stroke="currentColor" />
      <path d="M3 13.5 7.4 5.6l3.2 5.4L13 8.2l2 5.3" fill="none" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}
