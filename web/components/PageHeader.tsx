import type { ReactNode } from "react";

interface Props {
  index: string;
  section: string;
  title: string;
  description: string;
  aside?: ReactNode;
}

export default function PageHeader({ index, section, title, description, aside }: Props) {
  return (
    <header className="page pb-10 pt-12 md:pb-14 md:pt-16">
      <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-[color:var(--edge)] pb-4">
        <p className="eyebrow">
          {index} / {section}
        </p>
        {aside}
      </div>
      <div className="grid gap-6 pt-8 lg:grid-cols-12">
        <h1 className="display text-[clamp(2.4rem,5.4vw,4.6rem)] lg:col-span-7">{title}</h1>
        <p className="max-w-xl self-end text-[0.95rem] leading-relaxed text-graphite lg:col-span-5">{description}</p>
      </div>
    </header>
  );
}
