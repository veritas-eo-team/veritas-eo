import type { Metadata } from "next";
import type { ReactNode } from "react";

import Footer from "@/components/Footer";
import Nav from "@/components/Nav";

import "./globals.css";

export const metadata: Metadata = {
  title: "VERITAS · Weather data trust monitor",
  description:
    "Detects false-data injection in a daily solar-radiation series using NASA POWER, Sentinel-2 and a causal online detector.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <Nav />
        <main>{children}</main>
        <Footer />
      </body>
    </html>
  );
}
