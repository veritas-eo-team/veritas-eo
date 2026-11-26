import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0B0B0C",
        graphite: "#2A2B2E",
        ash: "#6B6C70",
        line: "#D9D6CE",
        paper: "#F3F1EA",
        bone: "#FBF9F4",
        signal: "#C8452A",
        moss: "#3F5F4B",
      },
      fontFamily: {
        sans: ["var(--font-sans)"],
        serif: ["var(--font-serif)"],
        mono: ["var(--font-mono)"],
      },
      letterSpacing: {
        tightest: "-0.045em",
        wider: "0.14em",
        widest: "0.24em",
      },
      maxWidth: { page: "1440px" },
    },
  },
  plugins: [],
};

export default config;
