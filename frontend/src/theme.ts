// Fixed colours (spec §11.3). Charts also use line style / markers so colour is never the only cue.
import type { OptimizerKey } from "./types";

export const COLORS = {
  sma: "#16a34a",
  pso: "#ea580c",
  ga: "#7c3aed",
  random: "#6b7280",
  default: "#2563eb",
  lr: "#0891b2",
  rf: "#a16207",
} as const;

export const OPT_ORDER: OptimizerKey[] = ["sma", "pso", "ga", "random"];

export const OPT_LABEL: Record<OptimizerKey, string> = {
  sma: "SMA",
  pso: "PSO",
  ga: "GA",
  random: "Random",
};

export const DASH: Record<OptimizerKey, string | undefined> = {
  sma: undefined,
  pso: "6 3",
  ga: "8 3 2 3",
  random: "2 3",
};

export const MODEL_COLOR: Record<string, string> = {
  "Linear Regression": COLORS.lr,
  "Random Forest": COLORS.rf,
  "XGBoost (default)": COLORS.default,
  "XGBoost + Random": COLORS.random,
  "XGBoost + GA": COLORS.ga,
  "XGBoost + PSO": COLORS.pso,
  "XGBoost + SMA": COLORS.sma,
};

export const AXIS_TICK = { fontSize: 10, fill: "#71717a", fontFamily: "IBM Plex Mono, monospace" };

export const TOOLTIP_STYLE = {
  borderRadius: 4,
  border: "1px solid #d4d4d8",
  boxShadow: "none",
  fontSize: 12,
  padding: "8px 12px",
};

export const fmt =(v: number | null | undefined, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);
