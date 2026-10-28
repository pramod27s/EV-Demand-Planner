// Axios instance + one typed function per backend endpoint.
import axios, { AxiosError } from "axios";
import type {
  AreaAnalysis, AreaEstimate, AreaEstimateRequest, AreaSample, Comparison, Convergence, DatasetSummary, FeatureImportance, Health, History, Hyperparams,
  Options, OptimizerKey, PredictRequest, PredictResponse, Station, TestSample,
} from "../types";

export const API_URL: string = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const http = axios.create({ baseURL: API_URL, timeout: 15000 });

const get = <T,>(url: string, params?: object) => http.get<T>(url, { params }).then((r) => r.data);

export const api = {
  health: () => get<Health>("/api/health"),
  datasetSummary: () => get<DatasetSummary>("/api/dataset/summary"),
  stations: () => get<Station[]>("/api/stations"),
  station: (name: string) => get<Station>(`/api/stations/${encodeURIComponent(name)}`),
  options: () => get<Options>("/api/options"),
  predict: (body: PredictRequest) =>
    http.post<PredictResponse>("/api/predict", body).then((r) => r.data),
  comparison: () => get<Comparison>("/api/results/comparison"),
  convergence: (optimizer?: OptimizerKey) =>
    get<Convergence>("/api/results/convergence", optimizer ? { optimizer } : undefined),
  hyperparams: () => get<Hyperparams>("/api/results/hyperparams"),
  testSample: (model: "default" | "sma") => get<TestSample>("/api/results/test-sample", { model }),
  featureImportance: () => get<FeatureImportance[]>("/api/results/feature-importance"),
  optimizationRuns: () => get<Record<string, number[]>>("/api/optimization/runs"),
  history: (optimizer: OptimizerKey, seed: number) =>
    get<History>("/api/optimization/history", { optimizer, seed }),

  // "Analyse my area": the CSV is sent as the raw request body
  analyzeArea: (csv: Blob | string, k: number) =>
    http.post<AreaAnalysis>("/api/area/analyze", csv, { params: { k }, headers: { "Content-Type": "text/csv" }, timeout: 120000 })
      .then((r) => r.data),
  areaPredictionsCsv: (csv: Blob | string) =>
    http.post<string>("/api/area/predictions", csv, { headers: { "Content-Type": "text/csv" }, responseType: "text", timeout: 120000 })
      .then((r) => r.data),
  estimateAt: (body: AreaEstimateRequest) => http.post<AreaEstimate>("/api/area/estimate", body).then((r) => r.data),
  areaSamples: () => get<AreaSample[]>("/api/area/samples"),
  areaSampleCsv: (id: string) => http.get<string>(`/api/area/samples/${id}`, { responseType: "text", timeout: 60000 }).then((r) => r.data),
  areaSample: () => http.get<string>("/api/area/sample", { responseType: "text" }).then((r) => r.data),
  areaTemplateUrl: `${API_URL}/api/area/template`,
};

/** Human-readable message for any API failure (including the 503 "run the pipeline" case). */
export function errorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    if (!err.response) return `Cannot reach the API at ${API_URL}. Is the backend running (uvicorn backend.main:app)?`;
    const detail = (err.response.data as { detail?: unknown })?.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object" && "msg" in detail) return String((detail as { msg: string }).msg);
    if (Array.isArray(detail)) return detail.map((d) => d.msg).join("; ");
    return `Request failed (${err.response.status})`;
  }
  return err instanceof Error ? err.message : String(err);
}
