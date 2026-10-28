// TypeScript mirrors of the FastAPI / Pydantic schemas (backend/schemas.py).

export type OptimizerKey = "sma" | "pso" | "ga" | "random";

export interface Health {
  status: string;
  models_loaded: string[];
  artifacts_ok: boolean;
  missing: string[];
}

export interface Options {
  stations: string[];
  port_types: string[];
  plug_types: string[];
}

export interface PredictRequest {
  station: string;
  port_type: string;
  plug_type: string;
  port_number: number;
  start_datetime: string;
  user_prev_sessions: number;
  user_prev_mean_kwh?: number | null;
}

export interface Prediction {
  model: string;
  kwh: number;
}

export interface PredictResponse {
  predictions: Prediction[];
  test_rmse: Record<string, number>;
  features_used: Record<string, string | number>;
}

export interface Station {
  name: string;
  lat: number;
  lon: number;
  port_types: string[];
  plug_types: string[];
  port_numbers: number[];
  session_count: number;
  first_session: string;
  last_session: string;
  mean_actual_kwh: number | null;
  mean_predicted_kwh: number | null;
  test_sessions: number;
  station_prev_mean_kwh: number;
  hourly_mean_kwh: (number | null)[];
}

export interface ComparisonRow {
  model: string;
  rmse: number;
  mae: number;
  r2: number;
  smape: number;
  train_time_s: number;
  rmse_improvement_pct: number;
  n_seeds: number;
  best_seed?: number | null;
  tuning_time_s?: number | null;
  rmse_mean?: number | null;
  rmse_std?: number | null;
}

export interface Comparison {
  quick: boolean;
  rows: ComparisonRow[];
}

export interface ConvergenceEntry {
  label: string;
  mean: number[];
  std: number[];
  seeds: Record<string, number[]>;
}

export type Convergence = Partial<Record<OptimizerKey, ConvergenceEntry>>;

export type HyperParams = Record<string, number>;

export interface HyperparamEntry {
  label: string;
  seed: number;
  best_val_rmse: number;
  best_iteration: number;
  params: HyperParams;
}

export type Hyperparams = Partial<Record<OptimizerKey, HyperparamEntry>>;

export interface TestSample {
  model: string;
  actual: number[];
  predicted: number[];
}

export interface FeatureImportance {
  feature: string;
  gain: number;
}

export interface HistoryStep {
  iteration: number;
  best_fitness: number;
  mean_fitness: number;
  best_params: HyperParams;
}

export interface History {
  optimizer: OptimizerKey;
  label: string;
  seed: number;
  history: HistoryStep[];
}

export interface SplitInfo {
  rows: number;
  start: string;
  end: string;
}

export interface DatasetSummary {
  steps: { step: string; rows: number }[];
  splits: Record<"train" | "val" | "test", SplitInfo>;
  target_stats: Record<string, number>;
  outlier_threshold_kwh: number;
  dropped_columns: string[];
  n_sessions: number;
  n_stations: number;
  quick: boolean;
}

// ---- "Analyse my area" (backend/services/area.py) ----
export interface AreaLocation {
  name: string;
  lat: number;
  lon: number;
  sessions: number;
  total_kwh: number;
  mean_kwh: number;
  daily_kwh: number;
  peak_hour: number | null;
  port_type: string;
  plug_type: string;
  mean_actual_kwh?: number | null;
  hourly_mean_kwh: (number | null)[];
  monthly_kwh?: { month: string; kwh: number }[];   // demand files only
}

export interface AreaSample {
  id: string;
  title: string;
  kind: "sessions" | "demand";
}

export interface AreaEstimateRequest {
  name: string;
  lat: number;
  lon: number;
  location_mean_kwh?: number | null;
  port_type: string;
  plug_type: string;
  port_number: number;
  start_datetime: string;
  user_prev_sessions: number;
  user_prev_mean_kwh?: number | null;
}

export interface AreaEstimate {
  kwh: number;
  low_kwh: number;
  high_kwh: number;
  typical_error_kwh: number;
}

export interface AreaSite {
  rank: number;
  lat: number;
  lon: number;
  served_kwh: number;
  share: number;
  served_locations: number;
  mean_distance_km: number;
}

export interface AreaAnalysis {
  summary: {
    mode: "sessions" | "demand";
    months?: number | null;
    rows_in: number;
    rows_used: number;
    rows_dropped: number;
    n_locations: number;
    date_start: string;
    date_end: string;
    days: number;
    total_kwh: number;
    mean_kwh: number;
    daily_kwh: number;
    has_actual: boolean;
    centre: { lat: number; lon: number };
    distance_from_training_km: number;
    far_from_training: boolean;
    accuracy?: { rmse: number; mae: number; r2: number; smape: number };
  };
  locations: AreaLocation[];
  hourly: { hour: number; sessions: number; total_kwh: number; mean_kwh: number }[];
  monthly: { month: string; total_kwh: number }[];
  placement: {
    k: number;
    sites: AreaSite[];
    objective_km: number;
    convergence_km: number[];
    evaluations: number;
    locations_considered: number;
  };
}
