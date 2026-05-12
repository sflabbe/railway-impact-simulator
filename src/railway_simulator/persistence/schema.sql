PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    root_dir TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS config_snapshots (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    source_path TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS studies (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    study_type TEXT NOT NULL,
    base_config_id TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    FOREIGN KEY(project_id) REFERENCES projects(id),
    FOREIGN KEY(base_config_id) REFERENCES config_snapshots(id)
);

CREATE TABLE IF NOT EXISTS scenarios (
    id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL,
    name TEXT NOT NULL,
    weight REAL NOT NULL,
    params_json TEXT NOT NULL,
    meta_json TEXT NOT NULL,
    FOREIGN KEY(study_id) REFERENCES studies(id)
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    scenario_id TEXT NOT NULL,
    status TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    result_csv_path TEXT,
    elapsed_s REAL,
    error_message TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    FOREIGN KEY(scenario_id) REFERENCES scenarios(id)
);

CREATE TABLE IF NOT EXISTS run_metrics (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS srs_curves (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    zeta REAL NOT NULL,
    oscillator_mass REAL NOT NULL,
    force_column TEXT NOT NULL,
    curve_csv_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    owner_type TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_config_project ON config_snapshots(project_id);
CREATE INDEX IF NOT EXISTS idx_studies_project ON studies(project_id);
CREATE INDEX IF NOT EXISTS idx_scenarios_study ON scenarios(study_id);
CREATE INDEX IF NOT EXISTS idx_runs_scenario ON runs(scenario_id);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON run_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_srs_run ON srs_curves(run_id);
