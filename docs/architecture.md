# AERIS-X Phase 14 Production Platform Architecture

## System Overview

AERIS-X now has a production-oriented foundation around the existing research code. HTTP routes are separated from domain contracts, application services, model adapters, repositories, and persistence.

## Layer Responsibilities

- **API:** versioned HTTP routes, request IDs, OpenAPI generation, and safe responses.
- **Domain:** Pydantic contracts for assets, telemetry, health, anomalies, FDI, risk, and model metadata.
- **Services:** application capability boundaries. Unconfigured inference fails explicitly.
- **Telemetry:** bounded, dataset-agnostic request contracts, deterministic validation, and model-input preparation.
- **Models:** framework-agnostic `ModelPredictor` interface; research checkpoints are not loaded at startup.
- **Repositories:** persistence abstractions independent of SQLAlchemy.
- **Database:** PostgreSQL-oriented SQLAlchemy engine and initial core records.
- **Core:** logging and error handling.

## Telemetry Lifecycle

The Phase 14C path is:

```
Telemetry -> Validation -> Preprocessing -> Model Registry
          -> Model Adapter -> Inference Result -> Persistence
```

`TelemetryInferenceRequest` carries an asset, bounded numeric window, source, optional feature ordering, operating-condition metadata, and quality metadata. It intentionally does not hard-code C-MAPSS. `TelemetryWindow` supports channel-labelled samples for callers that have explicit channel identifiers.

Validation rejects empty windows, inconsistent dimensions, duplicate feature ordering, non-numeric values, NaN/Inf values, and inconsistent channel sets. `DataQualityResult` reports validation status and explicit issue categories; no synthetic quality percentage is produced.

The monitoring quality layer also provides bounded sequence length, channel count, missing/non-finite counts, and per-feature descriptive statistics. Reference profiles are explicit inputs with provenance; comparisons report descriptive deltas only and do not claim drift or apply unvalidated alert thresholds.

`ModelInputPreprocessor` validates the model's declared shape and returns a model-ready array plus `PreprocessingMetadata`. It consumes already-prepared telemetry and never fits or refits a scaler. The existing C-MAPSS adapter remains responsible for loading immutable processed arrays, and the Step 6 preprocessing contract is not rewritten.

## Inference Orchestration

`InferenceOrchestrator` resolves the explicitly requested model, preprocesses the bounded window, invokes the framework-agnostic adapter, validates the finite `[0, 1]` health output, builds a `HealthState`, and optionally persists it. It contains no PyTorch implementation. Persistence failures are reported distinctly rather than returned as successful inference.

## Inference Lifecycle

```
Request -> Inference ID -> Validation -> Preprocessing -> Model Resolution
         -> Inference -> Output Validation -> Persistence -> Response
```

Each execution records its request correlation ID, resolved model identity, lifecycle status, actual measured duration, input/output status, and safe error category. The inference record contains no raw telemetry arrays. Repeated requests receive distinct server-generated inference IDs; distributed idempotency is not implemented.

`InferenceRepository` supports retrieval by inference ID, bounded asset history, and bounded health history. The API uses a process-local fallback when no database session is configured for development; SQLAlchemy/Alembic provide the durable PostgreSQL-oriented implementation.

`InferenceMetrics` is a framework-neutral in-process facade tracking counts and measured durations. Structured events carry request ID, inference ID, asset, model identity, task, status, and duration without telemetry arrays. It is intentionally not a Prometheus server or distributed tracing system.

## Model Lifecycle

The registry is the single model resolution boundary. Adapters expose the actual research checkpoint reference, task, framework, input shape, expected feature count and sequence length, and lazy loading strategy. Availability is inspected from metadata without loading every checkpoint. No semantic version is invented: identifiers such as `step8-fd001` are the existing research references.

## Runtime and Security Boundaries

Configuration is environment-driven. Liveness reports only process availability; readiness performs a real database connectivity check and returns `503` when unavailable. The container runs as a non-root user and has a bounded health check. Alembic migrations are explicit and are not run destructively during application import. API validation bounds telemetry dimensions and history limits; safe error handlers do not expose tracebacks, secrets, or checkpoint paths.

## API Layer

The application is created by `aeris.api.app:create_app`. Current endpoints:

- `GET /api/v1/health` — process liveness.
- `GET /api/v1/readiness` — application plus PostgreSQL connectivity.
- `GET /api/v1/version` — configured application identity.
- `GET /api/v1/models` — registered model metadata and artifact availability.
- `GET /api/v1/metrics` — bounded in-process inference metric snapshot.
- `POST /api/v1/telemetry/infer` — bounded telemetry health inference.
- `POST /api/v1/health-estimates` — backward-compatible alias for the telemetry inference flow.
- `GET /api/v1/inferences/{inference_id}` — retrieve one operational inference record.
- `GET /api/v1/assets/{asset_id}/inferences?limit=100` — bounded inference history.
- `GET /api/v1/assets/{asset_id}/health?limit=100` — bounded health history.

FastAPI generates `/docs` and `/openapi.json`; no hand-written OpenAPI contract is used.

## Model Serving

`ModelPredictor` keeps HTTP independent of PyTorch and scikit-learn. The Transformer and GRU health adapters lazily load the existing validated FD001 checkpoints; listing model metadata never loads weights. Missing artifacts and malformed windows return explicit capability/input errors.

## Persistence

SQLAlchemy models cover assets, model versions, health estimates, anomaly events, fault events, and risk assessments. Repository interfaces isolate the domain from SQLAlchemy, sessions roll back failed operations, and the initial schema is managed by Alembic. The database URL is environment-driven. No production connection is attempted during application import.

Health inference uses the Step 12 definition `clip(predicted_rul / 332.0, 0, 1)`. Raw telemetry and model binaries are not stored in PostgreSQL. Inference evidence identifies the source and bounded input dimensions without embedding the full payload.

## Configuration

`Settings` uses environment variables and optional local `.env` loading. `.env.example` contains placeholders only. No credentials are committed.

## Observability

Requests receive or propagate `X-Request-ID`. Structured logging includes timestamp, level, logger, and message. Future instrumentation can add latency and model-version fields without changing route contracts.

## Current Capabilities

Existing research capabilities include C-MAPSS health estimation, Transformer/GRU/LSTM experiments, sensor FDI, anomaly detection, SMAP/MSL analysis, and digital-twin fault injection.

## Not Yet Integrated

Anomaly and FDI services remain explicit capability interfaces; no new anomaly or fault algorithm is introduced and no detector is claimed to be production-integrated. Persistence callers must create referenced assets and model versions before writing dependent records.

## Planned Capabilities

Implemented in Phase 14C: bounded request/response inference, strict validation, preprocessing abstraction, Transformer health inference, and persistence integration points.

Implemented in Phase 14D: operational inference history, health history, model identity, request correlation, safe error categories, and measured inference duration.

Implemented in Phase 14E–14J: framework-neutral metrics/events, per-model usage counts, bounded data-quality profiles, explicit reference-profile comparison statuses, richer model lifecycle contracts, payload limits, safe diagnostics, and deployment/security documentation. Reference profiles must carry their own source metadata; no reference values are invented by the service.

The runtime lifecycle is documented as database availability -> explicit Alembic migration -> application startup -> readiness. Docker and Compose provide a reproducible foundation, but Docker execution is only claimed when the Docker CLI is available. Configuration is environment-driven and example credentials remain placeholders.

Future work: Prometheus, Grafana, distributed tracing/OpenTelemetry, validated drift alert policy, automated model promotion, online model monitoring, streaming ingestion, Kafka/RabbitMQ, authenticated access, dashboards, MLOps, and Aerospace Copilot/RAG. No real-time latency or production-readiness claim is made.

## Phase 15 Runtime Foundation

Implemented:

- environment-driven request, telemetry, history, CORS, and trusted-host limits
- bounded request-body rejection and safe security headers
- non-root Docker image
- explicit Compose topology: PostgreSQL -> migration service (`alembic upgrade head`) -> API
- required `POSTGRES_PASSWORD` for Compose rather than a weak fallback
- CI quality workflow for compilation, focused tests, SQLite migrations, roadmap/build gates, and protected-path checks

Validated locally:

- configuration and API security tests
- SQLite Alembic migration chain
- real Transformer inference, history, and metrics workflow

Not run in the current environment:

- Docker build/start/restart validation because Docker is unavailable
- PostgreSQL integration because PostgreSQL is unavailable

The migration service is explicit and non-destructive; application import does not initialize or mutate the database. Authentication, external metrics servers, streaming, and Kubernetes remain future boundaries.
