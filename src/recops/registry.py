"""Thin MLflow wrapper: tracking + Model Registry with a ``production`` alias.

MLflow >= 2.9 deprecates registry *stages* in favour of aliases; the alias ``production``
plays the role of the Production stage.
"""

from __future__ import annotations

from pathlib import Path

from recops.bundle import Bundle
from recops.config import Config


def _clean(name: str) -> str:
    return name.replace("@", "_at_")


class Registry:
    def __init__(self, cfg: Config):
        import mlflow
        from mlflow.tracking import MlflowClient

        self.cfg = cfg
        self.mlflow = mlflow
        root = Path(cfg.paths.artifacts) / "mlflow"
        root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{root.resolve()}/mlflow.db")
        name = cfg.mlflow.experiment
        if mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(
                name, artifact_location=(root / "artifacts").resolve().as_uri()
            )
        mlflow.set_experiment(name)
        self.client = MlflowClient()
        self.model_name = cfg.mlflow.model_name
        self.alias = cfg.mlflow.alias
        self.cache = root / "prod_cache"

    def production_version(self):
        """Current production ModelVersion or None."""
        try:
            return self.client.get_model_version_by_alias(self.model_name, self.alias)
        except Exception:  # noqa: BLE001 - no model/alias yet
            return None

    def load_production(self) -> Bundle | None:
        mv = self.production_version()
        if mv is None:
            return None
        self.cache.mkdir(parents=True, exist_ok=True)
        local = self.mlflow.artifacts.download_artifacts(
            artifact_uri=mv.source, dst_path=str(self.cache / f"v{mv.version}")
        )
        return Bundle.load(Path(local))

    def log_cycle(
        self,
        k: int,
        params: dict,
        metrics: dict[str, float],
        tags: dict[str, str],
        files: list[Path],
        bundle_dir: Path | None,
    ) -> str | None:
        """Log one cycle as a run. If ``bundle_dir`` is given, log it, register a new model
        version and move the production alias to it. Returns the new version or None."""
        mlflow = self.mlflow
        with mlflow.start_run(run_name=f"cycle-{k}") as run:
            mlflow.log_params({k_: str(v)[:450] for k_, v in params.items()})
            mlflow.log_metrics({_clean(n): float(v) for n, v in metrics.items() if v == v})
            mlflow.set_tags(tags)
            for f in files:
                mlflow.log_artifact(str(f))
            if bundle_dir is None:
                return None
            mlflow.log_artifacts(str(bundle_dir), artifact_path="bundle")
            source = f"{run.info.artifact_uri}/bundle"
            try:
                self.client.create_registered_model(self.model_name)
            except Exception:  # noqa: BLE001 - already exists
                pass
            mv = self.client.create_model_version(self.model_name, source, run.info.run_id)
            self.client.set_registered_model_alias(self.model_name, self.alias, mv.version)
            return str(mv.version)
