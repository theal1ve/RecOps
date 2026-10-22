import pytest
import yaml
from pydantic import ValidationError

from recops.config import CONFIG_DIR, Config, _deep_merge, load_config


@pytest.mark.parametrize("profile", ["base", "full", "smoke", "synthetic"])
def test_profiles_are_valid(profile: str) -> None:
    cfg = load_config(profile)
    assert cfg.seed == 42
    assert cfg.monitoring.psi_warn < cfg.monitoring.psi_alert


def test_unknown_key_is_rejected() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text(encoding="utf-8"))
    raw["timeline"]["eval_dayz"] = 7
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_invalid_value_is_rejected() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text(encoding="utf-8"))
    raw["timeline"]["eval_days"] = 0
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_workdir_overrides_paths(tmp_path) -> None:
    cfg = load_config("smoke", workdir=tmp_path)
    assert str(cfg.paths.artifacts).startswith(str(tmp_path))
    assert cfg.data.max_users == 400  # overlay applied


def test_deep_merge_is_recursive() -> None:
    assert _deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
