from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from aeris.db.models import Base
from aeris.domain.schemas import Asset, ModelMetadata
from aeris.models.adapters import InvalidModelInputError, ModelRegistry, ModelUnavailableError
from aeris.repositories.sqlalchemy import AssetRepository, ModelRepository


def test_registry_lists_models_without_loading_checkpoints():
    registry = ModelRegistry(Path(__file__).parents[1])
    metadata = registry.list_metadata()
    assert {item.model_id for item in metadata} == {
        "aeris-transformer-health", "aeris-gru-health"
    }
    assert all(item.input_shape == (30, 15) for item in metadata)


def test_missing_checkpoint_is_explicit(tmp_path):
    registry = ModelRegistry(tmp_path)
    metadata = registry.list_metadata()[0]
    assert metadata.status == "unavailable"
    try:
        registry.get(metadata.model_id).predict(np.zeros((30, 15), dtype=np.float32))
    except ModelUnavailableError as error:
        assert "unavailable" in str(error).lower()
    else:
        raise AssertionError("Expected unavailable model error")


def test_model_input_contract_rejects_bad_shape():
    registry = ModelRegistry(Path(__file__).parents[1])
    predictor = registry.get("aeris-transformer-health")
    try:
        predictor.predict(np.zeros((2, 15), dtype=np.float32))
    except InvalidModelInputError:
        pass
    else:
        raise AssertionError("Expected invalid input error")


def test_repositories_round_trip_metadata_and_assets():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        asset = Asset(asset_id="engine-1", asset_type="turbofan", metadata={"fleet": "test"})
        AssetRepository(session).save(asset)
        metadata = ModelMetadata(
            model_id="model-1", model_name="test", model_version="1",
            task="health_estimation", framework="pytorch", status="available",
            input_shape=(30, 15), expected_features=15,
            dataset_reference="FD001", metadata={"device": "cpu"},
        )
        ModelRepository(session).save(metadata)
        session.commit()
        assert AssetRepository(session).get("engine-1").metadata == {"fleet": "test"}
        assert ModelRepository(session).get("model-1") == metadata
