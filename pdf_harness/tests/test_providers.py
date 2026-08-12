import pytest

from pdf_harness.providers import ConnectorDefinition, load_connectors


def test_connector_rejects_http_and_arbitrary_environment_secrets():
    with pytest.raises(ValueError, match="HTTPS"):
        ConnectorDefinition(
            id="bad", label="Bad", base_url="http://attacker.test/v1",
            secret_ref="env:STANFORD_API_KEY", allowed_models=["x"],
        )
    with pytest.raises(ValueError, match="secret_ref"):
        ConnectorDefinition(
            id="bad", label="Bad", base_url="https://attacker.test/v1",
            secret_ref="env:HARNESS_MONGO_URI", allowed_models=["x"],
        )


def test_production_connector_requires_secret_manager_reference():
    raw = '[{"id":"s","label":"S","base_url":"https://example.test/v1","secret_ref":"env:STANFORD_API_KEY","allowed_models":["x"]}]'
    with pytest.raises(ValueError, match="production connectors"):
        load_connectors(raw, production=True)
