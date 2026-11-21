from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import Database  # noqa: E402
from app.gateway import LLMGateway  # noqa: E402
from tests.fakes import FakeCompletion, make_settings  # noqa: E402


@pytest.fixture
def fake() -> FakeCompletion:
    return FakeCompletion()


@pytest.fixture
def make_client(fake, tmp_path):
    """Build a TestClient around a real Database + real LLMGateway, with the provider faked."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    clients = []

    def _make(**overrides):
        settings = make_settings(database_url=f"sqlite:///{tmp_path}/app.db", **overrides)
        gateway = LLMGateway(settings, completion_fn=fake)
        app = create_app(settings=settings, gateway=gateway, database=Database(settings.database_url))
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()
