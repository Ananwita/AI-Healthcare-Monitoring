import os
import tempfile

import pytest

# Must run BEFORE any backend import: point the backend at a temporary test DB.
_TEST_DIR = tempfile.mkdtemp(prefix="hm_test_")
_TEST_DB = os.path.join(_TEST_DIR, "test.db").replace(os.sep, "/")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB}"

from fastapi.testclient import TestClient  # noqa: E402

from backend.database import Base, engine  # noqa: E402
from backend import models  # noqa: E402,F401  (registers tables)
from backend.main import app  # noqa: E402

# Safety guard: refuse to run if somehow pointed at the real database.
assert "hm_test_" in str(engine.url), f"Tests must not use real DB: {engine.url}"


@pytest.fixture(autouse=True)
def fresh_db():
    """Give every test an empty database."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def patient_id(client):
    """Create one patient and return its integer id."""
    response = client.post(
        "/patients",
        json={"name": "Test Patient", "age": 40, "language": "English"},
    )
    assert response.status_code == 200
    return response.json()["patient_id"]