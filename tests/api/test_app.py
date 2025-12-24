import pytest
from flask import Flask, url_for
from flask.testing import FlaskClient

from ml_monitoring_service.main import create_app


class TestApp:
    @pytest.fixture()
    def app(self) -> Flask:
        app = create_app()
        return app

    def test_health(self, client: FlaskClient):
        res = client.get(url_for("health"))

        assert res.status_code == 200
        assert res.json == {"status": "up"}

    def test_version(self, client: FlaskClient):
        res = client.get(url_for("version"))
        data = res.data.decode("utf-8")

        assert data.startswith("Python Version: 3")

    def test_app_runs(self):
        app = create_app()
        app.config["SERVER_NAME"] = "localhost"
        with app.test_client() as client:
            with app.app_context():
                res = client.get(url_for("health"))
                assert res.status_code == 200
