"""Acceso restringido: sólo la cuenta Google autorizada abre el gemelo."""

import importlib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import acceso

ROOT = Path(__file__).resolve().parents[1]


class FakeUser(dict):
    def __init__(self, logged_in, **claims):
        super().__init__(claims)
        self.is_logged_in = logged_in


class StopApp(Exception):
    pass


class FakeSt:
    def __init__(self, user):
        self.user = user
        self.secrets = {}
        self.calls = []
        self.sidebar = self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def stop(self):
        raise StopApp

    def login(self):
        pass

    def logout(self):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args))


@pytest.fixture
def real_gate(monkeypatch):
    monkeypatch.delenv("PREDWEEM_ACCESO_PRUEBAS", raising=False)


def run_gate(monkeypatch, user):
    fake = FakeSt(user)
    monkeypatch.setattr(acceso, "st", fake)
    try:
        acceso.requerir_acceso("Gemelo")
    except StopApp:
        return False, fake.calls
    return True, fake.calls


def test_only_the_authorized_account_passes(monkeypatch, real_gate):
    assert acceso.emails_autorizados() == {"chantreguillermo@gmail.com"}
    ok, calls = run_gate(monkeypatch, FakeUser(False))
    assert not ok and ("button", (acceso.BOTON_INGRESO,)) in [(n, a[:1]) for n, a in calls]
    ok, _ = run_gate(monkeypatch, FakeUser(True, email="ChantreGuillermo@gmail.com", email_verified=True))
    assert ok
    ok, calls = run_gate(monkeypatch, FakeUser(True, email="otra.persona@gmail.com", email_verified=True))
    assert not ok and any(name == "error" for name, _ in calls)
    ok, _ = run_gate(monkeypatch, FakeUser(True, email="chantreguillermo@gmail.com", email_verified=False))
    assert not ok


def test_secrets_can_replace_the_list(monkeypatch):
    fake = FakeSt(FakeUser(False))
    fake.secrets = {"acceso": {"emails": ["Uno@Example.com"]}}
    monkeypatch.setattr(acceso, "st", fake)
    assert acceso.emails_autorizados() == {"uno@example.com"}


def test_test_bypass_requires_pytest_and_flag(monkeypatch, real_gate):
    assert not acceso._modo_pruebas()
    monkeypatch.setenv("PREDWEEM_ACCESO_PRUEBAS", "1")
    assert acceso._modo_pruebas()


def test_app_shows_only_the_login_screen_without_session(monkeypatch, real_gate):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception, [error.message for error in app.exception]
    assert [button.label for button in app.button] == [acceso.BOTON_INGRESO]
    assert not app.date_input and not app.selectbox
