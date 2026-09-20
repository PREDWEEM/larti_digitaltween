"""Visita pública de Lartigau; comprueba la interfaz, no sólo una respuesta HTTP.

Se ejecuta desde GitHub Actions con Playwright, separado de las dependencias
del modelo. No modifica observaciones ni reinicia la aplicación.
"""

from __future__ import annotations

import os
from pathlib import Path
import time

from playwright.sync_api import Error, TimeoutError, sync_playwright


APP_URL = "https://citqf36upunnthfnngj7sy.streamlit.app/"
HEADING = "Gemelo Digital · Lolium Lartigau"
MAX_WAIT_SECONDS = 300


def app_is_ready(frame):
    """El gráfico y los indicadores deben existir en el mismo documento."""
    return (
        frame.get_by_role("heading", name=HEADING, exact=True).is_visible()
        and frame.get_by_text("Emergencia estimada", exact=True).is_visible()
        and frame.get_by_role("tabpanel", name="Estado del lote", exact=True).is_visible()
        and frame.locator('[data-testid="stPlotlyChart"]').first.is_visible()
    )


def visit_app():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_default_timeout(5000)
            started = time.monotonic()
            wake_clicked = False
            ready_since = None
            try:
                page.goto(APP_URL, wait_until="domcontentloaded", timeout=60000)
            except TimeoutError:
                # La carga puede continuar después del tiempo de navegación.
                print("La apertura inicial sigue en curso.", flush=True)

            while time.monotonic() - started < MAX_WAIT_SECONDS:
                ready = False
                # Community Cloud puede alojar la interfaz dentro de un iframe.
                for frame in page.frames:
                    try:
                        if frame.locator('[data-testid="stException"]').count():
                            raise RuntimeError("La aplicación muestra una excepción de Streamlit.")
                        wake = frame.get_by_role(
                            "button", name="Yes, get this app back up!", exact=True
                        )
                        if not wake_clicked and wake.is_visible() and wake.is_enabled():
                            wake.click()
                            wake_clicked = True
                            print("Se solicitó despertar la aplicación.", flush=True)
                        ready = app_is_ready(frame) or ready
                    except Error:
                        # Un iframe puede cambiar mientras Streamlit arranca.
                        continue

                if ready:
                    if ready_since is None:
                        ready_since = time.monotonic()
                    elif time.monotonic() - ready_since >= 5:
                        seconds = round(time.monotonic() - started)
                        return f"Lartigau operativo: indicadores y gráfico cargados ({seconds} s)."
                else:
                    ready_since = None
                page.wait_for_timeout(5000)

            raise RuntimeError("No se pudo confirmar la carga del gemelo en cinco minutos.")
        finally:
            browser.close()


def report(message):
    print(message, flush=True)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n\nAplicación: {APP_URL}\n")


if __name__ == "__main__":
    try:
        report(visit_app())
    except RuntimeError as error:
        report(f"FALLO: {error}")
        raise SystemExit(1)
    except Error:
        report("FALLO: no se pudo completar la visita con el navegador.")
        raise SystemExit(1)
