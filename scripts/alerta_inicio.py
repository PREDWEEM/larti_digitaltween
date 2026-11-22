"""Alerta preventiva de inicio para el aviso por email (GitHub Actions).

Ejecuta la app sin navegador, con la configuración que muestra al abrirse
(campaña actual, fecha de hoy, parámetros por defecto y sin conteos de lote),
y lee el mismo resultado de ``onset_alert`` que ve el usuario.

Si el modelo prevé el inicio de la emergencia dentro de los próximos 7 días
(estado ``watch``), deja ``avisar=true`` en ``GITHUB_OUTPUT`` junto con el
título y el cuerpo del aviso. El workflow abre entonces un issue, y GitHub
lo envía por email.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_URL = "https://citqf36upunnthfnngj7sy.streamlit.app/"
GITHUB_USUARIO = "PREDWEEM"


def nombre_sitio() -> str:
    texto = (ROOT / "app.py").read_text(encoding="utf-8")
    titulo = re.search(r"<h1>Gemelo Digital · Lolium ([^<]+)</h1>", texto)
    return titulo.group(1).strip() if titulo else ROOT.name


def calcular_alerta() -> dict:
    """Corre app.py como lo abriría un usuario y devuelve su alerta de inicio."""
    os.environ["PREDWEEM_ACCESO_PRUEBAS"] = "1"  # Ejecución local, sin servidor web.
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600).run()
    if app.exception:
        raise RuntimeError("; ".join(error.message for error in app.exception))
    alertas = []
    for elemento in app.json:
        try:
            datos = json.loads(elemento.value)
        except (TypeError, ValueError):
            continue
        if isinstance(datos, dict) and {"status", "horizon_days", "model_onset_date"} <= set(datos):
            alertas.append(datos)
    if len(alertas) != 1:
        info = " ".join(str(item.value) for item in app.info)
        raise RuntimeError(f"No se encontró la alerta de inicio en la app. {info}".strip())
    campania = int(app.selectbox(key="campaign_year").value)
    corte = app.date_input(key=f"state_date_{campania}").value
    return {**alertas[0], "campania": campania, "corte": str(corte)}


def titulo_aviso(alerta: dict, sitio: str) -> str:
    return f"Alerta de inicio de emergencia · {sitio} · campaña {alerta['campania']}"


def cuerpo_aviso(alerta: dict, sitio: str) -> str:
    inicio = alerta.get("model_onset_date") or "sin fecha"
    return "\n".join([
        f"@{GITHUB_USUARIO} **{alerta['title']}** en el gemelo de Lolium {sitio}.",
        "",
        str(alerta.get("message", "")),
        "",
        f"- Inicio modelado: **{inicio}**",
        f"- Fecha del estado: {alerta['corte']} (campaña {alerta['campania']})",
        f"- Base del aviso: {alerta.get('mode', '')}",
        f"- Gemelo: {APP_URL}",
        "",
        "Calculado automáticamente tras la actualización meteorológica, con los "
        "parámetros por defecto de la app y sin conteos de campo del lote. "
        "Se avisa una sola vez por campaña: cierre este issue cuando lo haya leído.",
    ])


def aviso_de_prueba(sitio: str, momento: str) -> tuple[str, str]:
    titulo = f"Prueba de aviso por email · {sitio} · {momento}"
    cuerpo = "\n".join([
        f"@{GITHUB_USUARIO} Prueba del aviso automático de inicio de emergencia "
        f"del gemelo de Lolium {sitio}.",
        "",
        "Si recibió este mensaje por email, el canal funciona. No es una alerta real:",
        "puede cerrar este issue.",
        "",
        f"- Gemelo: {APP_URL}",
    ])
    return titulo, cuerpo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cuerpo", default="alerta_inicio.md", help="Archivo para el texto del aviso.")
    parser.add_argument("--prueba", action="store_true", help="Genera un aviso de prueba sin calcular la alerta.")
    args = parser.parse_args(argv)

    sitio = nombre_sitio()
    if args.prueba:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        momento = datetime.now(ZoneInfo("America/Argentina/Buenos_Aires")).strftime("%d/%m/%Y %H:%M")
        titulo, cuerpo = aviso_de_prueba(sitio, momento)
        Path(args.cuerpo).write_text(cuerpo + "\n", encoding="utf-8")
        salida = os.environ.get("GITHUB_OUTPUT")
        if salida:
            with open(salida, "a", encoding="utf-8") as handle:
                handle.write("avisar=true\n")
                handle.write(f"titulo={titulo}\n")
        print(titulo)
        return 0
    alerta = calcular_alerta()
    avisar = alerta.get("status") == "watch"
    print(json.dumps(alerta, ensure_ascii=False, indent=2))
    if avisar:
        Path(args.cuerpo).write_text(cuerpo_aviso(alerta, sitio) + "\n", encoding="utf-8")
    salida = os.environ.get("GITHUB_OUTPUT")
    if salida:
        with open(salida, "a", encoding="utf-8") as handle:
            handle.write(f"avisar={'true' if avisar else 'false'}\n")
            handle.write(f"titulo={titulo_aviso(alerta, sitio)}\n")
    print("Aviso por email:", "sí" if avisar else f"no (estado {alerta.get('status')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
