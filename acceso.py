"""Acceso restringido al gemelo: ingreso con Google y lista de emails autorizados.

La autenticación usa ``st.login`` (OpenID Connect). Las credenciales del
cliente OAuth van en los secretos de la app (sección ``[auth]``), nunca en el
repositorio. Si faltan, la app no se abre: falla cerrada.

Los emails autorizados pueden ampliarse desde los secretos, sin tocar el código:

    [acceso]
    emails = ["chantreguillermo@gmail.com"]
"""

from __future__ import annotations

import os
import sys

import streamlit as st

EMAILS_AUTORIZADOS = ("chantreguillermo@gmail.com",)
BOTON_INGRESO = "Ingresar con Google"


def emails_autorizados() -> set[str]:
    try:
        configurados = st.secrets.get("acceso", {}).get("emails")
    except Exception:  # Sin archivo de secretos.
        configurados = None
    emails = configurados or EMAILS_AUTORIZADOS
    return {str(email).strip().lower() for email in emails if str(email).strip()}


def _modo_pruebas() -> bool:
    """Sólo la suite de pruebas puede omitir el ingreso, nunca el despliegue."""
    return os.environ.get("PREDWEEM_ACCESO_PRUEBAS") == "1" and "pytest" in sys.modules


def requerir_acceso(nombre_app: str) -> None:
    """Detiene la app salvo para una cuenta Google autorizada y verificada."""
    if _modo_pruebas():
        return

    if not getattr(st.user, "is_logged_in", False):
        st.title(nombre_app)
        st.info("Acceso restringido. Ingrese con la cuenta de Google autorizada.")
        st.button(BOTON_INGRESO, on_click=st.login, type="primary")
        st.stop()

    email = str(st.user.get("email") or "").strip().lower()
    verificado = st.user.get("email_verified", True) is not False
    if not verificado or email not in emails_autorizados():
        st.title(nombre_app)
        st.error(f"La cuenta {email or 'sin email'} no tiene acceso a este gemelo.")
        st.button("Cerrar sesión", on_click=st.logout)
        st.stop()

    with st.sidebar:
        st.caption(f"Sesión: {email}")
        st.button("Cerrar sesión", on_click=st.logout, key="acceso_cerrar_sesion")
