"""Las pruebas históricas fijan la campaña 2026, sin depender del reloj."""

import os

os.environ.setdefault("CAMPAIGN_YEAR", "2026")

# La suite abre la app sin ingreso con Google; ver acceso._modo_pruebas.
os.environ.setdefault("PREDWEEM_ACCESO_PRUEBAS", "1")
