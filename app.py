import io
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.express as px

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Markowitz — Tecnología + Industriales",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Markowitz — Tecnología + Industriales")
st.caption(
    "Universos oficiales: XLK (Tecnología) y XLI (Industriales). "
    "Precios históricos: Yahoo Finance."
)

TOP_N_DEFAULT = 10
MIN_COVERAGE = 0.90

# Archivos oficiales de holdings diarios de State Street.
STATE_STREET_FILES = {
    "Tecnología": {
        "fund": "XLK",
        "url": "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlk.xlsx",
    },
    "Industriales": {
        "fund": "XLI",
        "url": "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xli.xlsx",
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0 Safari/537.36"
    ),
    "Accept": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
        "application/octet-stream,*/*"
    ),
    "Referer": "https://www.ssga.com/",
}


# ============================================================
# STATE STREET — HOLDINGS OFICIALES
# ============================================================
def _find_header_row(raw: pd.DataFrame):
    """
    Busca automáticamente la fila de encabezados del XLSX.
    State Street puede introducir varias filas de metadata antes de la tabla.
    """
    for idx in range(min(len(raw), 40)):
        vals = [
            str(v).strip().lower()
            for v in raw.iloc[idx].tolist()
            if pd.notna(v)
        ]
        joined = " | ".join(vals)

        has_ticker = any(
            token in joined
            for token in ["ticker", "symbol"]
        )
        has_name = any(
            token in joined
            for token in ["name", "security"]
        )

        if has_ticker and has_name:
            return idx

    raise ValueError(
        "No se pudo localizar la fila de encabezados del archivo de holdings."
    )


def _normalize_column_name(value):
    return str(value).strip().lower().replace("\n", " ")


def _choose_column(columns, candidates):
    normalized = {
        col: _normalize_column_name(col)
        for col in columns
    }

    for candidate in candidates:
        for original, norm in normalized.items():
            if candidate == norm or candidate in norm:
                return original

    return None


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_state_street_holdings(sector_label: str) -> pd.DataFrame:
    """
    Descarga el XLSX oficial de holdings diarios del fondo sectorial.
    """
    cfg = STATE_STREET_FILES[sector_label]
    url = cfg["url"]
    fund = cfg["fund"]

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    if len(response.content) < 2000:
        raise ValueError(
            f"State Street devolvió un archivo demasiado pequeño para {fund}."
        )

    # Leer sin asumir en qué fila está el encabezado.
    raw = pd.read_excel(
        io.BytesIO(response.content),
        header=None,
        engine="openpyxl",
    )

    header_row = _find_header_row(raw)

    table = pd.read_excel(
        io.BytesIO(response.content),
        header=header_row,
        engine="openpyxl",
    )

    ticker_col = _choose_column(
        table.columns,
        ["ticker", "symbol"],
    )
    name_col = _choose_column(
        table.columns,
        ["name", "security name", "security"],
    )
    weight_col = _choose_column(
        table.columns,
        ["weight", "weight (%)", "weight %"],
    )

    if ticker_col is None:
        raise ValueError(
            f"No se encontró una columna Ticker/Symbol en el XLSX de {fund}. "
            f"Columnas: {list(table.columns)}"
        )

    out = pd.DataFrame()
    out["Ticker"] = table[ticker_col].astype(str).str.strip()

    if name_col is not None:
        out["Nombre"] = table[name_col].astype(str).str.strip()
    else:
        out["Nombre"] = ""

    if weight_col is not None:
        out["Peso fondo"] = pd.to_numeric(
            table[weight_col],
            errors="coerce",
        )
    else:
        out["Peso fondo"] = np.nan

    out["Sector"] = sector_label
    out["Fondo"] = fund

    # Normalización para Yahoo.
    out["Ticker"] = (
        out["Ticker"]
        .str.replace(".", "-", regex=False)
        .str.upper()
    )

    # Quitar filas que no son acciones.
    invalid = {
        "",
        "NAN",
        "NONE",
        "CASH",
        "USD",
        "-",
    }

    out = out[
        ~out["Ticker"].isin(invalid)
        & out["Ticker"].str.match(r"^[A-Z0-9\-]+$", na=False)
    ].copy()

    out = out.drop_duplicates("Ticker").reset_index(drop=True)

    if len(out) < 10:
        raise ValueError(
            f"Solo se detectaron {len(out)} holdings válidos para {fund}; "
            "se esperaba un universo sectorial mayor."
        )

    return out


# ============================================================
# YAHOO FINANCE — PRECIOS
# ============================================================
def extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        close = raw["Close"].copy()

    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()

        close = raw[["Close"]].copy()

        if len(tickers) == 1:
            close.columns = tickers

    if isinstance(close, pd.Series):
        close = close.to_frame()

    return close.sort_index()


def _download_batch(batch):
    raw = yf.download(
        tickers=batch,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
        timeout=25,
    )
    return extract_close(raw, batch)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def download_prices(tickers_tuple: tuple[str, ...]) -> pd.DataFrame:
    """
    Descarga todos los tickers de ambos sectores en bloques pequeños.
    Si falla un bloque, lo divide.
    """
    tickers = list(dict.fromkeys(tickers_tuple))

    if not tickers:
        return pd.DataFrame()

    frames = []
    batch_size = 35

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        try:
            close = _download_batch(batch)

            if not close.empty:
                frames.append(close)

        except Exception:
            # Fallback: dividir el bloque en grupos menores.
            midpoint = max(len(batch) // 2, 1)

            for small in (batch[:midpoint], batch[midpoint:]):
                if not small:
                    continue

                try:
                    close = _download_batch(small)
                    if not close.empty:
                        frames.append(close)
                except Exception:
                    pass

        # Pequeña pausa entre bloques.
        time.sleep(0.2)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index()

    return prices


# ============================================================
# CÁLCULOS
# ============================================================
def calculate_ranking(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    min_coverage: float = 0.90,
) -> pd.DataFrame:

    available = [
        t for t in metadata["Ticker"].tolist()
        if t in prices.columns
    ]

    if not available:
        return pd.DataFrame()

    sector_prices = prices[available]
    max_obs = int(sector_prices.notna().sum().max())

    rows = []

    for ticker in available:
        s = sector_prices[ticker].dropna()

        if len(s) < 2:
            continue

        coverage = len(s) / max_obs if max_obs else 0

        if coverage < min_coverage:
            continue

        initial = float(s.iloc[0])
        final = float(s.iloc[-1])

        if initial <= 0:
            continue

        total_return = final / initial - 1.0

        days = max((s.index[-1] - s.index[0]).days, 1)
        years = days / 365.25

        cagr = (final / initial) ** (1 / years) - 1

        daily = s.pct_change(fill_method=None).dropna()

        vol = (
            daily.std() * np.sqrt(252)
            if len(daily) > 1
            else np.nan
        )

        rows.append(
            {
                "Ticker": ticker,
                "Precio inicial": initial,
                "Precio final": final,
                "Rendimiento 5Y": total_return,
                "CAGR": cagr,
                "Volatilidad anual": vol,
                "Cobertura": coverage,
                "Fecha inicial": s.index[0].date(),
                "Fecha final": s.index[-1].date(),
            }
        )

    ranking = pd.DataFrame(rows)

    if ranking.empty:
        return ranking

    ranking = ranking.merge(
        metadata[
            ["Ticker", "Nombre", "Sector", "Fondo", "Peso fondo"]
        ],
        on="Ticker",
        how="left",
    )

    return (
        ranking
        .sort_values("Rendimiento 5Y", ascending=False)
        .reset_index(drop=True)
    )


def show_ranking(df: pd.DataFrame):
    show = df.copy()

    show.insert(0, "Posición", range(1, len(show) + 1))

    for col in [
        "Rendimiento 5Y",
        "CAGR",
        "Volatilidad anual",
        "Cobertura",
    ]:
        show[col] = show[col] * 100

    columns = [
        "Posición",
        "Ticker",
        "Nombre",
        "Rendimiento 5Y",
        "CAGR",
        "Volatilidad anual",
        "Cobertura",
        "Precio inicial",
        "Precio final",
    ]

    st.dataframe(
        show[columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rendimiento 5Y": st.column_config.NumberColumn(
                "Rendimiento 5 años", format="%.2f %%"
            ),
            "CAGR": st.column_config.NumberColumn(
                "CAGR anual", format="%.2f %%"
            ),
            "Volatilidad anual": st.column_config.NumberColumn(
                "Volatilidad anual", format="%.2f %%"
            ),
            "Cobertura": st.column_config.NumberColumn(
                "Cobertura", format="%.1f %%"
            ),
            "Precio inicial": st.column_config.NumberColumn(
                "Precio inicial", format="$ %.2f"
            ),
            "Precio final": st.column_config.NumberColumn(
                "Precio final", format="$ %.2f"
            ),
        },
    )


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Configuración")

top_n = st.sidebar.slider(
    "Acciones por sector",
    min_value=5,
    max_value=20,
    value=TOP_N_DEFAULT,
    step=1,
)

st.sidebar.write("**Periodo:** 5 años")
st.sidebar.write("**Cobertura mínima:** 90%")
st.sidebar.write("**Tecnología:** XLK")
st.sidebar.write("**Industriales:** XLI")

st.sidebar.caption(
    "XLK y XLI se usan únicamente para definir el universo actual "
    "de cada sector. Los rendimientos se calculan con precios de Yahoo Finance."
)


# ============================================================
# PANTALLA INICIAL
# ============================================================
st.info(
    "Fuente del universo: holdings diarios oficiales de State Street "
    "(XLK y XLI). Fuente de precios históricos: Yahoo Finance."
)

if not st.button(
    "🚀 Analizar acciones",
    type="primary",
    use_container_width=True,
):
    st.subheader("Metodología")
    st.markdown(
        """
        1. Descargar los holdings oficiales actuales de **XLK** y **XLI**.
        2. Extraer los tickers de cada sector.
        3. Descargar **5 años** de precios diarios desde Yahoo Finance.
        4. Exigir al menos **90% de cobertura** del periodo.
        5. Calcular rendimiento acumulado, CAGR y volatilidad.
        6. Seleccionar las **Top acciones de cada sector**.
        7. Usar el universo final para el modelo de **Markowitz**.
        """
    )
    st.stop()


# ============================================================
# OBTENER HOLDINGS OFICIALES
# ============================================================
universes = {}

for sector_label in ["Tecnología", "Industriales"]:
    fund = STATE_STREET_FILES[sector_label]["fund"]

    try:
        with st.spinner(
            f"Descargando holdings oficiales de {fund}..."
        ):
            holdings = get_state_street_holdings(sector_label)

        universes[sector_label] = holdings

        st.success(
            f"{fund}: {len(holdings)} holdings válidos detectados."
        )

    except Exception as e:
        st.error(
            f"No se pudieron obtener los holdings oficiales de {fund}."
        )
        st.code(str(e))
        st.stop()


# ============================================================
# DESCARGA CONJUNTA DE PRECIOS
# ============================================================
all_metadata = pd.concat(
    list(universes.values()),
    ignore_index=True,
)

all_tickers = tuple(
    all_metadata["Ticker"]
    .drop_duplicates()
    .tolist()
)

with st.spinner(
    f"Descargando 5 años de precios para {len(all_tickers)} acciones..."
):
    prices = download_prices(all_tickers)

if prices.empty:
    st.error(
        "Yahoo Finance no devolvió precios. "
        "Puede existir un límite temporal de solicitudes."
    )
    st.stop()

st.caption(
    f"Yahoo devolvió datos para {len(prices.columns)} de "
    f"{len(all_tickers)} tickers solicitados."
)


# ============================================================
# DOS RANKINGS
# ============================================================
selected = []

for sector_label in ["Tecnología", "Industriales"]:

    ranking = calculate_ranking(
        prices,
        universes[sector_label],
        MIN_COVERAGE,
    )

    st.divider()
    st.header(f"🏆 {sector_label}")

    if ranking.empty:
        st.error(
            f"No se pudo formar el ranking de {sector_label}."
        )
        continue

    st.write(
        f"Acciones válidas con cobertura ≥90%: **{len(ranking)}**"
    )

    top = ranking.head(top_n).copy()
    selected.append(top)

    show_ranking(top)

    chart = top.sort_values(
        "Rendimiento 5Y",
        ascending=True,
    ).copy()

    chart["Rendimiento (%)"] = chart["Rendimiento 5Y"] * 100

    fig = px.bar(
        chart,
        x="Rendimiento (%)",
        y="Ticker",
        orientation="h",
        hover_name="Nombre",
        title=f"Rendimiento acumulado de 5 años — {sector_label}",
    )

    fig.update_layout(
        xaxis_title="Rendimiento acumulado (%)",
        yaxis_title="",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# VALIDACIÓN
# ============================================================
if len(selected) != 2:
    st.error(
        "No se obtuvieron los dos rankings completos. "
        "No se construirá Markowitz con un universo parcial."
    )
    st.stop()


# ============================================================
# UNIVERSO FINAL
# ============================================================
combined = pd.concat(
    selected,
    ignore_index=True,
)

st.divider()
st.header("📊 Universo final para Markowitz")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Total",
    len(combined),
)

c2.metric(
    "Tecnología",
    (combined["Sector"] == "Tecnología").sum(),
)

c3.metric(
    "Industriales",
    (combined["Sector"] == "Industriales").sum(),
)

final = combined[
    [
        "Ticker",
        "Nombre",
        "Sector",
        "Fondo",
        "Rendimiento 5Y",
        "CAGR",
        "Volatilidad anual",
    ]
].copy()

for col in [
    "Rendimiento 5Y",
    "CAGR",
    "Volatilidad anual",
]:
    final[col] *= 100

st.dataframe(
    final,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rendimiento 5Y": st.column_config.NumberColumn(
            "Rendimiento 5 años", format="%.2f %%"
        ),
        "CAGR": st.column_config.NumberColumn(
            "CAGR anual", format="%.2f %%"
        ),
        "Volatilidad anual": st.column_config.NumberColumn(
            "Volatilidad anual", format="%.2f %%"
        ),
    },
)

csv = final.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "⬇️ Descargar universo final CSV",
    data=csv,
    file_name="universo_markowitz.csv",
    mime="text/csv",
    use_container_width=True,
)

st.success(
    "✅ Ambos sectores están completos. "
    "El universo está listo para la etapa Markowitz."
)

st.warning(
    "Limitación metodológica: los universos XLK/XLI son los holdings actuales. "
    "Seleccionar retrospectivamente las acciones que más subieron en 5 años "
    "introduce sesgo retrospectivo y de supervivencia."
)
