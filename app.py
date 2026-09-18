import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="Markowitz — Tecnología + Industriales",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Markowitz — Tecnología + Industriales")
st.caption(
    "Selección automática de acciones de mayor rendimiento en los últimos 5 años."
)

TOP_N_DEFAULT = 10
MIN_COVERAGE = 0.90

# DataHub: CSV estable de constituyentes S&P 500
DATAHUB_URL = (
    "https://datahub.io/core/s-and-p-500-companies-financials/"
    "_r/-/data/constituents.csv"
)

SECTORS = {
    "Information Technology": "Tecnología",
    "Industrials": "Industriales",
}


# ============================================================
# UNIVERSO DEL S&P 500
# ============================================================
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_sp500_universe() -> pd.DataFrame:
    """
    Obtiene símbolos, nombres y sectores del S&P 500 desde DataHub.

    Esta función reemplaza pd.read_html(Wikipedia), que puede devolver
    HTTP 403 desde Streamlit Cloud.
    """
    df = pd.read_csv(DATAHUB_URL)

    expected = {"Symbol", "Name", "Sector"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"El CSV no contiene las columnas esperadas. "
            f"Columnas recibidas: {list(df.columns)}"
        )

    df = df[["Symbol", "Name", "Sector"]].copy()
    df.columns = ["Ticker", "Nombre", "Sector"]

    # Convención de Yahoo Finance para tickers con punto.
    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
    )

    df["Nombre"] = df["Nombre"].astype(str).str.strip()
    df["Sector"] = df["Sector"].astype(str).str.strip()

    return df.drop_duplicates("Ticker").reset_index(drop=True)


# ============================================================
# YAHOO FINANCE
# ============================================================
def extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extrae Close de la estructura que devuelve yf.download()."""
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


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def download_prices(tickers_tuple: tuple[str, ...]) -> pd.DataFrame:
    """
    Descarga Tecnología + Industriales en una sola rutina.

    Los tickers se dividen en bloques moderados para reducir la probabilidad
    de límites temporales de Yahoo. Si un bloque falla, se intenta por mitades.
    """
    tickers = list(tickers_tuple)

    if not tickers:
        return pd.DataFrame()

    frames = []

    def fetch(batch):
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

    batch_size = 40

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        try:
            close = fetch(batch)

            if not close.empty:
                frames.append(close)

        except Exception:
            # Si el bloque falla, intentar dos grupos más pequeños.
            midpoint = max(len(batch) // 2, 1)

            for smaller in [batch[:midpoint], batch[midpoint:]]:
                if not smaller:
                    continue

                try:
                    close = fetch(smaller)
                    if not close.empty:
                        frames.append(close)
                except Exception:
                    # No tiramos toda la aplicación por unos cuantos símbolos.
                    pass

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index()

    return prices


# ============================================================
# RANKING
# ============================================================
def calculate_ranking(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    min_coverage: float = 0.90,
) -> pd.DataFrame:

    if prices.empty:
        return pd.DataFrame()

    available = [
        ticker for ticker in metadata["Ticker"].tolist()
        if ticker in prices.columns
    ]

    if not available:
        return pd.DataFrame()

    sector_prices = prices[available]

    observation_counts = sector_prices.notna().sum()
    max_obs = int(observation_counts.max())

    rows = []

    for ticker in available:
        serie = sector_prices[ticker].dropna()

        if len(serie) < 2:
            continue

        coverage = len(serie) / max_obs if max_obs else 0

        if coverage < min_coverage:
            continue

        p0 = float(serie.iloc[0])
        p1 = float(serie.iloc[-1])

        if p0 <= 0:
            continue

        total_return = p1 / p0 - 1.0

        days = max(
            (serie.index[-1] - serie.index[0]).days,
            1
        )
        years = days / 365.25

        cagr = (p1 / p0) ** (1 / years) - 1

        daily = serie.pct_change(fill_method=None).dropna()

        annual_vol = (
            daily.std() * np.sqrt(252)
            if len(daily) > 1
            else np.nan
        )

        rows.append(
            {
                "Ticker": ticker,
                "Rendimiento 5Y": total_return,
                "CAGR": cagr,
                "Volatilidad anual": annual_vol,
                "Precio inicial": p0,
                "Precio final": p1,
                "Fecha inicial": serie.index[0].date(),
                "Fecha final": serie.index[-1].date(),
                "Cobertura": coverage,
            }
        )

    ranking = pd.DataFrame(rows)

    if ranking.empty:
        return ranking

    ranking = ranking.merge(
        metadata[["Ticker", "Nombre", "Sector"]],
        on="Ticker",
        how="left",
    )

    return (
        ranking
        .sort_values("Rendimiento 5Y", ascending=False)
        .reset_index(drop=True)
    )


def display_ranking(df: pd.DataFrame):
    show = df.copy()

    show.insert(0, "Posición", range(1, len(show) + 1))
    show["Rendimiento 5Y"] *= 100
    show["CAGR"] *= 100
    show["Volatilidad anual"] *= 100
    show["Cobertura"] *= 100

    show = show[
        [
            "Posición",
            "Ticker",
            "Nombre",
            "Rendimiento 5Y",
            "CAGR",
            "Volatilidad anual",
            "Precio inicial",
            "Precio final",
            "Cobertura",
        ]
    ]

    st.dataframe(
        show,
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
            "Precio inicial": st.column_config.NumberColumn(
                "Precio inicial", format="$ %.2f"
            ),
            "Precio final": st.column_config.NumberColumn(
                "Precio final", format="$ %.2f"
            ),
            "Cobertura": st.column_config.NumberColumn(
                "Cobertura", format="%.1f %%"
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

st.sidebar.caption(
    "La cobertura mínima queda fija para impedir que una empresa "
    "con poco historial compita como si tuviera 5 años completos."
)

# ============================================================
# INICIO
# ============================================================
if not st.button(
    "🚀 Analizar acciones",
    type="primary",
    use_container_width=True,
):
    st.subheader("Metodología")
    st.markdown(
        """
        1. Obtener componentes del **S&P 500** y su sector.
        2. Tomar **Information Technology** e **Industrials**.
        3. Descargar **5 años de precios desde Yahoo Finance**.
        4. Calcular rendimiento acumulado, CAGR y volatilidad.
        5. Seleccionar las acciones de mayor rendimiento de cada sector.
        6. Usar el universo resultante para **Markowitz**.
        """
    )
    st.stop()


# ============================================================
# OBTENER UNIVERSO
# ============================================================
try:
    with st.spinner("Obteniendo universo del S&P 500..."):
        universe = get_sp500_universe()

except Exception as e:
    st.error(
        "No fue posible obtener el universo del S&P 500 desde DataHub."
    )
    st.code(str(e))
    st.stop()


# Seleccionar únicamente los dos sectores.
sector_universe = universe[
    universe["Sector"].isin(SECTORS.keys())
].copy()

tech_count = (
    sector_universe["Sector"] == "Information Technology"
).sum()

ind_count = (
    sector_universe["Sector"] == "Industrials"
).sum()

st.success(
    f"Universo cargado: {tech_count} empresas de Tecnología "
    f"y {ind_count} de Industriales."
)

if tech_count == 0 or ind_count == 0:
    st.error(
        "La fuente no devolvió correctamente ambos sectores. "
        "No se continuará para evitar un análisis incompleto."
    )
    st.write(
        "Sectores disponibles:",
        sorted(universe["Sector"].dropna().unique().tolist())
    )
    st.stop()


# ============================================================
# UNA SOLA RUTINA DE DESCARGA
# ============================================================
tickers = tuple(
    sector_universe["Ticker"]
    .drop_duplicates()
    .tolist()
)

with st.spinner(
    f"Descargando 5 años de precios para {len(tickers)} acciones desde Yahoo Finance..."
):
    prices = download_prices(tickers)

if prices.empty:
    st.error(
        "Yahoo Finance no devolvió precios. "
        "Puede tratarse de un límite temporal de solicitudes."
    )
    st.stop()


st.caption(
    f"Yahoo devolvió datos para {len(prices.columns)} de "
    f"{len(tickers)} símbolos solicitados."
)


# ============================================================
# SEPARAR Y MOSTRAR LOS DOS RANKINGS
# ============================================================
selected = []

for sector_code, sector_label in SECTORS.items():

    meta = sector_universe[
        sector_universe["Sector"] == sector_code
    ].copy()

    ranking = calculate_ranking(
        prices,
        meta,
        MIN_COVERAGE,
    )

    st.divider()
    st.header(f"🏆 {sector_label}")

    if ranking.empty:
        st.error(
            f"No se pudo construir el ranking de {sector_label}."
        )
        continue

    top = ranking.head(top_n).copy()
    top["Sector visible"] = sector_label
    selected.append(top)

    st.write(
        f"Acciones válidas con al menos 90% de cobertura: "
        f"**{len(ranking)}**"
    )

    display_ranking(top)

    chart_df = top.sort_values(
        "Rendimiento 5Y",
        ascending=True,
    ).copy()

    chart_df["Rendimiento (%)"] = (
        chart_df["Rendimiento 5Y"] * 100
    )

    fig = px.bar(
        chart_df,
        x="Rendimiento (%)",
        y="Ticker",
        orientation="h",
        hover_name="Nombre",
        title=f"Rendimiento acumulado a 5 años — {sector_label}",
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
# VALIDACIÓN: NO ACEPTAR SOLO UN SECTOR
# ============================================================
if len(selected) != 2:
    st.error(
        "El análisis no obtuvo los dos sectores completos. "
        "No se formará el universo de Markowitz hasta tener ambos."
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
    (combined["Sector visible"] == "Tecnología").sum(),
)

c3.metric(
    "Industriales",
    (combined["Sector visible"] == "Industriales").sum(),
)

final = combined[
    [
        "Ticker",
        "Nombre",
        "Sector visible",
        "Rendimiento 5Y",
        "CAGR",
        "Volatilidad anual",
    ]
].copy()

final["Rendimiento 5Y"] *= 100
final["CAGR"] *= 100
final["Volatilidad anual"] *= 100

st.dataframe(
    final,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Sector visible": "Sector",
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

csv = final.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "⬇️ Descargar universo final CSV",
    csv,
    "universo_markowitz.csv",
    "text/csv",
    use_container_width=True,
)

st.success(
    "✅ Se obtuvieron ambos sectores. "
    "Este universo ya está listo para aplicar Markowitz."
)

st.warning(
    "Nota académica: seleccionar hoy las acciones que más subieron "
    "durante los últimos 5 años introduce sesgo retrospectivo y "
    "sesgo de supervivencia. Debe mencionarse como limitación."
)
