import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
from yfinance import EquityQuery

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Markowitz — Technology + Industrials",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Markowitz — Technology + Industrials")
st.caption(
    "Fase 1: selección automática de las acciones con mayor rendimiento "
    "acumulado usando datos de Yahoo Finance."
)

SECTORS = {
    "Technology": "Tecnología",
    "Industrials": "Industriales",
}

# ============================================================
# FUNCIONES
# ============================================================

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def get_sector_tickers(sector: str, max_pages: int = 6) -> pd.DataFrame:
    """
    Obtiene acciones de EE.UU. del sector indicado mediante el screener
    de Yahoo Finance/yfinance. Se pagina en bloques de hasta 250 resultados.

    Nota:
    - Se restringe a NYSE/Nasdaq para evitar OTC y otros mercados.
    - Yahoo puede cambiar la disponibilidad de campos o límites.
    """
    query = EquityQuery(
        "and",
        [
            EquityQuery("eq", ["region", "us"]),
            EquityQuery("eq", ["sector", sector]),
            EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
        ],
    )

    rows = []
    page_size = 250

    for page in range(max_pages):
        offset = page * page_size

        result = yf.screen(
            query,
            offset=offset,
            size=page_size,
            sortField="ticker",
            sortAsc=True,
        )

        quotes = result.get("quotes", []) if isinstance(result, dict) else []

        if not quotes:
            break

        rows.extend(quotes)

        if len(quotes) < page_size:
            break

        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(columns=["Ticker", "Nombre", "Sector"])

    data = pd.DataFrame(rows)

    symbol_col = "symbol"
    name_col = "shortName" if "shortName" in data.columns else None

    out = pd.DataFrame()
    out["Ticker"] = data[symbol_col].astype(str)
    out["Nombre"] = data[name_col].astype(str) if name_col else ""
    out["Sector"] = sector

    # Evita símbolos problemáticos y duplicados.
    out = out[
        out["Ticker"].notna()
        & ~out["Ticker"].str.contains(r"[\^=/]", regex=True)
    ].drop_duplicates("Ticker")

    return out.reset_index(drop=True)


def _extract_close(downloaded: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normaliza la salida de yf.download para uno o varios tickers."""
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        level0 = downloaded.columns.get_level_values(0)
        if "Close" in level0:
            close = downloaded["Close"].copy()
        else:
            raise ValueError("Yahoo Finance no devolvió la columna Close.")
    else:
        if "Close" not in downloaded.columns:
            raise ValueError("Yahoo Finance no devolvió la columna Close.")
        close = downloaded[["Close"]].copy()
        if len(tickers) == 1:
            close.columns = tickers

    if isinstance(close, pd.Series):
        close = close.to_frame()

    close = close.sort_index()
    return close


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def download_prices(tickers_tuple: tuple[str, ...], period: str = "5y") -> pd.DataFrame:
    """
    Descarga precios diarios ajustados.
    Se hace por bloques para evitar solicitudes excesivamente grandes.
    """
    tickers = list(tickers_tuple)

    if not tickers:
        return pd.DataFrame()

    frames = []
    batch_size = 100

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        raw = yf.download(
            tickers=batch,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )

        close = _extract_close(raw, batch)
        frames.append(close)

        if i + batch_size < len(tickers):
            time.sleep(0.15)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices.sort_index()


def calculate_ranking(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    min_coverage: float = 0.90,
) -> pd.DataFrame:
    """
    Calcula el rendimiento acumulado durante la ventana disponible de 5 años.

    Para ser considerada '5 años', la acción debe tener al menos min_coverage
    de las observaciones del calendario bursátil disponible en la matriz.
    """
    if prices.empty:
        return pd.DataFrame()

    max_obs = prices.notna().sum().max()
    min_obs = int(max_obs * min_coverage)

    results = []

    for ticker in prices.columns:
        s = prices[ticker].dropna()

        if len(s) < min_obs or len(s) < 2:
            continue

        initial = float(s.iloc[0])
        final = float(s.iloc[-1])

        if initial <= 0:
            continue

        total_return = final / initial - 1.0

        # CAGR usando días naturales observados.
        days = max((s.index[-1] - s.index[0]).days, 1)
        years = days / 365.25
        cagr = (final / initial) ** (1 / years) - 1 if years > 0 else np.nan

        daily_ret = s.pct_change().dropna()
        annual_vol = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 1 else np.nan

        results.append(
            {
                "Ticker": ticker,
                "Fecha inicial": s.index[0].date(),
                "Fecha final": s.index[-1].date(),
                "Precio inicial": initial,
                "Precio final": final,
                "Rendimiento 5Y": total_return,
                "CAGR": cagr,
                "Volatilidad anual": annual_vol,
                "Observaciones": len(s),
                "Cobertura": len(s) / max_obs if max_obs else np.nan,
            }
        )

    ranking = pd.DataFrame(results)

    if ranking.empty:
        return ranking

    ranking = ranking.merge(
        metadata[["Ticker", "Nombre", "Sector"]],
        on="Ticker",
        how="left",
    )

    return ranking.sort_values("Rendimiento 5Y", ascending=False).reset_index(drop=True)


def format_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Versión amigable de la tabla para Streamlit."""
    out = df.copy()
    out.insert(0, "Posición", range(1, len(out) + 1))

    return out[
        [
            "Posición",
            "Ticker",
            "Nombre",
            "Rendimiento 5Y",
            "CAGR",
            "Volatilidad anual",
            "Precio inicial",
            "Precio final",
            "Fecha inicial",
            "Fecha final",
        ]
    ]


# ============================================================
# BARRA LATERAL
# ============================================================
st.sidebar.header("Configuración")

top_n = st.sidebar.slider(
    "Acciones por sector",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
)

min_coverage_pct = st.sidebar.slider(
    "Cobertura mínima del periodo",
    min_value=70,
    max_value=100,
    value=90,
    step=5,
    help=(
        "Evita considerar como '5 años' una empresa que comenzó a cotizar "
        "mucho después del inicio del periodo."
    ),
)

st.sidebar.info(
    "Periodo de esta primera versión: 5 años, frecuencia diaria. "
    "Más adelante añadiremos Markowitz y la frontera eficiente."
)

# ============================================================
# EJECUCIÓN
# ============================================================
run = st.button("🚀 Obtener Top acciones", type="primary", use_container_width=True)

if not run:
    st.subheader("Objetivo")
    st.write(
        "Seleccionar automáticamente las acciones con mayor rendimiento acumulado "
        "de **Technology** e **Industrials**, usando Yahoo Finance. "
        "Estas acciones serán la entrada para el modelo de Markowitz."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Sectores", "2")
    c2.metric("Periodo", "5 años")
    c3.metric("Selección", f"Top {top_n} por sector")

    st.markdown(
        """
        **Metodología de esta fase**

        1. Consultar empresas estadounidenses de Technology e Industrials.
        2. Descargar precios diarios ajustados de los últimos 5 años.
        3. Excluir acciones sin suficiente historial.
        4. Calcular rendimiento acumulado, CAGR y volatilidad anual.
        5. Seleccionar las mejores acciones de cada sector.
        """
    )

    st.warning(
        "Importante: este es un análisis retrospectivo. Elegir hoy las acciones "
        "que ya sabemos que fueron ganadoras introduce sesgo retrospectivo y "
        "sesgo de supervivencia; se documentará como limitación del estudio."
    )

    st.stop()


all_top = []
full_rankings = {}

for sector_key, sector_label in SECTORS.items():
    with st.spinner(f"Consultando universo de {sector_label}..."):
        universe = get_sector_tickers(sector_key)

    if universe.empty:
        st.error(f"No se pudo obtener el universo de {sector_label}.")
        continue

    st.caption(
        f"{sector_label}: {len(universe):,} símbolos recuperados del screener."
    )

    with st.spinner(f"Descargando 5 años de precios de {sector_label}..."):
        prices = download_prices(tuple(universe["Ticker"].tolist()), period="5y")

    ranking = calculate_ranking(
        prices,
        universe,
        min_coverage=min_coverage_pct / 100,
    )

    if ranking.empty:
        st.error(
            f"No hubo suficientes datos válidos para construir el ranking de {sector_label}."
        )
        continue

    full_rankings[sector_key] = ranking
    top = ranking.head(top_n).copy()
    top["Sector visible"] = sector_label
    all_top.append(top)

    st.divider()
    st.header(f"🏆 Top {top_n} — {sector_label}")

    top_display = format_ranking(top)
    top_display["Rendimiento 5Y"] = top_display["Rendimiento 5Y"] * 100
    top_display["CAGR"] = top_display["CAGR"] * 100
    top_display["Volatilidad anual"] = top_display["Volatilidad anual"] * 100

    st.dataframe(
        top_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rendimiento 5Y": st.column_config.NumberColumn(
                "Rendimiento 5Y", format="%.2f%%"
            ),
            "CAGR": st.column_config.NumberColumn(
                "CAGR anual", format="%.2f%%"
            ),
            "Volatilidad anual": st.column_config.NumberColumn(
                "Volatilidad anual", format="%.2f%%"
            ),
            "Precio inicial": st.column_config.NumberColumn(
                "Precio inicial", format="$%.2f"
            ),
            "Precio final": st.column_config.NumberColumn(
                "Precio final", format="$%.2f"
            ),
        },
    )

    # Streamlit espera decimales reales para porcentajes; para Plotly convertimos.
    chart_data = top.sort_values("Rendimiento 5Y", ascending=True).copy()
    chart_data["Rendimiento (%)"] = chart_data["Rendimiento 5Y"] * 100

    fig = px.bar(
        chart_data,
        x="Rendimiento (%)",
        y="Ticker",
        orientation="h",
        hover_data=["Nombre", "CAGR", "Volatilidad anual"],
        title=f"Rendimiento acumulado a 5 años — {sector_label}",
    )
    fig.update_layout(yaxis_title="", xaxis_title="Rendimiento acumulado (%)")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RESUMEN COMBINADO
# ============================================================
if all_top:
    combined = pd.concat(all_top, ignore_index=True)

    st.divider()
    st.header("📊 Universo final para Markowitz")

    c1, c2, c3 = st.columns(3)
    c1.metric("Acciones seleccionadas", len(combined))
    c2.metric("Tecnología", (combined["Sector"] == "Technology").sum())
    c3.metric("Industriales", (combined["Sector"] == "Industrials").sum())

    combined_display = combined[
        [
            "Ticker",
            "Nombre",
            "Sector visible",
            "Rendimiento 5Y",
            "CAGR",
            "Volatilidad anual",
        ]
    ].copy()

    combined_display["Rendimiento 5Y"] = combined_display["Rendimiento 5Y"] * 100
    combined_display["CAGR"] = combined_display["CAGR"] * 100
    combined_display["Volatilidad anual"] = combined_display["Volatilidad anual"] * 100

    st.dataframe(
        combined_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sector visible": "Sector",
            "Rendimiento 5Y": st.column_config.NumberColumn(
                "Rendimiento 5Y", format="%.2f%%"
            ),
            "CAGR": st.column_config.NumberColumn(
                "CAGR anual", format="%.2f%%"
            ),
            "Volatilidad anual": st.column_config.NumberColumn(
                "Volatilidad anual", format="%.2f%%"
            ),
        },
    )

    csv = combined_display.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Descargar selección CSV",
        data=csv,
        file_name="top_acciones_markowitz.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.success(
        "Esta selección ya queda lista para la Fase 2: rendimientos diarios, "
        "matriz de covarianzas, correlaciones, portafolios aleatorios, "
        "mínima varianza, máximo Sharpe y frontera eficiente."
    )

    with st.expander("Limitaciones metodológicas que conviene mencionar en la tarea"):
        st.markdown(
            """
            - **Sesgo retrospectivo:** seleccionamos las ganadoras después de conocer su desempeño.
            - **Sesgo de supervivencia:** el screener refleja principalmente empresas que cotizan actualmente.
            - **Historia mínima:** se exige cobertura suficiente para evitar comparar un IPO reciente con una acción que sí tiene cinco años completos.
            - **Rendimiento histórico ≠ rendimiento futuro:** Markowitz describe relaciones históricas de rendimiento, volatilidad y covarianza; no garantiza resultados futuros.
            """
        )
