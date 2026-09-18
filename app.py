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
    "Fase 1: seleccionar las acciones de mayor rendimiento de los últimos 5 años."
)

TOP_N = 10
MIN_COVERAGE = 0.90

SECTORS = {
    "Information Technology": "Tecnología",
    "Industrials": "Industriales",
}

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# ============================================================
# UNIVERSO DE ACCIONES
# ============================================================
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_sp500_universe() -> pd.DataFrame:
    """
    Obtiene los componentes actuales del S&P 500 y su sector GICS.

    Se usa esta fuente únicamente para definir el universo y los sectores.
    Los precios históricos se descargan desde Yahoo Finance.
    """
    tables = pd.read_html(SP500_URL)
    df = tables[0].copy()

    required = {"Symbol", "Security", "GICS Sector"}
    if not required.issubset(df.columns):
        raise ValueError("No se encontraron las columnas esperadas del S&P 500.")

    out = df[["Symbol", "Security", "GICS Sector"]].copy()
    out.columns = ["Ticker", "Nombre", "Sector"]

    # Yahoo Finance usa guion en símbolos como BRK-B en vez de BRK.B.
    out["Ticker"] = out["Ticker"].astype(str).str.replace(".", "-", regex=False)

    return out


# ============================================================
# DESCARGA YAHOO FINANCE
# ============================================================
def extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Yahoo Finance no devolvió precios de cierre.")
        close = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise ValueError("Yahoo Finance no devolvió precios de cierre.")
        close = raw[["Close"]].copy()
        if len(tickers) == 1:
            close.columns = tickers

    if isinstance(close, pd.Series):
        close = close.to_frame()

    return close.sort_index()


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def download_prices(tickers_tuple: tuple[str, ...], period: str = "5y") -> pd.DataFrame:
    tickers = list(tickers_tuple)

    if not tickers:
        return pd.DataFrame()

    frames = []

    # Bloques pequeños = menos riesgo de error/rate limit en Yahoo.
    for i in range(0, len(tickers), 60):
        batch = tickers[i:i + 60]

        try:
            raw = yf.download(
                batch,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
                timeout=20,
            )
            close = extract_close(raw, batch)

            if not close.empty:
                frames.append(close)

        except Exception:
            # Si un bloque falla, intentamos uno por uno.
            singles = []

            for ticker in batch:
                try:
                    raw = yf.download(
                        ticker,
                        period=period,
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                        timeout=20,
                    )
                    close = extract_close(raw, [ticker])

                    if not close.empty:
                        singles.append(close)

                except Exception:
                    pass

            if singles:
                frames.append(pd.concat(singles, axis=1))

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]

    return prices.sort_index()


# ============================================================
# CÁLCULOS
# ============================================================
def calculate_ranking(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    min_coverage: float = 0.90,
) -> pd.DataFrame:

    if prices.empty:
        return pd.DataFrame()

    obs = prices.notna().sum()
    max_obs = int(obs.max())

    rows = []

    for ticker in prices.columns:
        serie = prices[ticker].dropna()

        if len(serie) < max_obs * min_coverage:
            continue

        if len(serie) < 2:
            continue

        p0 = float(serie.iloc[0])
        p1 = float(serie.iloc[-1])

        if p0 <= 0:
            continue

        total_return = p1 / p0 - 1

        days = max((serie.index[-1] - serie.index[0]).days, 1)
        years = days / 365.25

        cagr = (p1 / p0) ** (1 / years) - 1

        daily = serie.pct_change(fill_method=None).dropna()
        vol = daily.std() * np.sqrt(252)

        rows.append({
            "Ticker": ticker,
            "Precio inicial": p0,
            "Precio final": p1,
            "Rendimiento 5Y": total_return,
            "CAGR": cagr,
            "Volatilidad anual": vol,
            "Fecha inicial": serie.index[0].date(),
            "Fecha final": serie.index[-1].date(),
            "Observaciones": len(serie),
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result = result.merge(
        metadata[["Ticker", "Nombre", "Sector"]],
        on="Ticker",
        how="left",
    )

    return result.sort_values(
        "Rendimiento 5Y",
        ascending=False
    ).reset_index(drop=True)


def show_table(df: pd.DataFrame):
    display = df.copy()
    display.insert(0, "Posición", range(1, len(display) + 1))

    display["Rendimiento 5Y"] *= 100
    display["CAGR"] *= 100
    display["Volatilidad anual"] *= 100

    display = display[
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

    st.dataframe(
        display,
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
        },
    )


# ============================================================
# INTERFAZ
# ============================================================
st.sidebar.header("Configuración")

top_n = st.sidebar.slider(
    "Acciones por sector",
    5,
    20,
    TOP_N,
)

st.sidebar.write("**Periodo:** 5 años")
st.sidebar.write("**Cobertura mínima:** 90%")
st.sidebar.caption(
    "La cobertura se mantiene fija para evitar incluir empresas "
    "sin suficiente historial de 5 años."
)

st.info(
    "El universo se limita al **S&P 500**. La clasificación sectorial se usa "
    "para seleccionar Tecnología e Industriales; los precios históricos se "
    "obtienen de Yahoo Finance."
)

if not st.button(
    "🚀 Analizar acciones",
    type="primary",
    use_container_width=True,
):
    st.subheader("Qué hará la aplicación")
    st.markdown(
        """
        1. Obtener las empresas actuales del **S&P 500**.
        2. Separar **Tecnología** e **Industriales**.
        3. Descargar sus precios de los últimos **5 años desde Yahoo Finance**.
        4. Calcular el rendimiento acumulado de cada acción.
        5. Elegir las **Top acciones de cada sector**.
        6. Preparar esas acciones para el modelo de **Markowitz**.
        """
    )
    st.stop()


# ============================================================
# EJECUCIÓN
# ============================================================
try:
    with st.spinner("Obteniendo componentes del S&P 500..."):
        universe = get_sp500_universe()

except Exception as e:
    st.error(
        "No fue posible obtener la lista del S&P 500. "
        "Intenta nuevamente en unos minutos."
    )
    st.exception(e)
    st.stop()


selected_all = []

for sector_code, sector_name in SECTORS.items():

    sector_universe = universe[
        universe["Sector"] == sector_code
    ].copy()

    st.divider()
    st.header(f"🏢 {sector_name}")

    st.write(
        f"Empresas del S&P 500 identificadas en el sector: "
        f"**{len(sector_universe)}**"
    )

    with st.spinner(
        f"Descargando 5 años de precios de {sector_name} desde Yahoo Finance..."
    ):
        prices = download_prices(
            tuple(sector_universe["Ticker"].tolist()),
            period="5y",
        )

    if prices.empty:
        st.error(
            f"Yahoo Finance no devolvió datos para {sector_name}. "
            "Puede tratarse de un bloqueo temporal o límite de solicitudes."
        )
        continue

    ranking = calculate_ranking(
        prices,
        sector_universe,
        min_coverage=MIN_COVERAGE,
    )

    if ranking.empty:
        st.warning(
            f"No se pudieron calcular suficientes acciones válidas "
            f"para {sector_name}."
        )
        continue

    top = ranking.head(top_n).copy()
    top["Sector visible"] = sector_name
    selected_all.append(top)

    st.subheader(f"🏆 Top {top_n}")

    show_table(top)

    graph = top.sort_values(
        "Rendimiento 5Y",
        ascending=True
    ).copy()

    graph["Rendimiento (%)"] = graph["Rendimiento 5Y"] * 100

    fig = px.bar(
        graph,
        x="Rendimiento (%)",
        y="Ticker",
        orientation="h",
        hover_name="Nombre",
        title=f"Rendimiento acumulado de 5 años — {sector_name}",
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
# UNIVERSO FINAL
# ============================================================
if not selected_all:
    st.error(
        "No se pudo formar el universo final. "
        "Revisa la conexión de Streamlit Cloud e inténtalo nuevamente."
    )
    st.stop()


combined = pd.concat(
    selected_all,
    ignore_index=True
)

st.divider()
st.header("📊 Universo final para Markowitz")

m1, m2, m3 = st.columns(3)

m1.metric(
    "Total de acciones",
    len(combined),
)

m2.metric(
    "Tecnología",
    (combined["Sector visible"] == "Tecnología").sum(),
)

m3.metric(
    "Industriales",
    (combined["Sector visible"] == "Industriales").sum(),
)

final_table = combined[
    [
        "Ticker",
        "Nombre",
        "Sector visible",
        "Rendimiento 5Y",
        "CAGR",
        "Volatilidad anual",
    ]
].copy()

final_table["Rendimiento 5Y"] *= 100
final_table["CAGR"] *= 100
final_table["Volatilidad anual"] *= 100

st.dataframe(
    final_table,
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

st.success(
    "✅ Las acciones ya están seleccionadas. "
    "El siguiente paso es aplicar Markowitz sobre este universo."
)

st.warning(
    "Nota metodológica: esta selección es retrospectiva. "
    "Elegir hoy las acciones que más ganaron durante el periodo introduce "
    "sesgo retrospectivo y de supervivencia. Esto debe señalarse en las "
    "conclusiones del trabajo."
)
