import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="Markowitz — Tecnología + Industriales",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Modelo de Markowitz — Tecnología + Industriales")
st.caption(
    "20 acciones seleccionadas por rendimiento histórico de 5 años: "
    "10 de Tecnología y 10 de Industriales."
)

TRADING_DAYS = 252

# Universo obtenido en la Fase 1.
DEFAULT_UNIVERSE = pd.DataFrame([
    ["MU",   "MICRON TECHNOLOGY INC",         "Tecnología"],
    ["DELL", "DELL TECHNOLOGIES C",           "Tecnología"],
    ["STX",  "SEAGATE TECHNOLOGY HOLDINGS",   "Tecnología"],
    ["SMCI", "SUPER MICRO COMPUTER INC",       "Tecnología"],
    ["LITE", "LUMENTUM HOLDINGS INC",          "Tecnología"],
    ["NVDA", "NVIDIA CORP",                    "Tecnología"],
    ["WDC",  "WESTERN DIGITAL CORP",           "Tecnología"],
    ["ANET", "ARISTA NETWORKS INC",            "Tecnología"],
    ["FLEX", "FLEX LTD",                       "Tecnología"],
    ["AVGO", "BROADCOM INC",                   "Tecnología"],
    ["FIX",  "COMFORT SYSTEMS USA INC",        "Industriales"],
    ["VRT",  "VERTIV HOLDINGS CO A",           "Industriales"],
    ["HWM",  "HOWMET AEROSPACE INC",           "Industriales"],
    ["EME",  "EMCOR GROUP INC",                "Industriales"],
    ["PWR",  "QUANTA SERVICES INC",            "Industriales"],
    ["GE",   "GENERAL ELECTRIC",                "Industriales"],
    ["CAT",  "CATERPILLAR INC",                 "Industriales"],
    ["PH",   "PARKER HANNIFIN CORP",            "Industriales"],
    ["WAB",  "WABTEC CORP",                    "Industriales"],
    ["GWW",  "WW GRAINGER INC",                "Industriales"],
], columns=["Ticker", "Nombre", "Sector"])


# ============================================================
# DESCARGA DE PRECIOS
# ============================================================
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def download_prices(tickers_tuple):
    tickers = list(tickers_tuple)

    raw = yf.download(
        tickers=tickers,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
        timeout=30,
    )

    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        if len(tickers) == 1:
            prices.columns = tickers

    prices = prices.sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]

    return prices


# ============================================================
# FUNCIONES MARKOWITZ
# ============================================================
def annual_statistics(prices):
    returns = prices.pct_change(fill_method=None).dropna(how="all")

    # Quitar días incompletos para que covarianzas usen fechas comunes.
    returns = returns.dropna(axis=0, how="any")

    mean_returns = returns.mean() * TRADING_DAYS
    cov_matrix = returns.cov() * TRADING_DAYS
    corr_matrix = returns.corr()

    return returns, mean_returns, cov_matrix, corr_matrix


def portfolio_performance(weights, mean_returns, cov_matrix, risk_free):
    weights = np.asarray(weights)

    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(weights.T @ cov_matrix.values @ weights))

    sharpe = (ret - risk_free) / vol if vol > 0 else np.nan

    return ret, vol, sharpe


def min_variance_portfolio(mean_returns, cov_matrix, risk_free):
    n = len(mean_returns)
    x0 = np.repeat(1 / n, n)

    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0
    },)

    result = minimize(
        lambda w: portfolio_performance(
            w, mean_returns, cov_matrix, risk_free
        )[1],
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError("No se pudo optimizar el portafolio de mínima varianza.")

    return result.x


def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free):
    n = len(mean_returns)
    x0 = np.repeat(1 / n, n)

    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0
    },)

    def negative_sharpe(w):
        _, vol, sharpe = portfolio_performance(
            w, mean_returns, cov_matrix, risk_free
        )
        if not np.isfinite(sharpe):
            return 1e9
        return -sharpe

    result = minimize(
        negative_sharpe,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError("No se pudo optimizar el portafolio de máximo Sharpe.")

    return result.x


def efficient_frontier(mean_returns, cov_matrix, risk_free, points=40):
    n = len(mean_returns)
    x0 = np.repeat(1 / n, n)

    min_w = min_variance_portfolio(
        mean_returns, cov_matrix, risk_free
    )
    min_ret, _, _ = portfolio_performance(
        min_w, mean_returns, cov_matrix, risk_free
    )

    max_asset_return = float(mean_returns.max())

    targets = np.linspace(min_ret, max_asset_return, points)

    rows = []

    for target in targets:
        constraints = (
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0
            },
            {
                "type": "eq",
                "fun": lambda w, t=target:
                    np.dot(w, mean_returns.values) - t
            },
        )

        bounds = tuple((0.0, 1.0) for _ in range(n))

        result = minimize(
            lambda w: portfolio_performance(
                w, mean_returns, cov_matrix, risk_free
            )[1],
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1500, "ftol": 1e-10},
        )

        if result.success:
            r, v, s = portfolio_performance(
                result.x, mean_returns, cov_matrix, risk_free
            )
            rows.append({
                "Rendimiento": r,
                "Volatilidad": v,
                "Sharpe": s,
            })

    return pd.DataFrame(rows)


def monte_carlo(mean_returns, cov_matrix, risk_free, simulations, seed=42):
    rng = np.random.default_rng(seed)
    n = len(mean_returns)

    # Dirichlet genera pesos positivos que suman 1.
    weights = rng.dirichlet(
        alpha=np.ones(n),
        size=simulations,
    )

    portfolio_returns = weights @ mean_returns.values

    covariance = cov_matrix.values

    variances = np.einsum(
        "ij,jk,ik->i",
        weights,
        covariance,
        weights,
    )

    volatilities = np.sqrt(variances)

    sharpes = np.divide(
        portfolio_returns - risk_free,
        volatilities,
        out=np.full_like(portfolio_returns, np.nan),
        where=volatilities > 0,
    )

    return pd.DataFrame({
        "Rendimiento": portfolio_returns,
        "Volatilidad": volatilities,
        "Sharpe": sharpes,
    })


def portfolio_weights_table(weights, universe):
    df = universe[["Ticker", "Nombre", "Sector"]].copy()
    df["Peso"] = np.asarray(weights)
    df = df[df["Peso"] > 0.0005]
    return df.sort_values("Peso", ascending=False).reset_index(drop=True)


def analyze_subset(tickers, prices, risk_free):
    p = prices[tickers].dropna(axis=0, how="any")
    _, mu, cov, corr = annual_statistics(p)

    w_min = min_variance_portfolio(mu, cov, risk_free)
    w_sharpe = max_sharpe_portfolio(mu, cov, risk_free)

    perf_min = portfolio_performance(w_min, mu, cov, risk_free)
    perf_sharpe = portfolio_performance(w_sharpe, mu, cov, risk_free)

    return {
        "mu": mu,
        "cov": cov,
        "corr": corr,
        "w_min": w_min,
        "w_sharpe": w_sharpe,
        "perf_min": perf_min,
        "perf_sharpe": perf_sharpe,
    }


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Configuración")

risk_free_pct = st.sidebar.number_input(
    "Tasa libre de riesgo anual (%)",
    min_value=0.0,
    max_value=20.0,
    value=4.0,
    step=0.25,
)

simulations = st.sidebar.select_slider(
    "Portafolios aleatorios",
    options=[10000, 25000, 50000, 100000],
    value=50000,
)

risk_free = risk_free_pct / 100.0

st.sidebar.write("**Periodo:** 5 años")
st.sidebar.write("**Activos:** 20")
st.sidebar.write("**Restricción:** sin ventas en corto")
st.sidebar.write("**Suma de pesos:** 100%")


# ============================================================
# UNIVERSO
# ============================================================
st.subheader("1. Universo seleccionado")

c1, c2, c3 = st.columns(3)
c1.metric("Total", 20)
c2.metric("Tecnología", 10)
c3.metric("Industriales", 10)

st.dataframe(
    DEFAULT_UNIVERSE,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DESCARGA
# ============================================================
tickers = DEFAULT_UNIVERSE["Ticker"].tolist()

with st.spinner("Descargando 5 años de precios desde Yahoo Finance..."):
    prices = download_prices(tuple(tickers))

missing = [t for t in tickers if t not in prices.columns]

if missing:
    st.error(
        "Yahoo Finance no devolvió datos para: "
        + ", ".join(missing)
    )
    st.stop()

# Eliminar acciones con demasiados huecos.
coverage = prices.notna().mean()
bad = coverage[coverage < 0.90].index.tolist()

if bad:
    st.error(
        "Estas acciones no tienen suficiente cobertura: "
        + ", ".join(bad)
    )
    st.stop()

prices = prices[tickers]

returns, mean_returns, cov_matrix, corr_matrix = annual_statistics(prices)

if len(returns) < 100:
    st.error("No hay suficientes observaciones comunes para realizar Markowitz.")
    st.stop()


# ============================================================
# ESTADÍSTICAS INDIVIDUALES
# ============================================================
st.divider()
st.subheader("2. Rendimiento y riesgo de las acciones")

stats = pd.DataFrame({
    "Ticker": mean_returns.index,
    "Rendimiento anual esperado": mean_returns.values,
    "Volatilidad anual": np.sqrt(np.diag(cov_matrix.values)),
})

stats = stats.merge(
    DEFAULT_UNIVERSE,
    on="Ticker",
    how="left",
)

stats_display = stats.copy()
stats_display["Rendimiento anual esperado"] *= 100
stats_display["Volatilidad anual"] *= 100

st.dataframe(
    stats_display[
        [
            "Ticker",
            "Nombre",
            "Sector",
            "Rendimiento anual esperado",
            "Volatilidad anual",
        ]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rendimiento anual esperado": st.column_config.NumberColumn(
            "Rendimiento anual esperado", format="%.2f %%"
        ),
        "Volatilidad anual": st.column_config.NumberColumn(
            "Volatilidad anual", format="%.2f %%"
        ),
    },
)


# ============================================================
# MATRIZ DE CORRELACIÓN
# ============================================================
st.divider()
st.subheader("3. Matriz de correlación")

fig_corr = px.imshow(
    corr_matrix,
    text_auto=".2f",
    aspect="auto",
    zmin=-1,
    zmax=1,
    title="Correlación entre rendimientos diarios",
)

st.plotly_chart(fig_corr, use_container_width=True)

with st.expander("Ver matriz de covarianzas anualizada"):
    st.dataframe(
        cov_matrix,
        use_container_width=True,
    )


# ============================================================
# OPTIMIZACIÓN COMBINADA
# ============================================================
st.divider()
st.subheader("4. Markowitz — 20 acciones combinadas")

try:
    w_min = min_variance_portfolio(
        mean_returns,
        cov_matrix,
        risk_free,
    )

    w_sharpe = max_sharpe_portfolio(
        mean_returns,
        cov_matrix,
        risk_free,
    )

    perf_min = portfolio_performance(
        w_min,
        mean_returns,
        cov_matrix,
        risk_free,
    )

    perf_sharpe = portfolio_performance(
        w_sharpe,
        mean_returns,
        cov_matrix,
        risk_free,
    )

except Exception as e:
    st.error(str(e))
    st.stop()


# Equal Weight
w_equal = np.repeat(1 / len(tickers), len(tickers))

perf_equal = portfolio_performance(
    w_equal,
    mean_returns,
    cov_matrix,
    risk_free,
)


# ============================================================
# MONTE CARLO + FRONTERA
# ============================================================
with st.spinner(
    f"Simulando {simulations:,} portafolios y calculando frontera eficiente..."
):
    mc = monte_carlo(
        mean_returns,
        cov_matrix,
        risk_free,
        simulations,
    )

    frontier = efficient_frontier(
        mean_returns,
        cov_matrix,
        risk_free,
        points=45,
    )


mc_plot = mc.copy()
mc_plot["Rendimiento (%)"] = mc_plot["Rendimiento"] * 100
mc_plot["Volatilidad (%)"] = mc_plot["Volatilidad"] * 100

fig = px.scatter(
    mc_plot,
    x="Volatilidad (%)",
    y="Rendimiento (%)",
    color="Sharpe",
    opacity=0.45,
    title=f"Simulación Monte Carlo — {simulations:,} portafolios",
    labels={
        "Volatilidad (%)": "Riesgo / volatilidad anual (%)",
        "Rendimiento (%)": "Rendimiento anual esperado (%)",
    },
)

if not frontier.empty:
    fig.add_trace(
        go.Scatter(
            x=frontier["Volatilidad"] * 100,
            y=frontier["Rendimiento"] * 100,
            mode="lines",
            name="Frontera eficiente",
            line=dict(width=4),
        )
    )

fig.add_trace(
    go.Scatter(
        x=[perf_min[1] * 100],
        y=[perf_min[0] * 100],
        mode="markers",
        marker=dict(size=16, symbol="diamond"),
        name="Mínima varianza",
    )
)

fig.add_trace(
    go.Scatter(
        x=[perf_sharpe[1] * 100],
        y=[perf_sharpe[0] * 100],
        mode="markers",
        marker=dict(size=18, symbol="star"),
        name="Máximo Sharpe",
    )
)

fig.add_trace(
    go.Scatter(
        x=[perf_equal[1] * 100],
        y=[perf_equal[0] * 100],
        mode="markers",
        marker=dict(size=14, symbol="circle-open"),
        name="Equal Weight",
    )
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# COMPARACIÓN DE PORTAFOLIOS
# ============================================================
st.subheader("5. Portafolios destacados")

comparison = pd.DataFrame([
    {
        "Portafolio": "Máximo Sharpe",
        "Rendimiento esperado": perf_sharpe[0],
        "Volatilidad": perf_sharpe[1],
        "Sharpe": perf_sharpe[2],
    },
    {
        "Portafolio": "Mínima varianza",
        "Rendimiento esperado": perf_min[0],
        "Volatilidad": perf_min[1],
        "Sharpe": perf_min[2],
    },
    {
        "Portafolio": "Equal Weight",
        "Rendimiento esperado": perf_equal[0],
        "Volatilidad": perf_equal[1],
        "Sharpe": perf_equal[2],
    },
])

comparison_display = comparison.copy()
comparison_display["Rendimiento esperado"] *= 100
comparison_display["Volatilidad"] *= 100

st.dataframe(
    comparison_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rendimiento esperado": st.column_config.NumberColumn(
            "Rendimiento esperado", format="%.2f %%"
        ),
        "Volatilidad": st.column_config.NumberColumn(
            "Volatilidad", format="%.2f %%"
        ),
        "Sharpe": st.column_config.NumberColumn(
            "Sharpe", format="%.3f"
        ),
    },
)


# ============================================================
# PESOS
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Pesos — Máximo Sharpe")
    weights_sharpe = portfolio_weights_table(
        w_sharpe,
        DEFAULT_UNIVERSE,
    )
    weights_sharpe["Peso"] *= 100

    st.dataframe(
        weights_sharpe,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Peso": st.column_config.NumberColumn(
                "Peso", format="%.2f %%"
            ),
        },
    )

with col2:
    st.markdown("### Pesos — Mínima varianza")
    weights_min = portfolio_weights_table(
        w_min,
        DEFAULT_UNIVERSE,
    )
    weights_min["Peso"] *= 100

    st.dataframe(
        weights_min,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Peso": st.column_config.NumberColumn(
                "Peso", format="%.2f %%"
            ),
        },
    )


# ============================================================
# COMPARACIÓN POR SECTOR
# ============================================================
st.divider()
st.subheader("6. ¿Qué aporta combinar los dos sectores?")

tech_tickers = DEFAULT_UNIVERSE.loc[
    DEFAULT_UNIVERSE["Sector"] == "Tecnología",
    "Ticker",
].tolist()

ind_tickers = DEFAULT_UNIVERSE.loc[
    DEFAULT_UNIVERSE["Sector"] == "Industriales",
    "Ticker",
].tolist()

try:
    tech_analysis = analyze_subset(
        tech_tickers,
        prices,
        risk_free,
    )

    ind_analysis = analyze_subset(
        ind_tickers,
        prices,
        risk_free,
    )

    sector_comparison = pd.DataFrame([
        {
            "Universo": "Solo Tecnología",
            "Rendimiento Máx. Sharpe":
                tech_analysis["perf_sharpe"][0],
            "Volatilidad Máx. Sharpe":
                tech_analysis["perf_sharpe"][1],
            "Sharpe máximo":
                tech_analysis["perf_sharpe"][2],
            "Volatilidad mínima":
                tech_analysis["perf_min"][1],
        },
        {
            "Universo": "Solo Industriales",
            "Rendimiento Máx. Sharpe":
                ind_analysis["perf_sharpe"][0],
            "Volatilidad Máx. Sharpe":
                ind_analysis["perf_sharpe"][1],
            "Sharpe máximo":
                ind_analysis["perf_sharpe"][2],
            "Volatilidad mínima":
                ind_analysis["perf_min"][1],
        },
        {
            "Universo": "Tecnología + Industriales",
            "Rendimiento Máx. Sharpe":
                perf_sharpe[0],
            "Volatilidad Máx. Sharpe":
                perf_sharpe[1],
            "Sharpe máximo":
                perf_sharpe[2],
            "Volatilidad mínima":
                perf_min[1],
        },
    ])

    sector_display = sector_comparison.copy()

    for col in [
        "Rendimiento Máx. Sharpe",
        "Volatilidad Máx. Sharpe",
        "Volatilidad mínima",
    ]:
        sector_display[col] *= 100

    st.dataframe(
        sector_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rendimiento Máx. Sharpe":
                st.column_config.NumberColumn(format="%.2f %%"),
            "Volatilidad Máx. Sharpe":
                st.column_config.NumberColumn(format="%.2f %%"),
            "Volatilidad mínima":
                st.column_config.NumberColumn(format="%.2f %%"),
            "Sharpe máximo":
                st.column_config.NumberColumn(format="%.3f"),
        },
    )

except Exception as e:
    st.warning(
        "No se pudo completar la comparación sectorial: "
        + str(e)
    )


# ============================================================
# EXPLICACIÓN ACADÉMICA
# ============================================================
st.divider()
st.subheader("7. Interpretación")

st.markdown(
    """
**Qué está haciendo Markowitz:** el modelo no escoge simplemente la acción
que más ganó. Analiza simultáneamente el rendimiento esperado de cada activo,
su volatilidad y, especialmente, cómo se mueven unas acciones respecto a otras
mediante la matriz de covarianzas.

**Máximo Sharpe:** busca una combinación que maximice el exceso de rendimiento
por unidad de riesgo, tomando en cuenta la tasa libre de riesgo elegida.

**Mínima varianza:** busca la combinación con la menor volatilidad posible,
sin ventas en corto y con 100% del capital invertido.

**Frontera eficiente:** representa las combinaciones que ofrecen el mayor
rendimiento esperado posible para diferentes niveles de riesgo.

**Equal Weight:** sirve como referencia sencilla: se invierte exactamente el
mismo porcentaje en cada una de las 20 acciones.
"""
)

st.warning(
    "Limitación importante: las 20 acciones fueron seleccionadas "
    "retrospectivamente porque tuvieron grandes rendimientos durante los "
    "últimos 5 años. Por ello existe sesgo retrospectivo y de supervivencia. "
    "Los resultados describen el comportamiento histórico y no constituyen "
    "una predicción de rendimientos futuros."
)


# ============================================================
# DESCARGA RESULTADOS
# ============================================================
st.divider()
st.subheader("8. Descargar resultados")

results_csv = comparison_display.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "⬇️ Descargar comparación de portafolios",
    data=results_csv,
    file_name="resultados_markowitz.csv",
    mime="text/csv",
    use_container_width=True,
)
