import io
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Markowitz — Tecnología + Industriales",
    page_icon="📈",
    layout="wide",
)

TRADING_DAYS = 252
MIN_COVERAGE = 0.90

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
# ESTADO DE LA APLICACIÓN
# ============================================================
if "fase" not in st.session_state:
    st.session_state.fase = 1

if "universo_final" not in st.session_state:
    st.session_state.universo_final = None

if "precios_fase1" not in st.session_state:
    st.session_state.precios_fase1 = None


# ============================================================
# FUNCIONES — STATE STREET
# ============================================================
def _find_header_row(raw: pd.DataFrame):
    for idx in range(min(len(raw), 40)):
        vals = [
            str(v).strip().lower()
            for v in raw.iloc[idx].tolist()
            if pd.notna(v)
        ]
        joined = " | ".join(vals)

        has_ticker = ("ticker" in joined) or ("symbol" in joined)
        has_name = ("name" in joined) or ("security" in joined)

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

    if len(response.content) < 2000:
        raise ValueError(
            f"State Street devolvió un archivo demasiado pequeño para {fund}."
        )

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
        ["security name", "name", "security"],
    )

    weight_col = _choose_column(
        table.columns,
        ["weight (%)", "weight %", "weight"],
    )

    if ticker_col is None:
        raise ValueError(
            f"No se encontró Ticker/Symbol en el archivo oficial de {fund}."
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

    out["Ticker"] = (
        out["Ticker"]
        .str.replace(".", "-", regex=False)
        .str.upper()
    )

    invalid = {"", "NAN", "NONE", "CASH", "USD", "-"}

    out = out[
        ~out["Ticker"].isin(invalid)
        & out["Ticker"].str.match(r"^[A-Z0-9\-]+$", na=False)
    ].copy()

    out = out.drop_duplicates("Ticker").reset_index(drop=True)

    if len(out) < 10:
        raise ValueError(
            f"Solo se detectaron {len(out)} holdings válidos para {fund}."
        )

    return out


# ============================================================
# FUNCIONES — YAHOO FINANCE
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

        time.sleep(0.15)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]

    return prices.sort_index()


# ============================================================
# FUNCIONES — FASE 1
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

        days = max(
            (s.index[-1] - s.index[0]).days,
            1,
        )

        years = days / 365.25

        cagr = (final / initial) ** (1 / years) - 1

        daily = s.pct_change(fill_method=None).dropna()

        vol = (
            daily.std() * np.sqrt(TRADING_DAYS)
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

    show.insert(
        0,
        "Posición",
        range(1, len(show) + 1),
    )

    for col in [
        "Rendimiento 5Y",
        "CAGR",
        "Volatilidad anual",
        "Cobertura",
    ]:
        show[col] *= 100

    st.dataframe(
        show[
            [
                "Posición",
                "Ticker",
                "Nombre",
                "Rendimiento 5Y",
                "CAGR",
                "Volatilidad anual",
                "Cobertura",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rendimiento 5Y":
                st.column_config.NumberColumn(format="%.2f %%"),
            "CAGR":
                st.column_config.NumberColumn(format="%.2f %%"),
            "Volatilidad anual":
                st.column_config.NumberColumn(format="%.2f %%"),
            "Cobertura":
                st.column_config.NumberColumn(format="%.1f %%"),
        },
    )


# ============================================================
# FUNCIONES — MARKOWITZ
# ============================================================
def annual_statistics(prices):
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    returns = returns.dropna(axis=0, how="any")

    mean_returns = returns.mean() * TRADING_DAYS
    cov_matrix = returns.cov() * TRADING_DAYS
    corr_matrix = returns.corr()

    return returns, mean_returns, cov_matrix, corr_matrix


def portfolio_performance(
    weights,
    mean_returns,
    cov_matrix,
    risk_free,
):
    weights = np.asarray(weights)

    ret = float(
        np.dot(weights, mean_returns.values)
    )

    vol = float(
        np.sqrt(
            weights.T
            @ cov_matrix.values
            @ weights
        )
    )

    sharpe = (
        (ret - risk_free) / vol
        if vol > 0
        else np.nan
    )

    return ret, vol, sharpe


def min_variance_portfolio(
    mean_returns,
    cov_matrix,
    risk_free,
):
    n = len(mean_returns)
    x0 = np.repeat(1 / n, n)

    bounds = tuple(
        (0.0, 1.0)
        for _ in range(n)
    )

    constraints = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    },)

    result = minimize(
        lambda w: portfolio_performance(
            w,
            mean_returns,
            cov_matrix,
            risk_free,
        )[1],
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "No se pudo calcular el portafolio de mínima varianza."
        )

    return result.x


def max_sharpe_portfolio(
    mean_returns,
    cov_matrix,
    risk_free,
):
    n = len(mean_returns)
    x0 = np.repeat(1 / n, n)

    bounds = tuple(
        (0.0, 1.0)
        for _ in range(n)
    )

    constraints = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0,
    },)

    def negative_sharpe(w):
        _, _, sharpe = portfolio_performance(
            w,
            mean_returns,
            cov_matrix,
            risk_free,
        )

        return -sharpe if np.isfinite(sharpe) else 1e9

    result = minimize(
        negative_sharpe,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "No se pudo calcular el portafolio de máximo Sharpe."
        )

    return result.x


def monte_carlo(
    mean_returns,
    cov_matrix,
    risk_free,
    simulations,
):
    rng = np.random.default_rng(42)
    n = len(mean_returns)

    weights = rng.dirichlet(
        np.ones(n),
        size=simulations,
    )

    portfolio_returns = (
        weights @ mean_returns.values
    )

    variances = np.einsum(
        "ij,jk,ik->i",
        weights,
        cov_matrix.values,
        weights,
    )

    vol = np.sqrt(variances)

    sharpe = np.divide(
        portfolio_returns - risk_free,
        vol,
        out=np.full_like(
            portfolio_returns,
            np.nan,
        ),
        where=vol > 0,
    )

    return pd.DataFrame({
        "Rendimiento": portfolio_returns,
        "Volatilidad": vol,
        "Sharpe": sharpe,
    })


def efficient_frontier(
    mean_returns,
    cov_matrix,
    risk_free,
    points=45,
):
    n = len(mean_returns)
    x0 = np.repeat(1 / n, n)

    min_w = min_variance_portfolio(
        mean_returns,
        cov_matrix,
        risk_free,
    )

    min_ret, _, _ = portfolio_performance(
        min_w,
        mean_returns,
        cov_matrix,
        risk_free,
    )

    max_ret = float(mean_returns.max())

    targets = np.linspace(
        min_ret,
        max_ret,
        points,
    )

    rows = []

    for target in targets:
        constraints = (
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },
            {
                "type": "eq",
                "fun": lambda w, t=target:
                    np.dot(
                        w,
                        mean_returns.values,
                    ) - t,
            },
        )

        bounds = tuple(
            (0.0, 1.0)
            for _ in range(n)
        )

        result = minimize(
            lambda w: portfolio_performance(
                w,
                mean_returns,
                cov_matrix,
                risk_free,
            )[1],
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 1500,
                "ftol": 1e-10,
            },
        )

        if result.success:
            r, v, s = portfolio_performance(
                result.x,
                mean_returns,
                cov_matrix,
                risk_free,
            )

            rows.append({
                "Rendimiento": r,
                "Volatilidad": v,
                "Sharpe": s,
            })

    return pd.DataFrame(rows)


def weights_table(weights, universe):
    df = universe[
        ["Ticker", "Nombre", "Sector"]
    ].copy()

    df["Peso"] = np.asarray(weights)

    # Mostrar solamente pesos de al menos 0.05%.
    df = df[
        df["Peso"] >= 0.0005
    ].copy()

    return (
        df
        .sort_values("Peso", ascending=False)
        .reset_index(drop=True)
    )


def analyze_subset(
    tickers,
    prices,
    risk_free,
):
    subset = prices[tickers].dropna(
        axis=0,
        how="any",
    )

    _, mu, cov, _ = annual_statistics(subset)

    w_min = min_variance_portfolio(
        mu,
        cov,
        risk_free,
    )

    w_sharpe = max_sharpe_portfolio(
        mu,
        cov,
        risk_free,
    )

    return {
        "min": portfolio_performance(
            w_min,
            mu,
            cov,
            risk_free,
        ),
        "sharpe": portfolio_performance(
            w_sharpe,
            mu,
            cov,
            risk_free,
        ),
    }


# ============================================================
# ENCABEZADO
# ============================================================
st.title(
    "📈 Modelo de Markowitz — Tecnología + Industriales"
)

st.caption(
    "Proceso completo: selección de datos → construcción del portafolio."
)

# Indicador visual de fase
p1, p2 = st.columns(2)

if st.session_state.fase == 1:
    p1.success("① Fase 1 — Selección de acciones")
    p2.info("② Fase 2 — Markowitz")
else:
    p1.info("① Fase 1 — Selección completada")
    p2.success("② Fase 2 — Markowitz")


# ============================================================
# FASE 1
# ============================================================
if st.session_state.fase == 1:

    st.header(
        "Fase 1 — Obtención y selección de datos"
    )

    st.write(
        "Primero se obtienen los componentes actuales de "
        "**XLK (Tecnología)** y **XLI (Industriales)**. "
        "Posteriormente Yahoo Finance proporciona los precios "
        "diarios de los últimos cinco años."
    )

    st.sidebar.header("Fase 1")

    top_n = st.sidebar.slider(
        "Acciones por sector",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    st.sidebar.write("Periodo: **5 años**")
    st.sidebar.write("Cobertura mínima: **90%**")

    if not st.button(
        "🔎 Obtener datos y seleccionar acciones",
        type="primary",
        use_container_width=True,
    ):
        st.info(
            "Presiona el botón para comenzar la Fase 1."
        )

        st.markdown(
            """
            **Procedimiento de esta fase**

            1. Obtener holdings oficiales de XLK y XLI.
            2. Descargar precios históricos desde Yahoo Finance.
            3. Calcular rendimiento acumulado de 5 años.
            4. Calcular CAGR y volatilidad anual.
            5. Excluir acciones sin suficiente historial.
            6. Seleccionar las acciones con mayor rendimiento de cada sector.
            """
        )

        st.stop()

    universes = {}

    for sector in [
        "Tecnología",
        "Industriales",
    ]:
        fund = STATE_STREET_FILES[
            sector
        ]["fund"]

        with st.spinner(
            f"Obteniendo holdings de {fund}..."
        ):
            universes[sector] = (
                get_state_street_holdings(
                    sector
                )
            )

        st.success(
            f"{fund}: "
            f"{len(universes[sector])} "
            "acciones detectadas."
        )

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
        "Descargando cinco años de precios "
        "desde Yahoo Finance..."
    ):
        prices = download_prices(
            all_tickers
        )

    if prices.empty:
        st.error(
            "Yahoo Finance no devolvió precios."
        )
        st.stop()

    st.write(
        f"Yahoo devolvió datos para "
        f"**{len(prices.columns)}** de "
        f"**{len(all_tickers)}** tickers."
    )

    selected = []

    for sector in [
        "Tecnología",
        "Industriales",
    ]:
        ranking = calculate_ranking(
            prices,
            universes[sector],
            MIN_COVERAGE,
        )

        st.divider()
        st.subheader(
            f"🏆 {sector} — Top {top_n}"
        )

        if ranking.empty:
            st.error(
                f"No se pudo construir "
                f"el ranking de {sector}."
            )
            st.stop()

        top = ranking.head(
            top_n
        ).copy()

        selected.append(top)

        show_ranking(top)

        graph = top.sort_values(
            "Rendimiento 5Y",
            ascending=True,
        ).copy()

        graph["Rendimiento (%)"] = (
            graph["Rendimiento 5Y"] * 100
        )

        fig = px.bar(
            graph,
            x="Rendimiento (%)",
            y="Ticker",
            orientation="h",
            hover_name="Nombre",
            title=(
                "Rendimiento acumulado "
                f"de 5 años — {sector}"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    combined = pd.concat(
        selected,
        ignore_index=True,
    )

    st.divider()
    st.subheader(
        "Universo seleccionado para Markowitz"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total",
        len(combined),
    )

    c2.metric(
        "Tecnología",
        (
            combined["Sector"]
            == "Tecnología"
        ).sum(),
    )

    c3.metric(
        "Industriales",
        (
            combined["Sector"]
            == "Industriales"
        ).sum(),
    )

    final_display = combined[
        [
            "Ticker",
            "Nombre",
            "Sector",
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
        final_display[col] *= 100

    st.dataframe(
        final_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rendimiento 5Y":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
            "CAGR":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
            "Volatilidad anual":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
        },
    )

    # Guardar resultados para la fase 2
    st.session_state.universo_final = (
        combined[
            [
                "Ticker",
                "Nombre",
                "Sector",
                "Fondo",
            ]
        ].copy()
    )

    st.session_state.precios_fase1 = (
        prices.copy()
    )

    st.success(
        "✅ Fase 1 completada. "
        "El universo está listo."
    )

    if st.button(
        "Siguiente → Aplicar Markowitz",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.fase = 2
        st.rerun()


# ============================================================
# FASE 2
# ============================================================
else:

    universe = st.session_state.universo_final

    if universe is None:
        st.warning(
            "Primero es necesario completar "
            "la Fase 1."
        )

        if st.button(
            "← Ir a Fase 1"
        ):
            st.session_state.fase = 1
            st.rerun()

        st.stop()

    st.header(
        "Fase 2 — Modelo de Markowitz"
    )

    st.write(
        "Ahora se utilizan exclusivamente las acciones "
        "seleccionadas en la Fase 1."
    )

    st.sidebar.header("Fase 2")

    risk_free_pct = st.sidebar.number_input(
        "Tasa libre de riesgo anual (%)",
        min_value=0.0,
        max_value=20.0,
        value=4.0,
        step=0.25,
    )

    simulations = st.sidebar.select_slider(
        "Portafolios simulados",
        options=[
            10000,
            25000,
            50000,
            100000,
        ],
        value=50000,
    )

    risk_free = risk_free_pct / 100

    if st.button(
        "← Volver a Fase 1"
    ):
        st.session_state.fase = 1
        st.rerun()

    tickers = universe[
        "Ticker"
    ].tolist()

    prices_all = (
        st.session_state.precios_fase1
    )

    # Si por alguna razón los precios ya no están
    # disponibles, se vuelven a descargar.
    if prices_all is None:
        with st.spinner(
            "Recuperando precios..."
        ):
            prices_all = download_prices(
                tuple(tickers)
            )

    missing = [
        t
        for t in tickers
        if t not in prices_all.columns
    ]

    if missing:
        st.error(
            "Faltan precios para: "
            + ", ".join(missing)
        )
        st.stop()

    prices = prices_all[
        tickers
    ].copy()

    returns, mean_returns, cov_matrix, corr_matrix = (
        annual_statistics(prices)
    )

    if len(returns) < 100:
        st.error(
            "No hay suficientes observaciones comunes."
        )
        st.stop()

    # --------------------------------------------------------
    # 2.1 UNIVERSO
    # --------------------------------------------------------
    st.subheader(
        "2.1 Universo utilizado"
    )

    st.dataframe(
        universe,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # 2.2 ESTADÍSTICAS
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "2.2 Rendimiento esperado y volatilidad"
    )

    individual = pd.DataFrame({
        "Ticker": mean_returns.index,
        "Rendimiento esperado":
            mean_returns.values,
        "Volatilidad":
            np.sqrt(
                np.diag(
                    cov_matrix.values
                )
            ),
    })

    individual = individual.merge(
        universe,
        on="Ticker",
        how="left",
    )

    individual_show = (
        individual.copy()
    )

    individual_show[
        "Rendimiento esperado"
    ] *= 100

    individual_show[
        "Volatilidad"
    ] *= 100

    st.dataframe(
        individual_show[
            [
                "Ticker",
                "Nombre",
                "Sector",
                "Rendimiento esperado",
                "Volatilidad",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rendimiento esperado":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
            "Volatilidad":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
        },
    )

    # --------------------------------------------------------
    # 2.3 CORRELACIÓN Y COVARIANZA
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "2.3 Correlaciones y covarianzas"
    )

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        zmin=-1,
        zmax=1,
        title=(
            "Matriz de correlación "
            "de rendimientos diarios"
        ),
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True,
    )

    with st.expander(
        "Ver matriz de covarianzas anualizada"
    ):
        st.dataframe(
            cov_matrix,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # 2.4 OPTIMIZACIÓN
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "2.4 Optimización de portafolios"
    )

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

    w_equal = np.repeat(
        1 / len(tickers),
        len(tickers),
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

    perf_equal = portfolio_performance(
        w_equal,
        mean_returns,
        cov_matrix,
        risk_free,
    )

    with st.spinner(
        f"Simulando {simulations:,} portafolios..."
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
        )

    mc_plot = mc.copy()

    mc_plot["Rendimiento (%)"] = (
        mc_plot["Rendimiento"] * 100
    )

    mc_plot["Volatilidad (%)"] = (
        mc_plot["Volatilidad"] * 100
    )

    fig = px.scatter(
        mc_plot,
        x="Volatilidad (%)",
        y="Rendimiento (%)",
        color="Sharpe",
        opacity=0.40,
        title=(
            f"Markowitz — "
            f"{simulations:,} portafolios"
        ),
    )

    if not frontier.empty:
        fig.add_trace(
            go.Scatter(
                x=(
                    frontier["Volatilidad"]
                    * 100
                ),
                y=(
                    frontier["Rendimiento"]
                    * 100
                ),
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
            marker=dict(
                size=16,
                symbol="diamond",
            ),
            name="Mínima varianza",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[perf_sharpe[1] * 100],
            y=[perf_sharpe[0] * 100],
            mode="markers",
            marker=dict(
                size=18,
                symbol="star",
            ),
            name="Máximo Sharpe",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[perf_equal[1] * 100],
            y=[perf_equal[0] * 100],
            mode="markers",
            marker=dict(
                size=14,
                symbol="circle-open",
            ),
            name="Equal Weight",
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 2.5 RESULTADOS
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "2.5 Comparación de resultados"
    )

    comparison = pd.DataFrame([
        {
            "Portafolio":
                "Máximo Sharpe",
            "Rendimiento esperado":
                perf_sharpe[0],
            "Volatilidad":
                perf_sharpe[1],
            "Sharpe":
                perf_sharpe[2],
        },
        {
            "Portafolio":
                "Mínima varianza",
            "Rendimiento esperado":
                perf_min[0],
            "Volatilidad":
                perf_min[1],
            "Sharpe":
                perf_min[2],
        },
        {
            "Portafolio":
                "Equal Weight",
            "Rendimiento esperado":
                perf_equal[0],
            "Volatilidad":
                perf_equal[1],
            "Sharpe":
                perf_equal[2],
        },
    ])

    comparison_show = (
        comparison.copy()
    )

    comparison_show[
        "Rendimiento esperado"
    ] *= 100

    comparison_show[
        "Volatilidad"
    ] *= 100

    st.dataframe(
        comparison_show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rendimiento esperado":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
            "Volatilidad":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
            "Sharpe":
                st.column_config.NumberColumn(
                    format="%.3f"
                ),
        },
    )

    # --------------------------------------------------------
    # 2.6 PESOS
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "2.6 Distribución del capital"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "### Máximo Sharpe"
        )

        sharpe_weights = weights_table(
            w_sharpe,
            universe,
        )

        sharpe_weights[
            "Peso"
        ] *= 100

        st.dataframe(
            sharpe_weights,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Peso":
                    st.column_config.NumberColumn(
                        format="%.2f %%"
                    ),
            },
        )

    with col2:
        st.markdown(
            "### Mínima varianza"
        )

        min_weights = weights_table(
            w_min,
            universe,
        )

        min_weights[
            "Peso"
        ] *= 100

        st.dataframe(
            min_weights,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Peso":
                    st.column_config.NumberColumn(
                        format="%.2f %%"
                    ),
            },
        )

    # --------------------------------------------------------
    # 2.7 COMPARACIÓN POR SECTOR
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "2.7 Tecnología vs Industriales vs combinación"
    )

    tech = universe.loc[
        universe["Sector"]
        == "Tecnología",
        "Ticker",
    ].tolist()

    industrial = universe.loc[
        universe["Sector"]
        == "Industriales",
        "Ticker",
    ].tolist()

    tech_result = analyze_subset(
        tech,
        prices,
        risk_free,
    )

    ind_result = analyze_subset(
        industrial,
        prices,
        risk_free,
    )

    sectors = pd.DataFrame([
        {
            "Universo":
                "Solo Tecnología",
            "Rendimiento Máx. Sharpe":
                tech_result["sharpe"][0],
            "Volatilidad Máx. Sharpe":
                tech_result["sharpe"][1],
            "Sharpe máximo":
                tech_result["sharpe"][2],
            "Volatilidad mínima":
                tech_result["min"][1],
        },
        {
            "Universo":
                "Solo Industriales",
            "Rendimiento Máx. Sharpe":
                ind_result["sharpe"][0],
            "Volatilidad Máx. Sharpe":
                ind_result["sharpe"][1],
            "Sharpe máximo":
                ind_result["sharpe"][2],
            "Volatilidad mínima":
                ind_result["min"][1],
        },
        {
            "Universo":
                "Tecnología + Industriales",
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

    sectors_show = (
        sectors.copy()
    )

    for col in [
        "Rendimiento Máx. Sharpe",
        "Volatilidad Máx. Sharpe",
        "Volatilidad mínima",
    ]:
        sectors_show[col] *= 100

    st.dataframe(
        sectors_show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rendimiento Máx. Sharpe":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
            "Volatilidad Máx. Sharpe":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
            "Sharpe máximo":
                st.column_config.NumberColumn(
                    format="%.3f"
                ),
            "Volatilidad mínima":
                st.column_config.NumberColumn(
                    format="%.2f %%"
                ),
        },
    )

    # --------------------------------------------------------
    # 2.8 INTERPRETACIÓN
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "2.8 Interpretación"
    )

    st.markdown(
        """
        El modelo de Markowitz analiza simultáneamente el
        **rendimiento esperado**, la **volatilidad** y la
        **covarianza** entre los activos.

        - **Máximo Sharpe:** busca la mayor compensación de
          rendimiento sobre la tasa libre de riesgo por unidad
          de volatilidad.
        - **Mínima varianza:** busca la combinación de menor
          volatilidad posible.
        - **Frontera eficiente:** representa los portafolios
          que ofrecen el mayor rendimiento esperado para cada
          nivel de riesgo.
        - **Equal Weight:** funciona como referencia al asignar
          el mismo capital a todas las acciones.
        """
    )

    st.warning(
        "Las acciones fueron seleccionadas retrospectivamente "
        "por su buen desempeño histórico. Por ello existe "
        "sesgo retrospectivo y de supervivencia. Los resultados "
        "no representan una predicción de rendimientos futuros."
    )

    # --------------------------------------------------------
    # DESCARGAS
    # --------------------------------------------------------
    st.divider()
    st.subheader(
        "Descargar resultados"
    )

    st.download_button(
        "⬇️ Resultados Markowitz CSV",
        data=(
            comparison_show
            .to_csv(index=False)
            .encode("utf-8-sig")
        ),
        file_name=(
            "resultados_markowitz.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )
