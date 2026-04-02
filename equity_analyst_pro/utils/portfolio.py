import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf


AED_COLORS = {
    "navy": "#0b1a2b",
    "navy_2": "#16283d",
    "gold": "#b08b40",
    "gold_soft": "#d4b26a",
    "cream": "#f8f6f1",
    "ink": "#253241",
    "line": "#d9d4c7",
}


BENCHMARKS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "CAC 40": "^FCHI",
    "STOXX Europe 600": "^STOXX",
    "MSCI World (proxy)": "URTH",
}


SECTOR_MAP = {
    "AAPL": "Technologie",
    "MSFT": "Technologie",
    "GOOGL": "Communication",
    "AMZN": "Consommation",
    "META": "Communication",
    "NVDA": "Technologie",
    "ASML": "Technologie",
    "JPM": "Finance",
    "V": "Finance",
    "MC": "Finance",
    "JNJ": "Santé",
    "PG": "Consommation défensive",
    "NESN.SW": "Consommation défensive",
    "NVO": "Santé",
    "SU.PA": "Industrie",
    "AI.PA": "Matériaux",
    "OR.PA": "Consommation",
    "RMS.PA": "Consommation",
    "ABI.BR": "Consommation défensive",
    "LVMUY": "Consommation",
}

REGION_MAP = {
    "AAPL": "États-Unis",
    "MSFT": "États-Unis",
    "GOOGL": "États-Unis",
    "AMZN": "États-Unis",
    "META": "États-Unis",
    "NVDA": "États-Unis",
    "ASML": "Europe",
    "JPM": "États-Unis",
    "V": "États-Unis",
    "MC": "États-Unis",
    "JNJ": "États-Unis",
    "PG": "États-Unis",
    "NESN.SW": "Europe",
    "NVO": "Europe",
    "SU.PA": "Europe",
    "AI.PA": "Europe",
    "OR.PA": "Europe",
    "RMS.PA": "Europe",
    "ABI.BR": "Europe",
    "LVMUY": "Europe",
}


def load_demo_portfolio():
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL", "JNJ", "PG"],
        "weight": [0.25, 0.20, 0.20, 0.15, 0.20],
    })


def fetch_price_history(tickers, period="2y"):
    try:
        df = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return df.dropna(how="all")
    except Exception:
        return None


def _clean_weights(weights, n_cols):
    w = np.array(weights[:n_cols], dtype=float)
    w = np.nan_to_num(w, nan=0.0)
    total = w.sum()
    if total <= 0:
        w = np.repeat(1 / n_cols, n_cols)
    else:
        w = w / total
    return w


def _build_line_chart(series, title):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            line=dict(color=AED_COLORS["navy"], width=3),
            name=title,
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=AED_COLORS["navy"])),
        paper_bgcolor=AED_COLORS["cream"],
        plot_bgcolor="white",
        font=dict(color=AED_COLORS["ink"]),
        margin=dict(l=30, r=30, t=70, b=30),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor=AED_COLORS["line"], zeroline=False),
        height=420,
    )
    return fig


def _build_bar_chart(df, x, y, title):
    fig = px.bar(df, x=x, y=y, title=title)
    fig.update_traces(marker_color=AED_COLORS["navy"])
    fig.update_layout(
        paper_bgcolor=AED_COLORS["cream"],
        plot_bgcolor="white",
        font=dict(color=AED_COLORS["ink"]),
        margin=dict(l=30, r=30, t=70, b=30),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor=AED_COLORS["line"], zeroline=False),
        height=380,
    )
    return fig


def compute_portfolio_analytics(prices: pd.DataFrame, weights: np.ndarray, benchmark_symbol: str = "^GSPC", risk_free_rate: float = 0.02):
    prices = prices.dropna(axis=1, how="all")
    returns = prices.pct_change().dropna()

    clean_cols = returns.columns.tolist()
    weights = _clean_weights(weights, len(clean_cols))

    mean_returns = returns.mean() * 252
    cov = returns.cov() * 252

    port_return = float(np.dot(weights, mean_returns))
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else np.nan

    portfolio_daily = returns.dot(weights)
    cumulative = (1 + portfolio_daily).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_drawdown = drawdown.min()
    var_95 = np.percentile(portfolio_daily, 5)

    benchmark_prices = fetch_price_history([benchmark_symbol], period="2y")
    bench_daily = None
    beta = np.nan
    alpha = np.nan
    tracking_error = np.nan
    info_ratio = np.nan
    relative_chart = None

    if benchmark_prices is not None and not benchmark_prices.empty:
        if isinstance(benchmark_prices, pd.Series):
            benchmark_prices = benchmark_prices.to_frame()
        bench_series = benchmark_prices.iloc[:, 0].dropna()
        bench_daily = bench_series.pct_change().dropna()

        aligned = pd.concat([portfolio_daily, bench_daily], axis=1, join="inner").dropna()
        if aligned.shape[0] > 10:
            aligned.columns = ["portfolio", "benchmark"]
            cov_pb = aligned["portfolio"].cov(aligned["benchmark"])
            var_b = aligned["benchmark"].var()
            if var_b and not np.isnan(var_b) and var_b != 0:
                beta = cov_pb / var_b

            bench_return = aligned["benchmark"].mean() * 252
            bench_vol = aligned["benchmark"].std() * np.sqrt(252)

            alpha = port_return - (risk_free_rate + beta * (bench_return - risk_free_rate)) if not np.isnan(beta) else np.nan
            active_daily = aligned["portfolio"] - aligned["benchmark"]
            tracking_error = active_daily.std() * np.sqrt(252)
            if tracking_error and not np.isnan(tracking_error) and tracking_error != 0:
                info_ratio = ((aligned["portfolio"].mean() - aligned["benchmark"].mean()) * 252) / tracking_error

            portfolio_curve = (1 + aligned["portfolio"]).cumprod()
            benchmark_curve = (1 + aligned["benchmark"]).cumprod()
            relative_chart = go.Figure()
            relative_chart.add_trace(go.Scatter(x=portfolio_curve.index, y=portfolio_curve.values, name="Portefeuille", line=dict(color=AED_COLORS["navy"], width=3)))
            relative_chart.add_trace(go.Scatter(x=benchmark_curve.index, y=benchmark_curve.values, name="Benchmark", line=dict(color=AED_COLORS["gold"], width=3)))
            relative_chart.update_layout(
                title=dict(text="Portefeuille vs benchmark", font=dict(size=20, color=AED_COLORS["navy"])),
                paper_bgcolor=AED_COLORS["cream"],
                plot_bgcolor="white",
                font=dict(color=AED_COLORS["ink"]),
                margin=dict(l=30, r=30, t=70, b=30),
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor=AED_COLORS["line"], zeroline=False),
                height=420,
            )

    contrib_perf = weights * mean_returns
    contrib_perf_df = pd.DataFrame({
        "Ligne": clean_cols,
        "Poids": weights,
        "Contribution à la performance attendue": contrib_perf,
    }).sort_values("Contribution à la performance attendue", ascending=False)

    marginal_risk = cov.dot(weights)
    total_vol = port_vol if port_vol > 0 else np.nan
    risk_contrib = weights * marginal_risk / total_vol if total_vol and not np.isnan(total_vol) else np.repeat(np.nan, len(weights))
    contrib_risk_df = pd.DataFrame({
        "Ligne": clean_cols,
        "Poids": weights,
        "Contribution au risque": risk_contrib,
    }).sort_values("Contribution au risque", ascending=False)

    sector_df = pd.DataFrame({
        "Ligne": clean_cols,
        "Poids": weights,
        "Secteur": [SECTOR_MAP.get(t, "Autre") for t in clean_cols],
        "Région": [REGION_MAP.get(t, "Autre") for t in clean_cols],
    })

    sector_exposure = sector_df.groupby("Secteur", as_index=False)["Poids"].sum().sort_values("Poids", ascending=False)
    region_exposure = sector_df.groupby("Région", as_index=False)["Poids"].sum().sort_values("Poids", ascending=False)

    top10_weight = sector_df["Poids"].sort_values(ascending=False).head(10).sum()
    max_line = sector_df["Poids"].max()
    concentration_warning = []
    if max_line > 0.25:
        concentration_warning.append("Une ligne dépasse 25% du portefeuille.")
    if top10_weight > 0.80:
        concentration_warning.append("Le portefeuille apparaît concentré sur un nombre limité de positions.")
    if sector_exposure["Poids"].max() > 0.40:
        concentration_warning.append("L'exposition sectorielle est élevée sur un même segment.")
    if not concentration_warning:
        concentration_warning.append("Alerte majeure de concentration non détectée sur ces seuils simples.")

    risk_table = pd.DataFrame([
        {"Métrique": "Rendement annualisé", "Valeur": f"{port_return:.2%}"},
        {"Métrique": "Volatilité annualisée", "Valeur": f"{port_vol:.2%}"},
        {"Métrique": "Sharpe", "Valeur": f"{sharpe:.2f}"},
        {"Métrique": "VaR 95% (1 jour)", "Valeur": f"{var_95:.2%}"},
        {"Métrique": "Max drawdown", "Valeur": f"{max_drawdown:.2%}"},
        {"Métrique": "Bêta", "Valeur": "N/D" if np.isnan(beta) else f"{beta:.2f}"},
        {"Métrique": "Alpha", "Valeur": "N/D" if np.isnan(alpha) else f"{alpha:.2%}"},
        {"Métrique": "Tracking error", "Valeur": "N/D" if np.isnan(tracking_error) else f"{tracking_error:.2%}"},
        {"Métrique": "Ratio d'information", "Valeur": "N/D" if np.isnan(info_ratio) else f"{info_ratio:.2f}"},
    ])

    cum_chart = _build_line_chart(cumulative, "Performance cumulée")

    corr_chart = px.imshow(
        returns.corr(),
        text_auto=".2f",
        aspect="auto",
        title="Corrélation des rendements",
        color_continuous_scale=[
            [0.0, "#f8f6f1"],
            [0.5, "#d4b26a"],
            [1.0, "#0b1a2b"],
        ],
    )
    corr_chart.update_layout(
        paper_bgcolor=AED_COLORS["cream"],
        plot_bgcolor="white",
        font=dict(color=AED_COLORS["ink"]),
        margin=dict(l=30, r=30, t=70, b=30),
    )

    sector_chart = _build_bar_chart(sector_exposure, "Secteur", "Poids", "Exposition sectorielle")
    region_chart = _build_bar_chart(region_exposure, "Région", "Poids", "Exposition géographique")

    return {
        "annual_return": f"{port_return:.2%}",
        "annual_vol": f"{port_vol:.2%}",
        "sharpe": f"{sharpe:.2f}",
        "var_95": f"{var_95:.2%}",
        "max_drawdown": f"{max_drawdown:.2%}",
        "beta": "N/D" if np.isnan(beta) else f"{beta:.2f}",
        "alpha": "N/D" if np.isnan(alpha) else f"{alpha:.2%}",
        "tracking_error": "N/D" if np.isnan(tracking_error) else f"{tracking_error:.2%}",
        "info_ratio": "N/D" if np.isnan(info_ratio) else f"{info_ratio:.2f}",
        "contrib_perf_df": contrib_perf_df,
        "contrib_risk_df": contrib_risk_df,
        "risk_table": risk_table,
        "cum_chart": cum_chart,
        "corr_chart": corr_chart,
        "relative_chart": relative_chart,
        "sector_chart": sector_chart,
        "region_chart": region_chart,
        "sector_exposure": sector_exposure,
        "region_exposure": region_exposure,
        "alerts": concentration_warning,
        "composition_df": sector_df.sort_values("Poids", ascending=False),
    }


def simulate_efficient_frontier(prices: pd.DataFrame, n_portfolios: int = 2000, risk_free_rate: float = 0.02):
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov = returns.cov() * 252
    n_assets = len(mean_returns)

    results = []
    for _ in range(n_portfolios):
        w = np.random.random(n_assets)
        w /= w.sum()
        ret = np.dot(w, mean_returns)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else np.nan
        results.append([ret, vol, sharpe, *w])

    cols = ["Rendement", "Volatilité", "Sharpe"] + list(prices.columns)
    frontier = pd.DataFrame(results, columns=cols)

    top_sharpe = frontier.sort_values("Sharpe", ascending=False).head(1)
    min_vol = frontier.sort_values("Volatilité", ascending=True).head(1)
    top = pd.concat([top_sharpe.assign(Type="Sharpe max"), min_vol.assign(Type="Variance minimale")], ignore_index=True)

    chart = px.scatter(
        frontier,
        x="Volatilité",
        y="Rendement",
        color="Sharpe",
        title="Frontière efficiente simulée",
        color_continuous_scale=[
            [0.0, "#d9d4c7"],
            [0.5, "#d4b26a"],
            [1.0, "#0b1a2b"],
        ],
    )
    chart.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0)))
    chart.update_layout(
        paper_bgcolor=AED_COLORS["cream"],
        plot_bgcolor="white",
        font=dict(color=AED_COLORS["ink"]),
        margin=dict(l=30, r=30, t=70, b=30),
        xaxis=dict(gridcolor=AED_COLORS["line"], zeroline=False, title="Volatilité"),
        yaxis=dict(gridcolor=AED_COLORS["line"], zeroline=False, title="Rendement annualisé"),
    )

    return {
        "chart": chart,
        "top_portfolios": top,
    }
