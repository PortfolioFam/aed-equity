import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import requests


AED_COLORS = {
    "navy": "#0b1a2b",
    "navy_2": "#16283d",
    "gold": "#b08b40",
    "gold_soft": "#d4b26a",
    "cream": "#f8f6f1",
    "ink": "#253241",
    "muted": "#6b7280",
    "line": "#d9d4c7",
}


def _fmt_pct(value):
    if value is None or pd.isna(value):
        return "N/D"
    return f"{value:.1%}"


def _fmt_num(value, digits=1, suffix=""):
    if value is None or pd.isna(value):
        return "N/D"
    return f"{value:.{digits}f}{suffix}"


def _fmt_large_amount(value, currency="USD"):
    if value is None or pd.isna(value):
        return "N/D"
    symbol_map = {"USD": "$", "EUR": "€", "CHF": "CHF", "DKK": "DKK", "GBP": "£"}
    symbol = symbol_map.get(currency, currency)

    abs_val = abs(value)
    if abs_val >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f} Md {symbol}"
    if abs_val >= 1_000_000:
        return f"{value/1_000_000:.1f} M {symbol}"
    return f"{value:.0f} {symbol}"


def _safe_float(x):
    try:
        if x in [None, "", "None", "N/D", "-", "null"]:
            return None
        return float(x)
    except Exception:
        return None


def _safe_ratio(num, den):
    if num is None or den in [None, 0]:
        return None
    try:
        return float(num) / float(den)
    except Exception:
        return None


def _get_yahoo_price_and_currency(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        hist = tk.history(period="1mo", interval="1d", auto_adjust=True)

        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        if price is None and not hist.empty and "Close" in hist.columns:
            price = float(hist["Close"].dropna().iloc[-1])

        currency = info.get("currency", "USD")
        company = info.get("shortName", ticker.upper())

        return {
            "price": price,
            "currency": currency,
            "company": company,
            "info": info,
        }
    except Exception:
        return {
            "price": None,
            "currency": "USD",
            "company": ticker.upper(),
            "info": {},
        }


def _get_alpha_income_statement(ticker: str, api_key: str):
    try:
        base_url = "https://www.alphavantage.co/query"
        data = requests.get(
            base_url,
            params={"function": "INCOME_STATEMENT", "symbol": ticker.upper(), "apikey": api_key},
            timeout=20,
        ).json()
        return data
    except Exception:
        return {}


def _build_history_rows_from_alpha(ticker: str, api_key: str):
    data = _get_alpha_income_statement(ticker, api_key)
    annual_reports = data.get("annualReports", [])

    if not annual_reports:
        return [], {}

    reports = []
    for report in annual_reports:
        fiscal_date = report.get("fiscalDateEnding")
        year = str(fiscal_date)[:4] if fiscal_date else None

        revenue = _safe_float(report.get("totalRevenue"))
        gross_profit = _safe_float(report.get("grossProfit"))
        ebitda = _safe_float(report.get("ebitda"))
        operating_income = _safe_float(report.get("operatingIncome"))
        net_income = _safe_float(report.get("netIncome"))

        if year is None:
            continue

        reports.append({
            "annee": int(year),
            "revenue": revenue,
            "gross_profit": gross_profit,
            "ebitda": ebitda,
            "operating_income": operating_income,
            "net_income": net_income,
            "marge_brute": _safe_ratio(gross_profit, revenue),
            "marge_operationnelle": _safe_ratio(operating_income, revenue),
            "marge_nette": _safe_ratio(net_income, revenue),
        })

    if not reports:
        return [], {}

    reports = sorted(reports, key=lambda x: x["annee"])
    reports = reports[-4:]

    base_revenue = None
    for r in reports:
        if r["revenue"] not in [None, 0]:
            base_revenue = r["revenue"]
            break

    history_rows = []
    for r in reports:
        revenue = r["revenue"]
        indice_ca = None
        if base_revenue not in [None, 0] and revenue is not None:
            indice_ca = (revenue / base_revenue) * 100

        history_rows.append({
            "annee": r["annee"],
            "indice_ca": indice_ca,
            "marge_brute": r["marge_brute"],
            "marge_operationnelle": r["marge_operationnelle"],
            "marge_flux_tresorerie": r["marge_nette"],
        })

    latest_report = reports[-1]

    latest_amounts = {
        "revenue": latest_report.get("revenue"),
        "gross_profit": latest_report.get("gross_profit"),
        "ebitda": latest_report.get("ebitda"),
        "operating_income": latest_report.get("operating_income"),
        "net_income": latest_report.get("net_income"),
        "gross_margin": latest_report.get("marge_brute"),
        "operating_margin": latest_report.get("marge_operationnelle"),
        "net_margin": latest_report.get("marge_nette"),
    }

    return history_rows, latest_amounts


def get_live_snapshot_alpha_vantage(ticker: str, api_key: str):
    yahoo_data = _get_yahoo_price_and_currency(ticker)
    info = yahoo_data.get("info", {})

    try:
        base_url = "https://www.alphavantage.co/query"

        overview = requests.get(
            base_url,
            params={"function": "OVERVIEW", "symbol": ticker.upper(), "apikey": api_key},
            timeout=20,
        ).json()

        history_rows, latest_amounts = _build_history_rows_from_alpha(ticker, api_key)

        if not overview or "Symbol" not in overview:
            fallback = build_yahoo_only_snapshot(ticker)
            if fallback is not None:
                return fallback
            return None

        pe_ratio = _safe_float(overview.get("PERatio"))

        roe = _safe_float(overview.get("ReturnOnEquityTTM"))
        if roe is not None and roe > 1:
            roe = roe / 100.0

        dte = _safe_float(overview.get("DebtToEquity"))
        if dte is not None and dte > 10:
            dte = dte / 100.0

        quarterly_growth = _safe_float(overview.get("QuarterlyRevenueGrowthYOY"))
        if quarterly_growth is not None and abs(quarterly_growth) > 1:
            quarterly_growth = quarterly_growth / 100.0

        revenue_ttm = _safe_float(overview.get("RevenueTTM"))
        gross_profit_ttm = _safe_float(overview.get("GrossProfitTTM"))
        ebitda_ttm = _safe_float(overview.get("EBITDA"))
        market_cap = _safe_float(overview.get("MarketCapitalization"))

        operating_margin_ttm = _safe_float(overview.get("OperatingMarginTTM"))
        if operating_margin_ttm is not None and operating_margin_ttm > 1:
            operating_margin_ttm = operating_margin_ttm / 100.0

        profit_margin_ttm = _safe_float(overview.get("ProfitMargin"))
        if profit_margin_ttm is not None and profit_margin_ttm > 1:
            profit_margin_ttm = profit_margin_ttm / 100.0

        gross_margin_ttm = _safe_ratio(gross_profit_ttm, revenue_ttm)

        operating_income_ttm = None
        if revenue_ttm is not None and operating_margin_ttm is not None:
            operating_income_ttm = revenue_ttm * operating_margin_ttm

        net_income_ttm = None
        if revenue_ttm is not None and profit_margin_ttm is not None:
            net_income_ttm = revenue_ttm * profit_margin_ttm

        if revenue_ttm is None:
            revenue_ttm = latest_amounts.get("revenue")
        if gross_profit_ttm is None:
            gross_profit_ttm = latest_amounts.get("gross_profit")
        if ebitda_ttm is None:
            ebitda_ttm = latest_amounts.get("ebitda")
        if operating_income_ttm is None:
            operating_income_ttm = latest_amounts.get("operating_income")
        if net_income_ttm is None:
            net_income_ttm = latest_amounts.get("net_income")

        gross_margin_display = gross_margin_ttm if gross_margin_ttm is not None else latest_amounts.get("gross_margin")
        operating_margin_display = operating_margin_ttm if operating_margin_ttm is not None else latest_amounts.get("operating_margin")
        net_margin_display = profit_margin_ttm if profit_margin_ttm is not None else latest_amounts.get("net_margin")

        if not history_rows:
            fallback = build_yahoo_only_snapshot(ticker)
            if fallback is not None:
                return fallback

        return {
            "ticker": ticker.upper(),
            "company": overview.get("Name", yahoo_data["company"]),
            "currency": yahoo_data["currency"] or overview.get("Currency", "USD"),
            "price": yahoo_data["price"],
            "pe_ratio": pe_ratio,
            "roe": roe,
            "revenue_growth": quarterly_growth,
            "debt_to_equity": dte,
            "gross_margin": gross_margin_display,
            "operating_margin": operating_margin_display,
            "fcf_margin": net_margin_display,
            "revenue_ttm": revenue_ttm,
            "gross_profit_ttm": gross_profit_ttm,
            "ebitda_ttm": ebitda_ttm,
            "operating_income_ttm": operating_income_ttm,
            "net_income_ttm": net_income_ttm,
            "market_cap": market_cap,
            "history_rows": history_rows,
            "source": "alpha_vantage_plus_yahoo",
            "price_source": "Yahoo Finance",
            "fundamentals_source": "Alpha Vantage",
        }
    except Exception:
        return build_yahoo_only_snapshot(ticker)


def build_yahoo_only_snapshot(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        hist = tk.history(period="5y", interval="1d", auto_adjust=True)

        revenue_growth = info.get("revenueGrowth")
        roe = info.get("returnOnEquity")
        debt_to_equity = info.get("debtToEquity")
        gross_margin = info.get("grossMargins")
        operating_margin = info.get("operatingMargins")
        profit_margin = info.get("profitMargins")

        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if price is None and not hist.empty and "Close" in hist.columns:
            price = float(hist["Close"].dropna().iloc[-1])

        revenue_ttm = info.get("totalRevenue")
        gross_profit_ttm = info.get("grossProfits")
        ebitda = info.get("ebitda")
        operating_income = info.get("operatingIncome")
        net_income = info.get("netIncomeToCommon")
        market_cap = info.get("marketCap")

        history_rows = []

        try:
            income_stmt = tk.income_stmt
            if income_stmt is not None and not income_stmt.empty:
                cols = list(income_stmt.columns)[-4:]
                cols = sorted(cols)

                revenues = []
                for col in cols:
                    revenue = _safe_float(income_stmt.loc["Total Revenue", col]) if "Total Revenue" in income_stmt.index else None
                    gross_profit = _safe_float(income_stmt.loc["Gross Profit", col]) if "Gross Profit" in income_stmt.index else None
                    operating_income_hist = _safe_float(income_stmt.loc["Operating Income", col]) if "Operating Income" in income_stmt.index else None
                    net_income_hist = _safe_float(income_stmt.loc["Net Income", col]) if "Net Income" in income_stmt.index else None

                    revenues.append(revenue)

                    history_rows.append({
                        "annee": int(pd.to_datetime(col).year),
                        "indice_ca": None,
                        "marge_brute": _safe_ratio(gross_profit, revenue),
                        "marge_operationnelle": _safe_ratio(operating_income_hist, revenue),
                        "marge_flux_tresorerie": _safe_ratio(net_income_hist, revenue),
                    })

                base_rev = next((r for r in revenues if r not in [None, 0]), None)
                if base_rev not in [None, 0]:
                    for i, row in enumerate(history_rows):
                        rev = revenues[i]
                        if rev is not None:
                            row["indice_ca"] = (rev / base_rev) * 100
        except Exception:
            history_rows = []

        if not history_rows:
            history_rows = [
                {"annee": 2021, "indice_ca": 100, "marge_brute": gross_margin, "marge_operationnelle": operating_margin, "marge_flux_tresorerie": profit_margin},
                {"annee": 2022, "indice_ca": 103 if revenue_growth is None else 103 * (1 + revenue_growth), "marge_brute": gross_margin, "marge_operationnelle": operating_margin, "marge_flux_tresorerie": profit_margin},
                {"annee": 2023, "indice_ca": 106 if revenue_growth is None else 106 * (1 + revenue_growth), "marge_brute": gross_margin, "marge_operationnelle": operating_margin, "marge_flux_tresorerie": profit_margin},
                {"annee": 2024, "indice_ca": 110 if revenue_growth is None else 110 * (1 + revenue_growth), "marge_brute": gross_margin, "marge_operationnelle": operating_margin, "marge_flux_tresorerie": profit_margin},
            ]

        return {
            "ticker": ticker.upper(),
            "company": info.get("shortName", ticker.upper()),
            "currency": info.get("currency", "USD"),
            "price": price,
            "pe_ratio": info.get("trailingPE"),
            "roe": roe,
            "revenue_growth": revenue_growth,
            "debt_to_equity": (debt_to_equity / 100) if debt_to_equity is not None else None,
            "gross_margin": gross_margin,
            "operating_margin": operating_margin,
            "fcf_margin": profit_margin,
            "revenue_ttm": revenue_ttm,
            "gross_profit_ttm": gross_profit_ttm,
            "ebitda_ttm": ebitda,
            "operating_income_ttm": operating_income,
            "net_income_ttm": net_income,
            "market_cap": market_cap,
            "history_rows": history_rows,
            "source": "yfinance",
            "price_source": "Yahoo Finance",
            "fundamentals_source": "Yahoo Finance",
        }
    except Exception:
        return None


def build_summary_metrics(snapshot: dict):
    currency = snapshot.get("currency", "USD")
    symbol_map = {
        "USD": "$",
        "EUR": "€",
        "CHF": "CHF",
        "DKK": "DKK",
        "GBP": "£",
    }
    currency_symbol = symbol_map.get(currency, currency)

    return {
        "price": _fmt_num(snapshot.get("price"), 2, f" {currency_symbol}") if snapshot.get("price") is not None else "Prix indisponible",
        "pe": _fmt_num(snapshot.get("pe_ratio"), 1, "x"),
        "roe": _fmt_pct(snapshot.get("roe")),
        "revenue_growth": _fmt_pct(snapshot.get("revenue_growth")),
        "debt_to_equity": _fmt_num(snapshot.get("debt_to_equity"), 2, "x"),
    }


def build_income_statement_table(snapshot: dict):
    currency = snapshot.get("currency", "USD")
    rows = [
        {
            "Indicateur": "Chiffre d'affaires",
            "Valeur": _fmt_large_amount(snapshot.get("revenue_ttm"), currency),
            "Lecture": "Mesure la taille économique actuelle de l'entreprise."
        },
        {
            "Indicateur": "Profit brut",
            "Valeur": _fmt_large_amount(snapshot.get("gross_profit_ttm"), currency),
            "Lecture": "Montre ce qu'il reste après les coûts directs de production."
        },
        {
            "Indicateur": "EBE / EBITDA",
            "Valeur": _fmt_large_amount(snapshot.get("ebitda_ttm"), currency),
            "Lecture": "Indique la performance opérationnelle avant amortissements et éléments financiers."
        },
        {
            "Indicateur": "Résultat opérationnel",
            "Valeur": _fmt_large_amount(snapshot.get("operating_income_ttm"), currency),
            "Lecture": "Mesure la rentabilité directement liée à l'activité."
        },
        {
            "Indicateur": "Résultat net",
            "Valeur": _fmt_large_amount(snapshot.get("net_income_ttm"), currency),
            "Lecture": "Correspond au bénéfice final revenant aux actionnaires."
        },
        {
            "Indicateur": "Capitalisation boursière",
            "Valeur": _fmt_large_amount(snapshot.get("market_cap"), currency),
            "Lecture": "Donne un ordre de grandeur de la valeur de marché de l'entreprise."
        },
    ]
    return pd.DataFrame(rows)


def generate_income_statement_commentary(snapshot: dict):
    revenue = snapshot.get("revenue_ttm")
    ebitda = snapshot.get("ebitda_ttm")
    net_income = snapshot.get("net_income_ttm")
    op_margin = snapshot.get("operating_margin")

    comments = []

    if revenue is not None:
        comments.append("Le chiffre d'affaires donne un premier repère sur la taille du dossier étudié.")

    if ebitda is not None:
        comments.append("L'EBE / EBITDA permet d'apprécier la capacité de l'entreprise à générer une rentabilité opérationnelle avant les éléments non cash.")

    if net_income is not None:
        comments.append("Le résultat net permet de vérifier si la rentabilité opérationnelle se traduit bien jusqu'au bas du compte de résultat.")

    if op_margin is not None:
        if op_margin >= 0.20:
            comments.append("La marge opérationnelle suggère un profil de rentabilité élevé.")
        elif op_margin >= 0.10:
            comments.append("La marge opérationnelle reste correcte sans être exceptionnelle.")
        else:
            comments.append("La marge opérationnelle reste plutôt faible, ce qui appelle davantage de prudence.")

    return " ".join(comments) if comments else "Les données disponibles sont trop partielles pour proposer une lecture complète du compte de résultat."


def generate_investment_view(snapshot: dict):
    company = snapshot.get("company", snapshot.get("ticker", "La société"))
    rg = snapshot.get("revenue_growth")
    pe = snapshot.get("pe_ratio")
    opm = snapshot.get("operating_margin")
    dte = snapshot.get("debt_to_equity")

    growth_text = "les données disponibles ne permettent pas de conclure proprement sur la dynamique de croissance"
    if rg is not None:
        if rg >= 0.12:
            growth_text = "la croissance du chiffre d'affaires demeure soutenue"
        elif rg >= 0.05:
            growth_text = "la croissance du chiffre d'affaires reste positive mais plus normalisée"
        else:
            growth_text = "la dynamique de croissance paraît plus modeste"

    margin_text = "la lecture de la profitabilité reste incomplète"
    if opm is not None:
        if opm >= 0.25:
            margin_text = "la rentabilité opérationnelle ressort à un niveau élevé"
        elif opm >= 0.15:
            margin_text = "la rentabilité opérationnelle apparaît correcte"
        else:
            margin_text = "la profitabilité opérationnelle semble plus contrainte"

    balance_text = "la structure bilancielle nécessite une analyse complémentaire"
    if dte is not None:
        if dte > 2.0:
            balance_text = "le levier financier appelle une vigilance renforcée"
        elif dte > 1.0:
            balance_text = "le bilan ne paraît pas déstabilisé mais le levier reste à surveiller"
        else:
            balance_text = "le bilan ne semble pas excessivement tendu dans les données disponibles"

    valuation_text = "la lecture de valorisation reste partielle"
    if pe is not None:
        if pe >= 30:
            valuation_text = "la valorisation paraît exigeante, ce qui limite la marge de sécurité"
        elif pe >= 20:
            valuation_text = "la valorisation semble cohérente avec un dossier de qualité, sans décote manifeste"
        else:
            valuation_text = "la valorisation paraît relativement plus abordable, sous réserve de la soutenabilité des résultats"

    return (
        f"Au vu des éléments disponibles sur {company}, {growth_text}, tandis que {margin_text}. "
        f"Par ailleurs, {balance_text}. En synthèse, {valuation_text}. "
        f"La lecture doit rester prudente : cette vue constitue un cadre d'analyse, pas une recommandation d'investissement."
    )


def generate_business_quality_commentary(snapshot: dict):
    gm = snapshot.get("gross_margin")
    opm = snapshot.get("operating_margin")
    rg = snapshot.get("revenue_growth")

    lines = []
    lines.append("**Lecture du business model**")
    lines.append("L'objectif de cette section est d'évaluer, à partir de quelques métriques simples, la qualité économique du dossier sans surinterpréter les données disponibles.")

    if gm is not None:
        if gm >= 0.45:
            lines.append("- La marge brute élevée suggère une offre différenciée, un certain pouvoir de prix ou une bonne qualité de mix produit.")
        elif gm >= 0.30:
            lines.append("- La marge brute ressort à un niveau correct, compatible avec un positionnement compétitif sans signal décisif d'avantage structurel exceptionnel.")
        else:
            lines.append("- La marge brute plus faible peut refléter un environnement plus concurrentiel ou une moindre différenciation du produit.")
    else:
        lines.append("- La marge brute n'est pas disponible ; il est donc difficile de tirer une conclusion solide sur le pouvoir de prix.")

    if opm is not None:
        if opm >= 0.25:
            lines.append("- La marge opérationnelle élevée renforce l'idée d'un modèle discipliné, capable de transformer l'activité en résultat de façon efficace.")
        elif opm >= 0.10:
            lines.append("- La marge opérationnelle suggère une rentabilité satisfaisante, mais pas nécessairement exceptionnelle au regard des meilleurs dossiers de qualité.")
        else:
            lines.append("- La profitabilité opérationnelle semble plus fragile, ce qui augmente la sensibilité du cas d'investissement à un ralentissement d'activité ou à une pression concurrentielle.")
    else:
        lines.append("- Les données de marge opérationnelle sont insuffisantes pour conclure proprement sur la qualité d'exécution.")

    if rg is not None:
        if rg >= 0.10:
            lines.append("- Le profil combine encore croissance et rentabilité, ce qui soutient généralement la perception d'un actif de qualité.")
        elif rg >= 0.04:
            lines.append("- Le dossier paraît davantage dans une phase de normalisation que dans une phase d'accélération, ce qui déplace l'attention vers la qualité des marges et du cash-flow.")
        else:
            lines.append("- En l'absence de croissance marquée, la création de valeur dépend davantage de la discipline opérationnelle, du capital alloué et de la résilience du modèle.")
    else:
        lines.append("- Faute de visibilité suffisante sur la croissance, le jugement doit rester davantage centré sur la qualité des marges et la solidité du bilan.")

    lines.append("- Cette lecture reste volontairement prudente : une analyse complète exigerait les publications, la dynamique concurrentielle, la qualité du management et l'allocation du capital.")
    return "\n".join(lines)


def _base_figure(title: str):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=AED_COLORS["navy"])),
        paper_bgcolor=AED_COLORS["cream"],
        plot_bgcolor="white",
        font=dict(color=AED_COLORS["ink"]),
        margin=dict(l=30, r=30, t=70, b=30),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor=AED_COLORS["line"], zeroline=False),
        height=320,
    )
    return fig


def build_revenue_chart(snapshot: dict):
    df = pd.DataFrame(snapshot.get("history_rows", []))

    if df.empty:
        return _base_figure("Évolution du chiffre d'affaires")

    if "annee" not in df.columns:
        if "year" in df.columns:
            df["annee"] = df["year"]
        else:
            return _base_figure("Évolution du chiffre d'affaires")

    fig = _base_figure("Évolution du chiffre d'affaires")

    if "indice_ca" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["annee"],
                y=df["indice_ca"],
                name="Indice CA",
                marker_color=AED_COLORS["navy"],
                opacity=0.92,
            )
        )
        fig.update_yaxes(title="Indice base 100")
    elif "revenue" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["annee"],
                y=df["revenue"],
                name="Chiffre d'affaires",
                marker_color=AED_COLORS["navy"],
                opacity=0.92,
            )
        )
        fig.update_yaxes(title="Chiffre d'affaires")
    else:
        fig.update_yaxes(title="")

    return fig


def build_gross_margin_chart(snapshot: dict):
    df = pd.DataFrame(snapshot.get("history_rows", []))
    fig = _base_figure("Évolution de la marge brute")

    if df.empty:
        return fig

    if "annee" not in df.columns:
        if "year" in df.columns:
            df["annee"] = df["year"]
        else:
            return fig

    y_col = None
    if "marge_brute" in df.columns:
        y_col = "marge_brute"
    elif "gross_margin" in df.columns:
        y_col = "gross_margin"

    if y_col is None:
        return fig

    fig.add_trace(
        go.Scatter(
            x=df["annee"],
            y=df[y_col],
            mode="lines+markers",
            name="Marge brute",
            line=dict(color=AED_COLORS["gold"], width=3),
            marker=dict(size=7, color=AED_COLORS["gold"]),
        )
    )
    fig.update_yaxes(title="Marge", tickformat=".0%")
    return fig


def build_operating_margin_chart(snapshot: dict):
    df = pd.DataFrame(snapshot.get("history_rows", []))
    fig = _base_figure("Évolution de la marge opérationnelle")

    if df.empty:
        return fig

    if "annee" not in df.columns:
        if "year" in df.columns:
            df["annee"] = df["year"]
        else:
            return fig

    y_col = None
    if "marge_operationnelle" in df.columns:
        y_col = "marge_operationnelle"
    elif "operating_margin" in df.columns:
        y_col = "operating_margin"

    if y_col is None:
        return fig

    fig.add_trace(
        go.Scatter(
            x=df["annee"],
            y=df[y_col],
            mode="lines+markers",
            name="Marge opérationnelle",
            line=dict(color=AED_COLORS["gold_soft"], width=3),
            marker=dict(size=7, color=AED_COLORS["gold_soft"]),
        )
    )
    fig.update_yaxes(title="Marge", tickformat=".0%")
    return fig


def build_fundamental_table(snapshot: dict):
    df = pd.DataFrame(snapshot["history_rows"]).copy()
    df = df.rename(columns={
        "annee": "Année",
        "indice_ca": "Indice du chiffre d'affaires",
        "marge_brute": "Marge brute",
        "marge_operationnelle": "Marge opérationnelle",
        "marge_flux_tresorerie": "Marge nette",
    })
    for col in ["Marge brute", "Marge opérationnelle", "Marge nette"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "N/D" if pd.isna(x) else f"{x:.1%}")
    if "Indice du chiffre d'affaires" in df.columns:
        df["Indice du chiffre d'affaires"] = df["Indice du chiffre d'affaires"].apply(lambda x: "N/D" if pd.isna(x) else f"{x:.1f}")
    return df


def generate_fundamental_commentary(snapshot: dict):
    df = pd.DataFrame(snapshot["history_rows"])
    if df.empty:
        return "Les données historiques sont insuffisantes pour proposer une lecture des fondamentaux."

    ca_debut = df["indice_ca"].iloc[0] if "indice_ca" in df.columns else None
    ca_fin = df["indice_ca"].iloc[-1] if "indice_ca" in df.columns else None
    mb = df["marge_brute"].dropna() if "marge_brute" in df.columns else pd.Series(dtype=float)
    mo = df["marge_operationnelle"].dropna() if "marge_operationnelle" in df.columns else pd.Series(dtype=float)
    mn = df["marge_flux_tresorerie"].dropna() if "marge_flux_tresorerie" in df.columns else pd.Series(dtype=float)

    comments = []

    if pd.notna(ca_debut) and pd.notna(ca_fin):
        if ca_fin > ca_debut * 1.12:
            comments.append("Le chiffre d'affaires montre une progression visible sur la période considérée.")
        elif ca_fin > ca_debut:
            comments.append("Le chiffre d'affaires progresse, mais à un rythme qui reste modéré.")
        else:
            comments.append("Le chiffre d'affaires ne montre pas de dynamique de progression évidente sur la période.")

    if not mb.empty:
        if mb.iloc[-1] >= mb.iloc[0]:
            comments.append("La marge brute se maintient ou s'améliore, ce qui va dans le sens d'un profil économique relativement solide.")
        else:
            comments.append("La marge brute s'érode légèrement, ce qui peut refléter une pression concurrentielle ou un mix moins favorable.")

    if not mo.empty:
        if mo.iloc[-1] >= 0.20:
            comments.append("La marge opérationnelle reste à un niveau confortable, ce qui soutient la qualité du dossier.")
        elif mo.iloc[-1] >= 0.10:
            comments.append("La marge opérationnelle reste correcte, sans signal de rentabilité exceptionnelle.")
        else:
            comments.append("La marge opérationnelle reste basse, ce qui appelle davantage de prudence dans la lecture du dossier.")

    if not mn.empty:
        comments.append("La marge nette permet d'apprécier ce qui descend réellement jusqu'au bénéfice final après l'ensemble des charges.")

    return " ".join(comments)


def build_valuation_table(snapshot: dict):
    pe = snapshot.get("pe_ratio")
    rows = [
        {"Ratio": "P/E", "Valeur actuelle": "N/D" if pe is None else f"{pe:.1f}x", "Zone indicative": "15x à 25x",
         "Lecture": "En dessous de cette zone, la valorisation peut paraître plus abordable ; au-dessus, le marché paie souvent une forte qualité ou une croissance élevée."},
        {"Ratio": "ROE", "Valeur actuelle": _fmt_pct(snapshot.get("roe")), "Zone indicative": "10% à 20%+",
         "Lecture": "Un ROE élevé est positif, mais il doit être interprété avec le niveau d'endettement."},
        {"Ratio": "Marge opérationnelle", "Valeur actuelle": _fmt_pct(snapshot.get("operating_margin")), "Zone indicative": "10% à 25%+ selon secteur",
         "Lecture": "Plus elle est stable et élevée, plus l'entreprise peut justifier une prime de valorisation."},
        {"Ratio": "Croissance du CA", "Valeur actuelle": _fmt_pct(snapshot.get("revenue_growth")), "Zone indicative": "5% à 10%+",
         "Lecture": "Une croissance saine soutient la thèse, à condition qu'elle ne soit pas surpayée."},
        {"Ratio": "Dette / Capitaux propres", "Valeur actuelle": _fmt_num(snapshot.get("debt_to_equity"), 2, "x"), "Zone indicative": "0.0x à 1.0x",
         "Lecture": "Un levier trop élevé peut fragiliser le dossier, même si la rentabilité reste élevée."},
    ]
    return pd.DataFrame(rows)


def dcf_scenarios(snapshot: dict, growth: float, margin: float, wacc: float, terminal: float):
    price = snapshot.get("price") or 100
    currency = snapshot.get("currency", "USD")
    symbol_map = {"USD": "$", "EUR": "€", "CHF": "CHF", "DKK": "DKK", "GBP": "£"}
    currency_symbol = symbol_map.get(currency, currency)

    base_revenue = 100
    scenarios = {
        "Prudent": (growth - 0.03, max(margin - 0.03, 0.05), wacc + 0.01, max(terminal - 0.005, 0.01)),
        "Central": (growth, margin, wacc, terminal),
        "Favorable": (growth + 0.03, min(margin + 0.03, 0.45), max(wacc - 0.01, 0.05), min(terminal + 0.005, 0.04)),
    }

    rows = []
    for name, (g, m, disc, tg) in scenarios.items():
        fcfs = []
        revenue = base_revenue
        for _ in range(5):
            revenue *= (1 + g)
            fcfs.append(revenue * m)

        terminal_value = fcfs[-1] * (1 + tg) / max(disc - tg, 0.01)
        pv = sum(fcf / ((1 + disc) ** (i + 1)) for i, fcf in enumerate(fcfs))
        pv_terminal = terminal_value / ((1 + disc) ** 5)
        equity_value = pv + pv_terminal
        implied_price = equity_value / 10
        upside = (implied_price / price) - 1

        rows.append({
            "Scénario": name,
            "Croissance retenue": f"{g:.1%}",
            "Marge de flux retenue": f"{m:.1%}",
            "Taux d'actualisation": f"{disc:.1%}",
            "Croissance à long terme": f"{tg:.1%}",
            "Valeur estimée": f"{implied_price:.2f} {currency_symbol}",
            "Écart vs cours actuel": f"{upside:.1%}",
        })
    return pd.DataFrame(rows)


def generate_dcf_commentary(snapshot: dict):
    return (
        "Le tableau ci-dessous ne donne pas une vérité absolue sur la valeur de l'action. "
        "Il montre surtout à quel point l'estimation dépend des hypothèses de croissance, "
        "de marge et de taux d'actualisation. Plus ces hypothèses sont ambitieuses, plus la valeur estimée augmente."
    )


def build_risk_commentary(snapshot: dict):
    lines = []
    pe = snapshot.get("pe_ratio")
    rg = snapshot.get("revenue_growth")
    dte = snapshot.get("debt_to_equity")

    lines.append("**Principaux points de vigilance**")

    if pe is not None and pe >= 28:
        lines.append("- **Risque de valorisation** : le multiple reste élevé, ce qui accroît mécaniquement la sensibilité du titre à une déception sur la croissance ou sur les marges.")
    else:
        lines.append("- **Risque de valorisation** : la valorisation ne paraît pas extrême dans les données disponibles, mais elle reste à confronter aux résultats futurs et au contexte de marché.")

    if rg is not None and rg < 0.04:
        lines.append("- **Risque de croissance** : une dynamique commerciale plus modeste pourrait peser sur la perception de qualité si le marché anticipe davantage.")
    else:
        lines.append("- **Risque d'exécution** : la capacité à préserver à la fois croissance et discipline de marge demeure centrale pour la thèse d'investissement.")

    if dte is not None and dte > 1.5:
        lines.append("- **Risque bilanciel** : le levier financier n'est pas négligeable et peut réduire la flexibilité stratégique ou la capacité à absorber un choc opérationnel.")
    else:
        lines.append("- **Risque bilanciel** : aucun signal d'alerte majeur ne ressort des données simplifiées, sous réserve d'une revue plus détaillée du passif et des engagements.")

    lines.append("- **Risque méthodologique** : cette lecture reste fondée sur des métriques agrégées. Une note d'investissement complète exigerait les publications, les échanges de résultats, le positionnement concurrentiel et l'analyse du management.")
    return "\n".join(lines)


def generate_investment_memo(snapshot: dict) -> dict:
    history_rows = snapshot.get("history_rows", []) or []
    latest = history_rows[-1] if history_rows else {}

    company = snapshot.get("name") or snapshot.get("company_name") or snapshot.get("ticker") or "L'entreprise"
    ticker = snapshot.get("ticker", "")

    revenue_growth = latest.get("revenue_growth")
    gross_margin = latest.get("gross_margin")
    operating_margin = latest.get("operating_margin")
    net_margin = latest.get("net_margin")
    pe = snapshot.get("pe_ratio")
    forward_pe = snapshot.get("forward_pe")
    pb = snapshot.get("price_to_book")
    roe = snapshot.get("roe")
    roic = snapshot.get("roic")
    market_cap = snapshot.get("market_cap")
    revenue = latest.get("revenue")
    net_income = latest.get("net_income")

    def pct(x):
        return isinstance(x, (int, float))

    quality_flags = 0
    if pct(gross_margin) and gross_margin >= 50:
        quality_flags += 1
    if pct(operating_margin) and operating_margin >= 20:
        quality_flags += 1
    if pct(net_margin) and net_margin >= 15:
        quality_flags += 1
    if pct(roe) and roe >= 15:
        quality_flags += 1
    if pct(roic) and roic >= 10:
        quality_flags += 1

    growth_profile = "modérée"
    if pct(revenue_growth):
        if revenue_growth >= 15:
            growth_profile = "soutenue"
        elif revenue_growth >= 6:
            growth_profile = "solide"
        elif revenue_growth < 2:
            growth_profile = "faible"

    valuation_profile = "raisonnable"
    pe_ref = forward_pe if pct(forward_pe) else pe
    if pct(pe_ref):
        if pe_ref >= 30:
            valuation_profile = "exigeante"
        elif pe_ref >= 22:
            valuation_profile = "tendue"
        elif pe_ref <= 15:
            valuation_profile = "modérée"

    profitability_profile = "correcte"
    if pct(operating_margin):
        if operating_margin >= 25:
            profitability_profile = "très élevée"
        elif operating_margin >= 15:
            profitability_profile = "solide"
        elif operating_margin < 8:
            profitability_profile = "limitée"

    if quality_flags >= 4:
        thesis = (
            f"{company} présente un profil de qualité élevée, combinant une croissance {growth_profile}, "
            f"une rentabilité {profitability_profile} et une capacité visible à transformer son activité en résultats. "
            f"La thèse d’investissement repose principalement sur la solidité du modèle économique, la résilience opérationnelle "
            f"et la faculté du groupe à prolonger sa création de valeur dans la durée."
        )
    elif quality_flags >= 2:
        thesis = (
            f"{company} présente un profil fondamental globalement solide, avec une croissance {growth_profile} "
            f"et une rentabilité {profitability_profile}. La lecture du dossier suggère une entreprise capable de créer de la valeur, "
            f"même si la visibilité n’apparaît pas aussi élevée que sur les franchises les plus dominantes du marché."
        )
    else:
        thesis = (
            f"{company} appelle une lecture plus prudente. Les fondamentaux suggèrent un profil moins robuste, "
            f"avec une croissance {growth_profile} et une rentabilité {profitability_profile}. "
            f"La thèse d’investissement dépend alors davantage d’une amélioration opérationnelle future ou d’un point d’entrée plus favorable."
        )

    business_model = (
        f"Le business model de {company} doit être apprécié à travers la qualité de ses revenus, "
        f"sa capacité à défendre ses marges et la discipline de son exécution opérationnelle. "
        f"À ce stade, le dossier affiche une rentabilité {profitability_profile}"
    )
    if pct(gross_margin):
        business_model += f", avec une marge brute d’environ {gross_margin:.1f}%"
    if pct(operating_margin):
        business_model += f" et une marge opérationnelle proche de {operating_margin:.1f}%"
    business_model += (
        ". Cette combinaison aide à juger si l’entreprise bénéficie d’un véritable pouvoir de pricing, "
        "d’un avantage concurrentiel ou d’un positionnement plus exposé à la pression concurrentielle."
    )

    drivers = []
    if pct(revenue_growth):
        if revenue_growth >= 10:
            drivers.append("la poursuite d’une croissance organique soutenue")
        elif revenue_growth >= 3:
            drivers.append("la capacité à maintenir une croissance organique régulière")
        else:
            drivers.append("une stabilisation puis un redressement du rythme de croissance")
    else:
        drivers.append("la capacité à maintenir une trajectoire d’activité lisible")

    if pct(operating_margin):
        if operating_margin >= 20:
            drivers.append("le maintien de marges opérationnelles élevées")
        elif operating_margin >= 10:
            drivers.append("une amélioration graduelle de l’efficacité opérationnelle")
        else:
            drivers.append("un redressement progressif de la rentabilité opérationnelle")
    else:
        drivers.append("une meilleure conversion de l’activité en rentabilité")

    if pct(roe) and roe >= 15:
        drivers.append("une allocation du capital disciplinée")
    elif pct(roic) and roic >= 10:
        drivers.append("la préservation d’un retour sur capital satisfaisant")
    else:
        drivers.append("une amélioration de la qualité économique des capitaux employés")

    drivers.append("une exécution cohérente permettant de renforcer la visibilité sur les résultats")

    drivers_text = "Les principaux moteurs de création de valeur résident dans " + ", ".join(drivers[:-1]) + " et " + drivers[-1] + "."

    risks = []
    if valuation_profile in ["exigeante", "tendue"]:
        risks.append("un niveau de valorisation déjà élevé par rapport au profil fondamental")
    if pct(revenue_growth) and revenue_growth < 3:
        risks.append("une croissance d’activité insuffisante pour soutenir durablement une réappréciation du titre")
    if pct(operating_margin) and operating_margin < 10:
        risks.append("une rentabilité opérationnelle encore trop limitée")
    if pct(net_margin) and net_margin < 8:
        risks.append("une conversion en résultat net encore perfectible")

    if not risks:
        risks = [
            "une déception sur la trajectoire de croissance",
            "une pression sur les marges ou sur les multiples de valorisation",
            "une baisse de la visibilité opérationnelle"
        ]

    risks_text = "Les principaux points de vigilance concernent " + ", ".join(risks[:-1]) + " et " + risks[-1] + "."

    valuation = (
        f"La valorisation ressort comme {valuation_profile}. "
        "Elle doit être mise en regard non seulement du niveau actuel de rentabilité, mais aussi de la visibilité du dossier "
        "et de la capacité de l’entreprise à prolonger sa création de valeur."
    )
    if pct(pe_ref):
        valuation += f" À titre indicatif, le marché valorise actuellement le dossier autour de {pe_ref:.1f}x les bénéfices"
        if pct(forward_pe) and forward_pe == pe_ref:
            valuation += " sur une base forward"
        valuation += "."
    else:
        valuation += " Les multiples disponibles doivent donc être lus avec prudence."

    if valuation_profile == "exigeante":
        conclusion = (
            "Dossier de qualité, mais point d’entrée exigeant. La solidité fondamentale peut justifier une prime, "
            "sans pour autant offrir une asymétrie très favorable à court terme."
        )
    elif valuation_profile == "modérée" and quality_flags >= 3:
        conclusion = (
            "Profil intéressant dans une logique de portefeuille long terme, avec une qualité fondamentale appréciable "
            "et une valorisation qui ne paraît pas excessive au regard des caractéristiques du dossier."
        )
    elif quality_flags <= 1:
        conclusion = (
            "Lecture de gestion plus prudente à ce stade. Le dossier peut devenir plus intéressant en cas d’amélioration opérationnelle "
            "ou de point d’entrée plus favorable."
        )
    else:
        conclusion = (
            "Dossier globalement cohérent, à suivre dans une logique sélective. L’intérêt du titre dépend surtout de la capacité "
            "de l’entreprise à confirmer ses fondamentaux et de la marge de sécurité offerte par le prix."
        )

    return {
        "these": thesis,
        "business_model": business_model,
        "drivers": drivers_text,
        "risks": risks_text,
        "valuation": valuation,
        "conclusion": conclusion,
    }

    abs_val = abs(value)
    if abs_val >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f} Md {symbol}"
    if abs_val >= 1_000_000:
        return f"{value/1_000_000:.1f} M {symbol}"
    return f"{value:.0f} {symbol}"


def _safe_float(x):
    try:
        if x in [None, "", "None", "N/D", "-", "null"]:
            return None
        return float(x)
    except Exception:
        return None


def _safe_ratio(num, den):
    if num is None or den in [None, 0]:
        return None
    try:
        return float(num) / float(den)
    except Exception:
        return None


def _get_yahoo_price_and_currency(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        hist = tk.history(period="1mo", interval="1d", auto_adjust=True)

        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        if price is None and not hist.empty and "Close" in hist.columns:
            price = float(hist["Close"].dropna().iloc[-1])

        currency = info.get("currency", "USD")
        company = info.get("shortName", ticker.upper())

        return {
            "price": price,
            "currency": currency,
            "company": company,
            "info": info,
        }
    except Exception:
        return {
            "price": None,
            "currency": "USD",
            "company": ticker.upper(),
            "info": {},
        }


def _get_alpha_income_statement(ticker: str, api_key: str):
    try:
        base_url = "https://www.alphavantage.co/query"
        data = requests.get(
            base_url,
            params={"function": "INCOME_STATEMENT", "symbol": ticker.upper(), "apikey": api_key},
            timeout=20,
        ).json()
        return data
    except Exception:
        return {}


def _build_history_rows_from_alpha(ticker: str, api_key: str):
    data = _get_alpha_income_statement(ticker, api_key)
    annual_reports = data.get("annualReports", [])

    if not annual_reports:
        return [], {}

    reports = []
    for report in annual_reports:
        fiscal_date = report.get("fiscalDateEnding")
        year = str(fiscal_date)[:4] if fiscal_date else None

        revenue = _safe_float(report.get("totalRevenue"))
        gross_profit = _safe_float(report.get("grossProfit"))
        ebitda = _safe_float(report.get("ebitda"))
        operating_income = _safe_float(report.get("operatingIncome"))
        net_income = _safe_float(report.get("netIncome"))

        if year is None:
            continue

        reports.append({
            "annee": int(year),
            "revenue": revenue,
            "gross_profit": gross_profit,
            "ebitda": ebitda,
            "operating_income": operating_income,
            "net_income": net_income,
            "marge_brute": _safe_ratio(gross_profit, revenue),
            "marge_operationnelle": _safe_ratio(operating_income, revenue),
            "marge_nette": _safe_ratio(net_income, revenue),
        })

    if not reports:
        return [], {}

    reports = sorted(reports, key=lambda x: x["annee"])
    reports = reports[-4:]

    base_revenue = None
    for r in reports:
        if r["revenue"] not in [None, 0]:
            base_revenue = r["revenue"]
            break

    history_rows = []
    for r in reports:
        revenue = r["revenue"]
        indice_ca = None
        if base_revenue not in [None, 0] and revenue is not None:
            indice_ca = (revenue / base_revenue) * 100

        history_rows.append({
            "annee": r["annee"],
            "indice_ca": indice_ca,
            "marge_brute": r["marge_brute"],
            "marge_operationnelle": r["marge_operationnelle"],
            "marge_flux_tresorerie": r["marge_nette"],
        })

    latest_report = reports[-1]

    latest_amounts = {
        "revenue": latest_report.get("revenue"),
        "gross_profit": latest_report.get("gross_profit"),
        "ebitda": latest_report.get("ebitda"),
        "operating_income": latest_report.get("operating_income"),
        "net_income": latest_report.get("net_income"),
        "gross_margin": latest_report.get("marge_brute"),
        "operating_margin": latest_report.get("marge_operationnelle"),
        "net_margin": latest_report.get("marge_nette"),
    }

    return history_rows, latest_amounts


def get_live_snapshot_alpha_vantage(ticker: str, api_key: str):
    yahoo_data = _get_yahoo_price_and_currency(ticker)
    info = yahoo_data.get("info", {})

    try:
        base_url = "https://www.alphavantage.co/query"

        overview = requests.get(
            base_url,
            params={"function": "OVERVIEW", "symbol": ticker.upper(), "apikey": api_key},
            timeout=20,
        ).json()

        history_rows, latest_amounts = _build_history_rows_from_alpha(ticker, api_key)

        if not overview or "Symbol" not in overview:
            fallback = build_yahoo_only_snapshot(ticker)
            if fallback is not None:
                return fallback
            return None

        pe_ratio = _safe_float(overview.get("PERatio"))

        roe = _safe_float(overview.get("ReturnOnEquityTTM"))
        if roe is not None and roe > 1:
            roe = roe / 100.0

        dte = _safe_float(overview.get("DebtToEquity"))
        if dte is not None and dte > 10:
            dte = dte / 100.0

        quarterly_growth = _safe_float(overview.get("QuarterlyRevenueGrowthYOY"))
        if quarterly_growth is not None and abs(quarterly_growth) > 1:
            quarterly_growth = quarterly_growth / 100.0

        revenue_ttm = _safe_float(overview.get("RevenueTTM"))
        gross_profit_ttm = _safe_float(overview.get("GrossProfitTTM"))
        ebitda_ttm = _safe_float(overview.get("EBITDA"))
        market_cap = _safe_float(overview.get("MarketCapitalization"))

        operating_margin_ttm = _safe_float(overview.get("OperatingMarginTTM"))
        if operating_margin_ttm is not None and operating_margin_ttm > 1:
            operating_margin_ttm = operating_margin_ttm / 100.0

        profit_margin_ttm = _safe_float(overview.get("ProfitMargin"))
        if profit_margin_ttm is not None and profit_margin_ttm > 1:
            profit_margin_ttm = profit_margin_ttm / 100.0

        gross_margin_ttm = _safe_ratio(gross_profit_ttm, revenue_ttm)

        operating_income_ttm = None
        if revenue_ttm is not None and operating_margin_ttm is not None:
            operating_income_ttm = revenue_ttm * operating_margin_ttm

        net_income_ttm = None
        if revenue_ttm is not None and profit_margin_ttm is not None:
            net_income_ttm = revenue_ttm * profit_margin_ttm

        if revenue_ttm is None:
            revenue_ttm = latest_amounts.get("revenue")
        if gross_profit_ttm is None:
            gross_profit_ttm = latest_amounts.get("gross_profit")
        if ebitda_ttm is None:
            ebitda_ttm = latest_amounts.get("ebitda")
        if operating_income_ttm is None:
            operating_income_ttm = latest_amounts.get("operating_income")
        if net_income_ttm is None:
            net_income_ttm = latest_amounts.get("net_income")

        gross_margin_display = gross_margin_ttm if gross_margin_ttm is not None else latest_amounts.get("gross_margin")
        operating_margin_display = operating_margin_ttm if operating_margin_ttm is not None else latest_amounts.get("operating_margin")
        net_margin_display = profit_margin_ttm if profit_margin_ttm is not None else latest_amounts.get("net_margin")

        if not history_rows:
            fallback = build_yahoo_only_snapshot(ticker)
            if fallback is not None:
                return fallback

        return {
            "ticker": ticker.upper(),
            "company": overview.get("Name", yahoo_data["company"]),
            "currency": yahoo_data["currency"] or overview.get("Currency", "USD"),
            "price": yahoo_data["price"],
            "pe_ratio": pe_ratio,
            "roe": roe,
            "revenue_growth": quarterly_growth,
            "debt_to_equity": dte,
            "gross_margin": gross_margin_display,
            "operating_margin": operating_margin_display,
            "fcf_margin": net_margin_display,
            "revenue_ttm": revenue_ttm,
            "gross_profit_ttm": gross_profit_ttm,
            "ebitda_ttm": ebitda_ttm,
            "operating_income_ttm": operating_income_ttm,
            "net_income_ttm": net_income_ttm,
            "market_cap": market_cap,
            "history_rows": history_rows,
            "source": "alpha_vantage_plus_yahoo",
            "price_source": "Yahoo Finance",
            "fundamentals_source": "Alpha Vantage",
        }
    except Exception:
        return build_yahoo_only_snapshot(ticker)


def build_yahoo_only_snapshot(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        hist = tk.history(period="5y", interval="1d", auto_adjust=True)

        revenue_growth = info.get("revenueGrowth")
        roe = info.get("returnOnEquity")
        debt_to_equity = info.get("debtToEquity")
        gross_margin = info.get("grossMargins")
        operating_margin = info.get("operatingMargins")
        profit_margin = info.get("profitMargins")

        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        if price is None and not hist.empty and "Close" in hist.columns:
            price = float(hist["Close"].dropna().iloc[-1])

        revenue_ttm = info.get("totalRevenue")
        gross_profit_ttm = info.get("grossProfits")
        ebitda = info.get("ebitda")
        operating_income = info.get("operatingIncome")
        net_income = info.get("netIncomeToCommon")
        market_cap = info.get("marketCap")

        history_rows = []

        try:
            income_stmt = tk.income_stmt
            if income_stmt is not None and not income_stmt.empty:
                cols = list(income_stmt.columns)[-4:]
                cols = sorted(cols)

                revenues = []
                for col in cols:
                    revenue = _safe_float(income_stmt.loc["Total Revenue", col]) if "Total Revenue" in income_stmt.index else None
                    gross_profit = _safe_float(income_stmt.loc["Gross Profit", col]) if "Gross Profit" in income_stmt.index else None
                    operating_income_hist = _safe_float(income_stmt.loc["Operating Income", col]) if "Operating Income" in income_stmt.index else None
                    net_income_hist = _safe_float(income_stmt.loc["Net Income", col]) if "Net Income" in income_stmt.index else None

                    revenues.append(revenue)

                    history_rows.append({
                        "annee": int(pd.to_datetime(col).year),
                        "indice_ca": None,
                        "marge_brute": _safe_ratio(gross_profit, revenue),
                        "marge_operationnelle": _safe_ratio(operating_income_hist, revenue),
                        "marge_flux_tresorerie": _safe_ratio(net_income_hist, revenue),
                    })

                base_rev = next((r for r in revenues if r not in [None, 0]), None)
                if base_rev not in [None, 0]:
                    for i, row in enumerate(history_rows):
                        rev = revenues[i]
                        if rev is not None:
                            row["indice_ca"] = (rev / base_rev) * 100
        except Exception:
            history_rows = []

        if not history_rows:
            history_rows = [
                {"annee": 2021, "indice_ca": 100, "marge_brute": gross_margin, "marge_operationnelle": operating_margin, "marge_flux_tresorerie": profit_margin},
                {"annee": 2022, "indice_ca": 103 if revenue_growth is None else 103 * (1 + revenue_growth), "marge_brute": gross_margin, "marge_operationnelle": operating_margin, "marge_flux_tresorerie": profit_margin},
                {"annee": 2023, "indice_ca": 106 if revenue_growth is None else 106 * (1 + revenue_growth), "marge_brute": gross_margin, "marge_operationnelle": operating_margin, "marge_flux_tresorerie": profit_margin},
                {"annee": 2024, "indice_ca": 110 if revenue_growth is None else 110 * (1 + revenue_growth), "marge_brute": gross_margin, "marge_operationnelle": operating_margin, "marge_flux_tresorerie": profit_margin},
            ]

        return {
            "ticker": ticker.upper(),
            "company": info.get("shortName", ticker.upper()),
            "currency": info.get("currency", "USD"),
            "price": price,
            "pe_ratio": info.get("trailingPE"),
            "roe": roe,
            "revenue_growth": revenue_growth,
            "debt_to_equity": (debt_to_equity / 100) if debt_to_equity is not None else None,
            "gross_margin": gross_margin,
            "operating_margin": operating_margin,
            "fcf_margin": profit_margin,
            "revenue_ttm": revenue_ttm,
            "gross_profit_ttm": gross_profit_ttm,
            "ebitda_ttm": ebitda,
            "operating_income_ttm": operating_income,
            "net_income_ttm": net_income,
            "market_cap": market_cap,
            "history_rows": history_rows,
            "source": "yfinance",
            "price_source": "Yahoo Finance",
            "fundamentals_source": "Yahoo Finance",
        }
    except Exception:
        return None


def build_summary_metrics(snapshot: dict):
    currency = snapshot.get("currency", "USD")
    symbol_map = {
        "USD": "$",
        "EUR": "€",
        "CHF": "CHF",
        "DKK": "DKK",
        "GBP": "£",
    }
    currency_symbol = symbol_map.get(currency, currency)

    return {
        "price": _fmt_num(snapshot.get("price"), 2, f" {currency_symbol}") if snapshot.get("price") is not None else "Prix indisponible",
        "pe": _fmt_num(snapshot.get("pe_ratio"), 1, "x"),
        "roe": _fmt_pct(snapshot.get("roe")),
        "revenue_growth": _fmt_pct(snapshot.get("revenue_growth")),
        "debt_to_equity": _fmt_num(snapshot.get("debt_to_equity"), 2, "x"),
    }


def build_income_statement_table(snapshot: dict):
    currency = snapshot.get("currency", "USD")
    rows = [
        {
            "Indicateur": "Chiffre d'affaires",
            "Valeur": _fmt_large_amount(snapshot.get("revenue_ttm"), currency),
            "Lecture": "Mesure la taille économique actuelle de l'entreprise."
        },
        {
            "Indicateur": "Profit brut",
            "Valeur": _fmt_large_amount(snapshot.get("gross_profit_ttm"), currency),
            "Lecture": "Montre ce qu'il reste après les coûts directs de production."
        },
        {
            "Indicateur": "EBE / EBITDA",
            "Valeur": _fmt_large_amount(snapshot.get("ebitda_ttm"), currency),
            "Lecture": "Indique la performance opérationnelle avant amortissements et éléments financiers."
        },
        {
            "Indicateur": "Résultat opérationnel",
            "Valeur": _fmt_large_amount(snapshot.get("operating_income_ttm"), currency),
            "Lecture": "Mesure la rentabilité directement liée à l'activité."
        },
        {
            "Indicateur": "Résultat net",
            "Valeur": _fmt_large_amount(snapshot.get("net_income_ttm"), currency),
            "Lecture": "Correspond au bénéfice final revenant aux actionnaires."
        },
        {
            "Indicateur": "Capitalisation boursière",
            "Valeur": _fmt_large_amount(snapshot.get("market_cap"), currency),
            "Lecture": "Donne un ordre de grandeur de la valeur de marché de l'entreprise."
        },
    ]
    return pd.DataFrame(rows)


def generate_income_statement_commentary(snapshot: dict):
    revenue = snapshot.get("revenue_ttm")
    ebitda = snapshot.get("ebitda_ttm")
    net_income = snapshot.get("net_income_ttm")
    op_margin = snapshot.get("operating_margin")

    comments = []

    if revenue is not None:
        comments.append("Le chiffre d'affaires donne un premier repère sur la taille du dossier étudié.")

    if ebitda is not None:
        comments.append("L'EBE / EBITDA permet d'apprécier la capacité de l'entreprise à générer une rentabilité opérationnelle avant les éléments non cash.")

    if net_income is not None:
        comments.append("Le résultat net permet de vérifier si la rentabilité opérationnelle se traduit bien jusqu'au bas du compte de résultat.")

    if op_margin is not None:
        if op_margin >= 0.20:
            comments.append("La marge opérationnelle suggère un profil de rentabilité élevé.")
        elif op_margin >= 0.10:
            comments.append("La marge opérationnelle reste correcte sans être exceptionnelle.")
        else:
            comments.append("La marge opérationnelle reste plutôt faible, ce qui appelle davantage de prudence.")

    return " ".join(comments) if comments else "Les données disponibles sont trop partielles pour proposer une lecture complète du compte de résultat."


def generate_investment_view(snapshot: dict):
    company = snapshot.get("company", snapshot.get("ticker", "La société"))
    rg = snapshot.get("revenue_growth")
    pe = snapshot.get("pe_ratio")
    opm = snapshot.get("operating_margin")
    dte = snapshot.get("debt_to_equity")

    growth_text = "les données disponibles ne permettent pas de conclure proprement sur la dynamique de croissance"
    if rg is not None:
        if rg >= 0.12:
            growth_text = "la croissance du chiffre d'affaires demeure soutenue"
        elif rg >= 0.05:
            growth_text = "la croissance du chiffre d'affaires reste positive mais plus normalisée"
        else:
            growth_text = "la dynamique de croissance paraît plus modeste"

    margin_text = "la lecture de la profitabilité reste incomplète"
    if opm is not None:
        if opm >= 0.25:
            margin_text = "la rentabilité opérationnelle ressort à un niveau élevé"
        elif opm >= 0.15:
            margin_text = "la rentabilité opérationnelle apparaît correcte"
        else:
            margin_text = "la profitabilité opérationnelle semble plus contrainte"

    balance_text = "la structure bilancielle nécessite une analyse complémentaire"
    if dte is not None:
        if dte > 2.0:
            balance_text = "le levier financier appelle une vigilance renforcée"
        elif dte > 1.0:
            balance_text = "le bilan ne paraît pas déstabilisé mais le levier reste à surveiller"
        else:
            balance_text = "le bilan ne semble pas excessivement tendu dans les données disponibles"

    valuation_text = "la lecture de valorisation reste partielle"
    if pe is not None:
        if pe >= 30:
            valuation_text = "la valorisation paraît exigeante, ce qui limite la marge de sécurité"
        elif pe >= 20:
            valuation_text = "la valorisation semble cohérente avec un dossier de qualité, sans décote manifeste"
        else:
            valuation_text = "la valorisation paraît relativement plus abordable, sous réserve de la soutenabilité des résultats"

    return (
        f"Au vu des éléments disponibles sur {company}, {growth_text}, tandis que {margin_text}. "
        f"Par ailleurs, {balance_text}. En synthèse, {valuation_text}. "
        f"La lecture doit rester prudente : cette vue constitue un cadre d'analyse, pas une recommandation d'investissement."
    )


def generate_business_quality_commentary(snapshot: dict):
    gm = snapshot.get("gross_margin")
    opm = snapshot.get("operating_margin")
    rg = snapshot.get("revenue_growth")

    lines = []
    lines.append("**Lecture du business model**")
    lines.append("L'objectif de cette section est d'évaluer, à partir de quelques métriques simples, la qualité économique du dossier sans surinterpréter les données disponibles.")

    if gm is not None:
        if gm >= 0.45:
            lines.append("- La marge brute élevée suggère une offre différenciée, un certain pouvoir de prix ou une bonne qualité de mix produit.")
        elif gm >= 0.30:
            lines.append("- La marge brute ressort à un niveau correct, compatible avec un positionnement compétitif sans signal décisif d'avantage structurel exceptionnel.")
        else:
            lines.append("- La marge brute plus faible peut refléter un environnement plus concurrentiel ou une moindre différenciation du produit.")
    else:
        lines.append("- La marge brute n'est pas disponible ; il est donc difficile de tirer une conclusion solide sur le pouvoir de prix.")

    if opm is not None:
        if opm >= 0.25:
            lines.append("- La marge opérationnelle élevée renforce l'idée d'un modèle discipliné, capable de transformer l'activité en résultat de façon efficace.")
        elif opm >= 0.10:
            lines.append("- La marge opérationnelle suggère une rentabilité satisfaisante, mais pas nécessairement exceptionnelle au regard des meilleurs dossiers de qualité.")
        else:
            lines.append("- La profitabilité opérationnelle semble plus fragile, ce qui augmente la sensibilité du cas d'investissement à un ralentissement d'activité ou à une pression concurrentielle.")
    else:
        lines.append("- Les données de marge opérationnelle sont insuffisantes pour conclure proprement sur la qualité d'exécution.")

    if rg is not None:
        if rg >= 0.10:
            lines.append("- Le profil combine encore croissance et rentabilité, ce qui soutient généralement la perception d'un actif de qualité.")
        elif rg >= 0.04:
            lines.append("- Le dossier paraît davantage dans une phase de normalisation que dans une phase d'accélération, ce qui déplace l'attention vers la qualité des marges et du cash-flow.")
        else:
            lines.append("- En l'absence de croissance marquée, la création de valeur dépend davantage de la discipline opérationnelle, du capital alloué et de la résilience du modèle.")
    else:
        lines.append("- Faute de visibilité suffisante sur la croissance, le jugement doit rester davantage centré sur la qualité des marges et la solidité du bilan.")

    lines.append("- Cette lecture reste volontairement prudente : une analyse complète exigerait les publications, la dynamique concurrentielle, la qualité du management et l'allocation du capital.")
    return "\n".join(lines)


def _base_figure(title: str):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=AED_COLORS["navy"])),
        paper_bgcolor=AED_COLORS["cream"],
        plot_bgcolor="white",
        font=dict(color=AED_COLORS["ink"]),
        margin=dict(l=30, r=30, t=70, b=30),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor=AED_COLORS["line"], zeroline=False),
        height=320,
    )
    return fig


def build_revenue_chart(snapshot: dict):
    df = pd.DataFrame(snapshot.get("history_rows", []))

    if df.empty:
        return _base_figure("Évolution du chiffre d'affaires")

    if "annee" not in df.columns:
        if "year" in df.columns:
            df["annee"] = df["year"]
        else:
            return _base_figure("Évolution du chiffre d'affaires")

    fig = _base_figure("Évolution du chiffre d'affaires")

    if "indice_ca" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["annee"],
                y=df["indice_ca"],
                name="Indice CA",
                marker_color=AED_COLORS["navy"],
                opacity=0.92,
            )
        )
        fig.update_yaxes(title="Indice base 100")
    elif "revenue" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["annee"],
                y=df["revenue"],
                name="Chiffre d'affaires",
                marker_color=AED_COLORS["navy"],
                opacity=0.92,
            )
        )
        fig.update_yaxes(title="Chiffre d'affaires")
    else:
        fig.update_yaxes(title="")

    return fig


def build_gross_margin_chart(snapshot: dict):
    df = pd.DataFrame(snapshot["history_rows"])
    fig = _base_figure("Évolution de la marge brute")
    fig.add_trace(go.Scatter(x=df["annee"], y=df["marge_brute"], mode="lines+markers", name="Marge brute",
                             line=dict(color=AED_COLORS["gold"], width=3), marker=dict(size=7, color=AED_COLORS["gold"])))
    fig.update_yaxes(title="Marge", tickformat=".0%")
    return fig


def build_operating_margin_chart(snapshot: dict):
    df = pd.DataFrame(snapshot["history_rows"])
    fig = _base_figure("Évolution de la marge opérationnelle")
    fig.add_trace(go.Scatter(x=df["annee"], y=df["marge_operationnelle"], mode="lines+markers", name="Marge opérationnelle",
                             line=dict(color=AED_COLORS["gold_soft"], width=3), marker=dict(size=7, color=AED_COLORS["gold_soft"])))
    fig.update_yaxes(title="Marge", tickformat=".0%")
    return fig


def build_fundamental_table(snapshot: dict):
    df = pd.DataFrame(snapshot["history_rows"]).copy()
    df = df.rename(columns={
        "annee": "Année",
        "indice_ca": "Indice du chiffre d'affaires",
        "marge_brute": "Marge brute",
        "marge_operationnelle": "Marge opérationnelle",
        "marge_flux_tresorerie": "Marge nette",
    })
    for col in ["Marge brute", "Marge opérationnelle", "Marge nette"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "N/D" if pd.isna(x) else f"{x:.1%}")
    if "Indice du chiffre d'affaires" in df.columns:
        df["Indice du chiffre d'affaires"] = df["Indice du chiffre d'affaires"].apply(lambda x: "N/D" if pd.isna(x) else f"{x:.1f}")
    return df


def generate_fundamental_commentary(snapshot: dict):
    df = pd.DataFrame(snapshot["history_rows"])
    if df.empty:
        return "Les données historiques sont insuffisantes pour proposer une lecture des fondamentaux."

    ca_debut = df["indice_ca"].iloc[0] if "indice_ca" in df.columns else None
    ca_fin = df["indice_ca"].iloc[-1] if "indice_ca" in df.columns else None
    mb = df["marge_brute"].dropna() if "marge_brute" in df.columns else pd.Series(dtype=float)
    mo = df["marge_operationnelle"].dropna() if "marge_operationnelle" in df.columns else pd.Series(dtype=float)
    mn = df["marge_flux_tresorerie"].dropna() if "marge_flux_tresorerie" in df.columns else pd.Series(dtype=float)

    comments = []

    if pd.notna(ca_debut) and pd.notna(ca_fin):
        if ca_fin > ca_debut * 1.12:
            comments.append("Le chiffre d'affaires montre une progression visible sur la période considérée.")
        elif ca_fin > ca_debut:
            comments.append("Le chiffre d'affaires progresse, mais à un rythme qui reste modéré.")
        else:
            comments.append("Le chiffre d'affaires ne montre pas de dynamique de progression évidente sur la période.")

    if not mb.empty:
        if mb.iloc[-1] >= mb.iloc[0]:
            comments.append("La marge brute se maintient ou s'améliore, ce qui va dans le sens d'un profil économique relativement solide.")
        else:
            comments.append("La marge brute s'érode légèrement, ce qui peut refléter une pression concurrentielle ou un mix moins favorable.")

    if not mo.empty:
        if mo.iloc[-1] >= 0.20:
            comments.append("La marge opérationnelle reste à un niveau confortable, ce qui soutient la qualité du dossier.")
        elif mo.iloc[-1] >= 0.10:
            comments.append("La marge opérationnelle reste correcte, sans signal de rentabilité exceptionnelle.")
        else:
            comments.append("La marge opérationnelle reste basse, ce qui appelle davantage de prudence dans la lecture du dossier.")

    if not mn.empty:
        comments.append("La marge nette permet d'apprécier ce qui descend réellement jusqu'au bénéfice final après l'ensemble des charges.")

    return " ".join(comments)


def build_valuation_table(snapshot: dict):
    pe = snapshot.get("pe_ratio")
    rows = [
        {"Ratio": "P/E", "Valeur actuelle": "N/D" if pe is None else f"{pe:.1f}x", "Zone indicative": "15x à 25x",
         "Lecture": "En dessous de cette zone, la valorisation peut paraître plus abordable ; au-dessus, le marché paie souvent une forte qualité ou une croissance élevée."},
        {"Ratio": "ROE", "Valeur actuelle": _fmt_pct(snapshot.get("roe")), "Zone indicative": "10% à 20%+",
         "Lecture": "Un ROE élevé est positif, mais il doit être interprété avec le niveau d'endettement."},
        {"Ratio": "Marge opérationnelle", "Valeur actuelle": _fmt_pct(snapshot.get("operating_margin")), "Zone indicative": "10% à 25%+ selon secteur",
         "Lecture": "Plus elle est stable et élevée, plus l'entreprise peut justifier une prime de valorisation."},
        {"Ratio": "Croissance du CA", "Valeur actuelle": _fmt_pct(snapshot.get("revenue_growth")), "Zone indicative": "5% à 10%+",
         "Lecture": "Une croissance saine soutient la thèse, à condition qu'elle ne soit pas surpayée."},
        {"Ratio": "Dette / Capitaux propres", "Valeur actuelle": _fmt_num(snapshot.get("debt_to_equity"), 2, "x"), "Zone indicative": "0.0x à 1.0x",
         "Lecture": "Un levier trop élevé peut fragiliser le dossier, même si la rentabilité reste élevée."},
    ]
    return pd.DataFrame(rows)


def dcf_scenarios(snapshot: dict, growth: float, margin: float, wacc: float, terminal: float):
    price = snapshot.get("price") or 100
    currency = snapshot.get("currency", "USD")
    symbol_map = {"USD": "$", "EUR": "€", "CHF": "CHF", "DKK": "DKK", "GBP": "£"}
    currency_symbol = symbol_map.get(currency, currency)

    base_revenue = 100
    scenarios = {
        "Prudent": (growth - 0.03, max(margin - 0.03, 0.05), wacc + 0.01, max(terminal - 0.005, 0.01)),
        "Central": (growth, margin, wacc, terminal),
        "Favorable": (growth + 0.03, min(margin + 0.03, 0.45), max(wacc - 0.01, 0.05), min(terminal + 0.005, 0.04)),
    }

    rows = []
    for name, (g, m, disc, tg) in scenarios.items():
        fcfs = []
        revenue = base_revenue
        for _ in range(5):
            revenue *= (1 + g)
            fcfs.append(revenue * m)

        terminal_value = fcfs[-1] * (1 + tg) / max(disc - tg, 0.01)
        pv = sum(fcf / ((1 + disc) ** (i + 1)) for i, fcf in enumerate(fcfs))
        pv_terminal = terminal_value / ((1 + disc) ** 5)
        equity_value = pv + pv_terminal
        implied_price = equity_value / 10
        upside = (implied_price / price) - 1

        rows.append({
            "Scénario": name,
            "Croissance retenue": f"{g:.1%}",
            "Marge de flux retenue": f"{m:.1%}",
            "Taux d'actualisation": f"{disc:.1%}",
            "Croissance à long terme": f"{tg:.1%}",
            "Valeur estimée": f"{implied_price:.2f} {currency_symbol}",
            "Écart vs cours actuel": f"{upside:.1%}",
        })
    return pd.DataFrame(rows)


def generate_dcf_commentary(snapshot: dict):
    return (
        "Le tableau ci-dessous ne donne pas une vérité absolue sur la valeur de l'action. "
        "Il montre surtout à quel point l'estimation dépend des hypothèses de croissance, "
        "de marge et de taux d'actualisation. Plus ces hypothèses sont ambitieuses, plus la valeur estimée augmente."
    )


def build_risk_commentary(snapshot: dict):
    lines = []
    pe = snapshot.get("pe_ratio")
    rg = snapshot.get("revenue_growth")
    dte = snapshot.get("debt_to_equity")

    lines.append("**Principaux points de vigilance**")

    if pe is not None and pe >= 28:
        lines.append("- **Risque de valorisation** : le multiple reste élevé, ce qui accroît mécaniquement la sensibilité du titre à une déception sur la croissance ou sur les marges.")
    else:
        lines.append("- **Risque de valorisation** : la valorisation ne paraît pas extrême dans les données disponibles, mais elle reste à confronter aux résultats futurs et au contexte de marché.")

    if rg is not None and rg < 0.04:
        lines.append("- **Risque de croissance** : une dynamique commerciale plus modeste pourrait peser sur la perception de qualité si le marché anticipe davantage.")
    else:
        lines.append("- **Risque d'exécution** : la capacité à préserver à la fois croissance et discipline de marge demeure centrale pour la thèse d'investissement.")

    if dte is not None and dte > 1.5:
        lines.append("- **Risque bilanciel** : le levier financier n'est pas négligeable et peut réduire la flexibilité stratégique ou la capacité à absorber un choc opérationnel.")
    else:
        lines.append("- **Risque bilanciel** : aucun signal d'alerte majeur ne ressort des données simplifiées, sous réserve d'une revue plus détaillée du passif et des engagements.")

    lines.append("- **Risque méthodologique** : cette lecture reste fondée sur des métriques agrégées. Une note d'investissement complète exigerait les publications, les échanges de résultats, le positionnement concurrentiel et l'analyse du management.")
    return "\n".join(lines)


def generate_investment_memo(snapshot: dict) -> dict:
    history_rows = snapshot.get("history_rows", []) or []
    latest = history_rows[-1] if history_rows else {}

    company = snapshot.get("name") or snapshot.get("company_name") or snapshot.get("ticker") or "L'entreprise"
    ticker = snapshot.get("ticker", "")

    revenue_growth = latest.get("revenue_growth")
    gross_margin = latest.get("gross_margin")
    operating_margin = latest.get("operating_margin")
    net_margin = latest.get("net_margin")
    pe = snapshot.get("pe_ratio")
    forward_pe = snapshot.get("forward_pe")
    pb = snapshot.get("price_to_book")
    roe = snapshot.get("roe")
    roic = snapshot.get("roic")
    market_cap = snapshot.get("market_cap")
    revenue = latest.get("revenue")
    net_income = latest.get("net_income")

    def pct(x):
        return isinstance(x, (int, float))

    quality_flags = 0
    if pct(gross_margin) and gross_margin >= 50:
        quality_flags += 1
    if pct(operating_margin) and operating_margin >= 20:
        quality_flags += 1
    if pct(net_margin) and net_margin >= 15:
        quality_flags += 1
    if pct(roe) and roe >= 15:
        quality_flags += 1
    if pct(roic) and roic >= 10:
        quality_flags += 1

    growth_profile = "modérée"
    if pct(revenue_growth):
        if revenue_growth >= 15:
            growth_profile = "soutenue"
        elif revenue_growth >= 6:
            growth_profile = "solide"
        elif revenue_growth < 2:
            growth_profile = "faible"

    valuation_profile = "raisonnable"
    pe_ref = forward_pe if pct(forward_pe) else pe
    if pct(pe_ref):
        if pe_ref >= 30:
            valuation_profile = "exigeante"
        elif pe_ref >= 22:
            valuation_profile = "tendue"
        elif pe_ref <= 15:
            valuation_profile = "modérée"

    profitability_profile = "correcte"
    if pct(operating_margin):
        if operating_margin >= 25:
            profitability_profile = "très élevée"
        elif operating_margin >= 15:
            profitability_profile = "solide"
        elif operating_margin < 8:
            profitability_profile = "limitée"

    if quality_flags >= 4:
        thesis = (
            f"{company} présente un profil de qualité élevée, combinant une croissance {growth_profile}, "
            f"une rentabilité {profitability_profile} et une capacité visible à transformer son activité en résultats. "
            f"La thèse d’investissement repose principalement sur la solidité du modèle économique, la résilience opérationnelle "
            f"et la faculté du groupe à prolonger sa création de valeur dans la durée."
        )
    elif quality_flags >= 2:
        thesis = (
            f"{company} présente un profil fondamental globalement solide, avec une croissance {growth_profile} "
            f"et une rentabilité {profitability_profile}. La lecture du dossier suggère une entreprise capable de créer de la valeur, "
            f"même si la visibilité n’apparaît pas aussi élevée que sur les franchises les plus dominantes du marché."
        )
    else:
        thesis = (
            f"{company} appelle une lecture plus prudente. Les fondamentaux suggèrent un profil moins robuste, "
            f"avec une croissance {growth_profile} et une rentabilité {profitability_profile}. "
            f"La thèse d’investissement dépend alors davantage d’une amélioration opérationnelle future ou d’un point d’entrée plus favorable."
        )

    business_model = (
        f"Le business model de {company} doit être apprécié à travers la qualité de ses revenus, "
        f"sa capacité à défendre ses marges et la discipline de son exécution opérationnelle. "
        f"À ce stade, le dossier affiche une rentabilité {profitability_profile}"
    )
    if pct(gross_margin):
        business_model += f", avec une marge brute d’environ {gross_margin:.1f}%"
    if pct(operating_margin):
        business_model += f" et une marge opérationnelle proche de {operating_margin:.1f}%"
    business_model += (
        ". Cette combinaison aide à juger si l’entreprise bénéficie d’un véritable pouvoir de pricing, "
        "d’un avantage concurrentiel ou d’un positionnement plus exposé à la pression concurrentielle."
    )

    drivers = []
    if pct(revenue_growth):
        if revenue_growth >= 10:
            drivers.append("la poursuite d’une croissance organique soutenue")
        elif revenue_growth >= 3:
            drivers.append("la capacité à maintenir une croissance organique régulière")
        else:
            drivers.append("une stabilisation puis un redressement du rythme de croissance")
    else:
        drivers.append("la capacité à maintenir une trajectoire d’activité lisible")

    if pct(operating_margin):
        if operating_margin >= 20:
            drivers.append("le maintien de marges opérationnelles élevées")
        elif operating_margin >= 10:
            drivers.append("une amélioration graduelle de l’efficacité opérationnelle")
        else:
            drivers.append("un redressement progressif de la rentabilité opérationnelle")
    else:
        drivers.append("une meilleure conversion de l’activité en rentabilité")

    if pct(roe) and roe >= 15:
        drivers.append("une allocation du capital disciplinée")
    elif pct(roic) and roic >= 10:
        drivers.append("la préservation d’un retour sur capital satisfaisant")
    else:
        drivers.append("une amélioration de la qualité économique des capitaux employés")

    drivers.append("une exécution cohérente permettant de renforcer la visibilité sur les résultats")

    drivers_text = "Les principaux moteurs de création de valeur résident dans " + ", ".join(drivers[:-1]) + " et " + drivers[-1] + "."

    risks = []
    if valuation_profile in ["exigeante", "tendue"]:
        risks.append("un niveau de valorisation déjà élevé par rapport au profil fondamental")
    if pct(revenue_growth) and revenue_growth < 3:
        risks.append("une croissance d’activité insuffisante pour soutenir durablement une réappréciation du titre")
    if pct(operating_margin) and operating_margin < 10:
        risks.append("une rentabilité opérationnelle encore trop limitée")
    if pct(net_margin) and net_margin < 8:
        risks.append("une conversion en résultat net encore perfectible")

    if not risks:
        risks = [
            "une déception sur la trajectoire de croissance",
            "une pression sur les marges ou sur les multiples de valorisation",
            "une baisse de la visibilité opérationnelle"
        ]

    risks_text = "Les principaux points de vigilance concernent " + ", ".join(risks[:-1]) + " et " + risks[-1] + "."

    valuation = (
        f"La valorisation ressort comme {valuation_profile}. "
        "Elle doit être mise en regard non seulement du niveau actuel de rentabilité, mais aussi de la visibilité du dossier "
        "et de la capacité de l’entreprise à prolonger sa création de valeur."
    )
    if pct(pe_ref):
        valuation += f" À titre indicatif, le marché valorise actuellement le dossier autour de {pe_ref:.1f}x les bénéfices"
        if pct(forward_pe) and forward_pe == pe_ref:
            valuation += " sur une base forward"
        valuation += "."
    else:
        valuation += " Les multiples disponibles doivent donc être lus avec prudence."

    if valuation_profile == "exigeante":
        conclusion = (
            "Dossier de qualité, mais point d’entrée exigeant. La solidité fondamentale peut justifier une prime, "
            "sans pour autant offrir une asymétrie très favorable à court terme."
        )
    elif valuation_profile == "modérée" and quality_flags >= 3:
        conclusion = (
            "Profil intéressant dans une logique de portefeuille long terme, avec une qualité fondamentale appréciable "
            "et une valorisation qui ne paraît pas excessive au regard des caractéristiques du dossier."
        )
    elif quality_flags <= 1:
        conclusion = (
            "Lecture de gestion plus prudente à ce stade. Le dossier peut devenir plus intéressant en cas d’amélioration opérationnelle "
            "ou de point d’entrée plus favorable."
        )
    else:
        conclusion = (
            "Dossier globalement cohérent, à suivre dans une logique sélective. L’intérêt du titre dépend surtout de la capacité "
            "de l’entreprise à confirmer ses fondamentaux et de la marge de sécurité offerte par le prix."
        )

    return {
        "these": thesis,
        "business_model": business_model,
        "drivers": drivers_text,
        "risks": risks_text,
        "valuation": valuation,
        "conclusion": conclusion,
    }
