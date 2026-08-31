import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import pandas as pd
import streamlit as st

from utils.financial_analysis import (
    get_live_snapshot_alpha_vantage,
    build_summary_metrics,
    generate_investment_view,
    generate_business_quality_commentary,
    build_revenue_chart,
    build_gross_margin_chart,
    build_operating_margin_chart,
    build_fundamental_table,
    generate_fundamental_commentary,
    build_income_statement_table,
    generate_income_statement_commentary,
    build_valuation_table,
    dcf_scenarios,
    generate_dcf_commentary,
    build_risk_commentary,
    generate_conclusion,
)
from utils.portfolio import (
    BENCHMARKS,
    load_demo_portfolio,
    fetch_price_history,
    compute_portfolio_analytics,
    simulate_efficient_frontier,
)
from utils.pedagogy import PEDAGOGY_CONTENT

st.set_page_config(
    page_title="AED Equity",
    page_icon="AE",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

        .stApp {
            background: #f6f4ef;
            color: #0f172a;
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        header[data-testid="stHeader"] {
            background: rgba(248, 246, 241, 0.92);
        }

        .block-container {
            padding-top: 1.2rem;
            max-width: 1380px;
        }

        .aed-topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.6rem 0 1.2rem 0;
            border-bottom: 1px solid rgba(12, 26, 42, 0.10);
            margin-bottom: 1.2rem;
        }

        .aed-brand {
            font-family: 'Cormorant Garamond', serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: #0b1a2b;
            letter-spacing: 0.02em;
            line-height: 1;
        }

        .aed-tagline {
            color: #8a6a2f;
            font-size: 0.88rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 600;
            text-align: right;
        }

        .hero-card {
            background: #0f1e30;
            padding: 2rem 2.2rem;
            border-radius: 8px;
            color: #f8f6f1;
            border: 1px solid rgba(176, 139, 64, 0.18);
            margin-bottom: 1.4rem;
        }

        .hero-title {
            font-family: 'Cormorant Garamond', serif;
            font-size: 2.6rem;
            font-weight: 600;
            line-height: 1.05;
            margin-bottom: 0.55rem;
            color: #f7f2e8;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            color: #d8d2c5;
            max-width: 880px;
        }

        .section-title {
            font-family: 'Cormorant Garamond', serif;
            font-size: 2rem;
            font-weight: 700;
            color: #0b1a2b;
            margin-top: 0.3rem;
            margin-bottom: 0.8rem;
            letter-spacing: 0.01em;
        }

        .subsection-title {
            font-family: 'Cormorant Garamond', serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: #0b1a2b;
            margin-top: 0.2rem;
            margin-bottom: 0.55rem;
        }

        .glass-card {
            background: #ffffff;
            border: 1px solid rgba(11, 26, 43, 0.10);
            border-radius: 8px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            margin-bottom: 1rem;
        }

        .quote-card {
            background: #fcfaf5;
            border-left: 3px solid #b08b40;
            border-radius: 4px;
            padding: 1rem 1.1rem;
            color: #0b1a2b;
            margin-bottom: 0.8rem;
        }

        .small-muted {
            color: #5c6673;
            font-size: 0.96rem;
        }

        .pill {
            display: inline-block;
            padding: 0.3rem 0.6rem;
            border-radius: 4px;
            background: transparent;
            color: #6b5730;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
            border: 1px solid rgba(176, 139, 64, 0.35);
        }

        div[data-testid="metric-container"] {
            background: #ffffff;
            border: 1px solid rgba(11, 26, 43, 0.10);
            padding: 0.95rem;
            border-radius: 8px;
        }

        div[data-testid="metric-container"] label {
            color: #6b7280 !important;
            font-weight: 600 !important;
        }

        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #0b1a2b !important;
            font-weight: 700 !important;
        }

        div[data-testid="stTabs"] button {
            font-weight: 600;
            color: #0b1a2b;
            border-radius: 10px 10px 0 0;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #8a6a2f;
        }

        div.stButton > button {
            background: #0f1e30;
            color: #f8f6f1;
            border: 1px solid rgba(176, 139, 64, 0.35);
            border-radius: 6px;
            padding: 0.62rem 1.2rem;
            font-weight: 600;
            transition: background 0.15s ease;
        }

        div.stButton > button:hover {
            background: #16283d;
            border: 1px solid rgba(176, 139, 64, 0.55);
            color: #ffffff;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        .stTextInput > div > div,
        .stNumberInput > div > div {
            background: #ffffff !important;
            border-radius: 6px !important;
            border: 1px solid rgba(11, 26, 43, 0.14) !important;
        }

        .stDataFrame, .stTable {
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid rgba(11, 26, 43, 0.12);
        }

        [data-testid="stDataFrame"] div[role="grid"] {
            border-radius: 6px;
        }

        [data-testid="stDataFrame"] [role="columnheader"] {
            background: #f4efe4 !important;
            color: #0b1a2b !important;
            font-weight: 700 !important;
            border-bottom: 1px solid rgba(176, 139, 64, 0.18) !important;
        }

        [data-testid="stDataFrame"] [role="gridcell"] {
            background: #ffffff !important;
            color: #253241 !important;
            border-bottom: 1px solid rgba(11, 26, 43, 0.06) !important;
        }

        [data-testid="stDataEditor"] {
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid rgba(11, 26, 43, 0.12);
        }

        .stAlert {
            border-radius: 6px !important;
            border: 1px solid rgba(11, 26, 43, 0.10) !important;
        }

        .stInfo {
            background: rgba(255,255,255,0.72) !important;
        }

        .stWarning {
            background: #faf5e8 !important;
        }

        .stSuccess {
            background: #f6f4ee !important;
        }

        hr {
            border: none;
            border-top: 1px solid rgba(11, 26, 43, 0.10);
            margin-top: 2rem;
            margin-bottom: 1.2rem;
        }

        .footer-note {
            text-align: center;
            color: #5c6673;
            font-size: 0.9rem;
            padding-top: 1rem;
            padding-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")

st.markdown(
    """
    <div class="aed-topbar">
        <div class="aed-brand">AED Equity</div>
        <div class="aed-tagline">Equity Research • Portfolio Construction • Risk Discipline</div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_home, nav_analysis, nav_portfolio, nav_pedagogy = st.tabs(
    ["Accueil", "Analyse action", "Gestion portefeuille", "Base pédagogique"]
)

with nav_home:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">AED Equity</div>
            <div class="hero-subtitle">
                Projet personnel construit autour d’une conviction simple :
                une analyse actions crédible repose sur l’articulation rigoureuse entre qualité du business,
                valorisation, discipline du risque et construction de portefeuille.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Intention du projet</div>
            <p style="margin-top:0; color:#0b1a2b;">
                Je m’intéresse particulièrement à la <strong>gestion de portefeuille actions</strong>, à la manière
                dont une thèse d’investissement se construit, se teste, puis se traduit en position dans un portefeuille.
                AED Equity est un projet qui me tenait à cœur, pensé comme un support de travail et d’apprentissage,
                avec l’ambition de réunir dans un même cadre une lecture fondamentale, une logique de valorisation
                et des réflexes simples de gestion du risque.
            </p>
            <p class="small-muted" style="margin-bottom:0;">
                L’objectif n’est pas de produire un outil spectaculaire, mais de présenter une méthode de raisonnement :
                observer les données, expliciter les hypothèses, rester prudent dans les conclusions
                et relier chaque jugement à une logique d’investissement.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Analyse fondamentale</div>
                <div class="small-muted">
                    Résumé exécutif, lecture du business model, fondamentaux, valorisation et points de vigilance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Construction de portefeuille</div>
                <div class="small-muted">
                    Rendement, volatilité, Sharpe, drawdown, corrélations et première lecture de la diversification.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Base de connaissances</div>
                <div class="small-muted">
                    Concepts-clés d’un futur gérant actions : ROIC, DCF, VaR, Markowitz, sizing et discipline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Principes de travail")
    a1, a2 = st.columns([1.2, 1])

    with a1:
        st.markdown(
            """
            <div class="glass-card">
                <span class="pill">Analyse fondamentale</span>
                <span class="pill">Valorisation</span>
                <span class="pill">Risque</span>
                <span class="pill">Sizing</span>
                <span class="pill">Diversification</span>
                <span class="pill">Discipline</span>
                <p class="small-muted" style="margin-top:0.9rem; margin-bottom:0;">
                    Cette application reste un outil pédagogique. En mode démo, certaines données sont simplifiées.
                    L’ambition n’est pas de simuler un terminal professionnel, mais de rendre visible une structure
                    de réflexion cohérente avec un apprentissage du métier de gérant actions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            """
            <div class="quote-card">
                <div style="font-size:1.02rem; font-weight:600;">“You can’t predict. You can prepare.”</div>
                <div class="small-muted" style="margin-top:0.35rem;">— Howard Marks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="quote-card">
                <div style="font-size:1.02rem; font-weight:600;">“Monthly or yearly movements of stocks are often erratic and not indicative of changes in intrinsic value.”</div>
                <div class="small-muted" style="margin-top:0.35rem;">— Warren Buffett</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with nav_analysis:
    st.markdown('<div class="section-title">Analyse fondamentale d’une action</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        ticker = st.text_input("Ticker", value="MSFT").upper().strip()
    with c2:
        run = st.button("Lancer l'analyse", type="primary")

    if run:
        snapshot = get_live_snapshot_alpha_vantage(ticker, API_KEY)

        if snapshot is None:
            st.error("Impossible de récupérer les données pour ce ticker.")
        else:
            summary = build_summary_metrics(snapshot)
            st.markdown('<div class="subsection-title">Résumé exécutif</div>', unsafe_allow_html=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Prix", summary["price"])
            m2.metric("P/E", summary["pe"])
            m3.metric("ROE", summary["roe"])
            m4.metric("Croissance CA", summary["revenue_growth"])
            m5.metric("Dette / Capitaux propres", summary["debt_to_equity"])

            st.markdown("### Vue d'investissement")
            st.info(generate_investment_view(snapshot))

            tabs = st.tabs([
                "Business model",
                "Fondamentaux",
                "Valorisation",
                "DCF (scénarios)",
                "Risques",
                "Conclusion",
            ])

            with tabs[0]:
                st.markdown('<div class="subsection-title">Qualité du business model</div>', unsafe_allow_html=True)
                st.markdown(generate_business_quality_commentary(snapshot))

            with tabs[1]:
                st.markdown('<div class="subsection-title">Indicateurs fondamentaux</div>', unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.plotly_chart(build_revenue_chart(snapshot), width="stretch")
                with c2:
                    st.plotly_chart(build_gross_margin_chart(snapshot), width="stretch")
                with c3:
                    st.plotly_chart(build_operating_margin_chart(snapshot), width="stretch")

                st.markdown('<div class="subsection-title">Lecture des fondamentaux</div>', unsafe_allow_html=True)
                st.info(generate_fundamental_commentary(snapshot))

                st.markdown('<div class="subsection-title">Tableau récapitulatif</div>', unsafe_allow_html=True)
                st.dataframe(build_fundamental_table(snapshot), width="stretch")

                st.markdown('<div class="subsection-title">Principaux chiffres du compte de résultat</div>', unsafe_allow_html=True)
                st.dataframe(build_income_statement_table(snapshot), width="stretch")
                st.info(generate_income_statement_commentary(snapshot))

            with tabs[2]:
                st.markdown('<div class="subsection-title">Valorisation</div>', unsafe_allow_html=True)
                valuation_df = build_valuation_table(snapshot)
                st.dataframe(
                    valuation_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Lecture": st.column_config.TextColumn("Lecture", width="large"),
                    },
                )

            with tabs[3]:
                st.markdown('<div class="subsection-title">Valorisation par DCF (scénarios)</div>', unsafe_allow_html=True)
                st.caption(
                    "Exercice pédagogique de sensibilité : la valeur estimée dépend entièrement des hypothèses "
                    "choisies ci-dessous, pas d'une projection financière détaillée de l'entreprise. "
                    "L'objectif est de visualiser l'écart entre un scénario prudent, central et favorable."
                )

                history_rows = snapshot.get("history_rows", []) or []
                latest_row = history_rows[-1] if history_rows else {}
                default_growth = latest_row.get("revenue_growth")
                default_margin = latest_row.get("operating_margin")

                d1, d2, d3, d4 = st.columns(4)
                with d1:
                    growth_input = st.slider(
                        "Croissance annuelle retenue",
                        min_value=-10.0, max_value=40.0,
                        value=round((default_growth or 0.08) * 100, 1),
                        step=0.5, format="%.1f%%",
                        key="dcf_growth",
                    ) / 100
                with d2:
                    margin_input = st.slider(
                        "Marge de flux retenue",
                        min_value=1.0, max_value=45.0,
                        value=round(max(default_margin or 0.15, 0.01) * 100, 1),
                        step=0.5, format="%.1f%%",
                        key="dcf_margin",
                    ) / 100
                with d3:
                    wacc_input = st.slider(
                        "Taux d'actualisation (WACC)",
                        min_value=4.0, max_value=15.0,
                        value=9.0, step=0.25, format="%.2f%%",
                        key="dcf_wacc",
                    ) / 100
                with d4:
                    terminal_input = st.slider(
                        "Croissance à long terme",
                        min_value=0.0, max_value=4.0,
                        value=2.5, step=0.25, format="%.2f%%",
                        key="dcf_terminal",
                    ) / 100

                dcf_df = dcf_scenarios(snapshot, growth_input, margin_input, wacc_input, terminal_input)
                st.dataframe(dcf_df, width="stretch", hide_index=True)
                st.info(generate_dcf_commentary(snapshot))

            with tabs[4]:
                st.markdown('<div class="subsection-title">Principaux risques</div>', unsafe_allow_html=True)
                st.markdown(build_risk_commentary(snapshot))

            with tabs[5]:
                st.markdown('<div class="subsection-title">Conclusion</div>', unsafe_allow_html=True)
                st.success(generate_conclusion(snapshot, dcf_df))
                st.markdown(
                    """
                    **Cadre méthodologique**
                    - Les commentaires sont produits à partir des métriques disponibles.
                    - Les données manquantes sont laissées en N/D.
                    - La conclusion est une lecture analytique, pas un conseil d'investissement personnalisé.
                    """
                )

with nav_portfolio:
    st.markdown('<div class="section-title">Gestion de portefeuille actions</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        default_df = load_demo_portfolio()
        edited = st.data_editor(default_df, num_rows="dynamic", width="stretch")
    with c2:
        benchmark_name = st.selectbox("Benchmark", list(BENCHMARKS.keys()))
        benchmark_symbol = BENCHMARKS[benchmark_name]

    if edited.empty or "ticker" not in edited.columns or "weight" not in edited.columns:
        st.error("Le portefeuille doit contenir au moins les colonnes 'ticker' et 'weight'.")
    else:
        weights_sum = float(pd.to_numeric(edited["weight"], errors="coerce").fillna(0).sum())

        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Somme des poids", f"{weights_sum:.2%}")
        h2.metric("Benchmark", benchmark_name)
        h3.metric("Nombre de lignes", f"{len(edited)}")
        h4.metric("Poids max", f"{pd.to_numeric(edited['weight'], errors='coerce').fillna(0).max():.2%}")
        h5.metric("Poids min", f"{pd.to_numeric(edited['weight'], errors='coerce').fillna(0).min():.2%}")

        if abs(weights_sum - 1.0) > 0.02:
            st.warning("Les poids devraient idéalement totaliser 100%.")

        tickers = edited["ticker"].astype(str).str.upper().tolist()
        weights = pd.to_numeric(edited["weight"], errors="coerce").fillna(0).to_numpy()

        with st.spinner("Récupération de l'historique de prix..."):
            prices = fetch_price_history(tickers)

        if prices is None or prices.empty or prices.shape[1] < 2:
            st.error("Pas assez de données de marché pour calculer les métriques du portefeuille.")
        else:
            analytics = compute_portfolio_analytics(prices, weights, benchmark_symbol=benchmark_symbol)

            a1, a2, a3, a4, a5 = st.columns(5)
            a1.metric("Rendement annualisé", analytics["annual_return"])
            a2.metric("Volatilité annualisée", analytics["annual_vol"])
            a3.metric("Sharpe", analytics["sharpe"])
            a4.metric("Bêta", analytics["beta"])
            a5.metric("Alpha", analytics["alpha"])

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Tracking error", analytics["tracking_error"])
            b2.metric("Ratio d'information", analytics["info_ratio"])
            b3.metric("VaR 95% (1j)", analytics["var_95"])
            b4.metric("Max drawdown", analytics["max_drawdown"])

            for alert in analytics["alerts"]:
                st.warning(alert)

            tabs = st.tabs(["Vue d'ensemble", "Expositions", "Risque", "Corrélations", "Optimisation"])

            with tabs[0]:
                st.markdown('<div class="subsection-title">Portefeuille vs benchmark</div>', unsafe_allow_html=True)
                if analytics["relative_chart"] is not None:
                    st.plotly_chart(analytics["relative_chart"], width="stretch")
                else:
                    st.plotly_chart(analytics["cum_chart"], width="stretch")

                st.markdown('<div class="subsection-title">Composition du portefeuille</div>', unsafe_allow_html=True)
                st.dataframe(analytics["composition_df"], width="stretch")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="subsection-title">Contribution à la performance</div>', unsafe_allow_html=True)
                    st.dataframe(analytics["contrib_perf_df"], width="stretch")
                with c2:
                    st.markdown('<div class="subsection-title">Contribution au risque</div>', unsafe_allow_html=True)
                    st.dataframe(analytics["contrib_risk_df"], width="stretch")

            with tabs[1]:
                st.markdown('<div class="subsection-title">Exposition sectorielle</div>', unsafe_allow_html=True)
                st.plotly_chart(analytics["sector_chart"], width="stretch")
                st.dataframe(analytics["sector_exposure"], width="stretch")

                st.markdown('<div class="subsection-title">Exposition géographique</div>', unsafe_allow_html=True)
                st.plotly_chart(analytics["region_chart"], width="stretch")
                st.dataframe(analytics["region_exposure"], width="stretch")

            with tabs[2]:
                st.markdown('<div class="subsection-title">Mesures de risque</div>', unsafe_allow_html=True)
                st.dataframe(analytics["risk_table"], width="stretch")

            with tabs[3]:
                st.markdown('<div class="subsection-title">Matrice de corrélation</div>', unsafe_allow_html=True)
                st.plotly_chart(analytics["corr_chart"], width="stretch")

            with tabs[4]:
                st.markdown('<div class="subsection-title">Frontière efficiente simulée</div>', unsafe_allow_html=True)
                frontier = simulate_efficient_frontier(prices, n_portfolios=2000)
                st.plotly_chart(frontier["chart"], width="stretch")
                st.dataframe(frontier["top_portfolios"], width="stretch")

with nav_pedagogy:
    st.markdown('<div class="section-title">Base pédagogique — Gérant actions</div>', unsafe_allow_html=True)

    concept = st.selectbox("Choisir un concept", list(PEDAGOGY_CONTENT.keys()))
    item = PEDAGOGY_CONTENT[concept]

    st.subheader(concept)
    st.markdown(f"**Définition** : {item['definition']}")
    st.markdown(f"**Intuition** : {item['intuition']}")
    st.markdown(f"**Formule / cadre** : {item['formula']}")
    st.markdown(f"**Usage concret** : {item['practical_use']}")
    st.markdown(f"**Erreur fréquente** : {item['common_mistake']}")

st.markdown("---")
st.markdown(
    f'<div class="footer-note">Mis à jour le {datetime.now().strftime("%d/%m/%Y")} — AED Equity, projet personnel orienté analyse actions et gestion de portefeuille.</div>',
    unsafe_allow_html=True,
)
