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
            background: linear-gradient(180deg, #f8f6f1 0%, #f2efe8 100%);
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
            background: linear-gradient(135deg, #0b1a2b 0%, #16283d 100%);
            padding: 2.2rem;
            border-radius: 22px;
            color: #f8f6f1;
            border: 1px solid rgba(176, 139, 64, 0.22);
            box-shadow: 0 24px 48px rgba(11, 26, 43, 0.18);
            margin-bottom: 1.4rem;
        }

        .hero-title {
            font-family: 'Cormorant Garamond', serif;
            font-size: 3.2rem;
            font-weight: 700;
            line-height: 1.02;
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
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(11, 26, 43, 0.08);
            border-radius: 18px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            box-shadow: 0 10px 26px rgba(11, 26, 43, 0.05);
            margin-bottom: 1rem;
        }

        .quote-card {
            background: #fcfaf5;
            border-left: 4px solid #b08b40;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 6px 18px rgba(11, 26, 43, 0.04);
            color: #0b1a2b;
            margin-bottom: 0.8rem;
        }

        .small-muted {
            color: #5c6673;
            font-size: 0.96rem;
        }

        .pill {
            display: inline-block;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            background: rgba(176, 139, 64, 0.12);
            color: #7a5b21;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
            border: 1px solid rgba(176, 139, 64, 0.18);
        }

        div[data-testid="metric-container"] {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(11, 26, 43, 0.08);
            padding: 0.95rem;
            border-radius: 16px;
            box-shadow: 0 6px 18px rgba(11, 26, 43, 0.04);
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
            background: linear-gradient(135deg, #0b1a2b 0%, #16283d 100%);
            color: #f8f6f1;
            border: 1px solid rgba(176, 139, 64, 0.35);
            border-radius: 12px;
            padding: 0.62rem 1.2rem;
            font-weight: 600;
            box-shadow: 0 8px 18px rgba(11, 26, 43, 0.12);
            transition: all 0.2s ease;
        }

        div.stButton > button:hover {
            border: 1px solid rgba(176, 139, 64, 0.65);
            box-shadow: 0 10px 22px rgba(11, 26, 43, 0.18);
            transform: translateY(-1px);
            color: #ffffff;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        .stTextInput > div > div,
        .stNumberInput > div > div {
            background: rgba(255,255,255,0.82) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(11, 26, 43, 0.10) !important;
            box-shadow: 0 4px 12px rgba(11, 26, 43, 0.03) !important;
        }

        .stDataFrame, .stTable {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(11, 26, 43, 0.08);
            box-shadow: 0 10px 24px rgba(11, 26, 43, 0.04);
        }

        [data-testid="stDataFrame"] div[role="grid"] {
            border-radius: 16px;
        }

        [data-testid="stDataFrame"] [role="columnheader"] {
            background: #f4efe4 !important;
            color: #0b1a2b !important;
            font-weight: 700 !important;
            border-bottom: 1px solid rgba(176, 139, 64, 0.18) !important;
        }

        [data-testid="stDataFrame"] [role="gridcell"] {
            background: rgba(255,255,255,0.92) !important;
            color: #253241 !important;
            border-bottom: 1px solid rgba(11, 26, 43, 0.05) !important;
        }

        [data-testid="stDataEditor"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(11, 26, 43, 0.08);
            box-shadow: 0 10px 24px rgba(11, 26, 43, 0.04);
        }

        .stAlert {
            border-radius: 16px !important;
            border: 1px solid rgba(11, 26, 43, 0.08) !important;
            box-shadow: 0 8px 20px rgba(11, 26, 43, 0.04) !important;
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

        st.caption(f"Debug — API key présente: {'oui' if API_KEY else 'non'}")
        st.caption(f"Debug — snapshot vide: {'oui' if snapshot is None else 'non'}")
        if snapshot is not None:
            st.caption(f"Debug — company: {snapshot.get('company')}")
            st.caption(f"Debug — price: {snapshot.get('price')}")
            st.caption(f"Debug — source: {snapshot.get('source')}")
            st.caption(f"Debug — fundamentals_source: {snapshot.get('fundamentals_source')}")
            st.caption(f"Debug — price_source: {snapshot.get('price_source')}")
            st.caption(f"Debug — history_rows: {len(snapshot.get('history_rows', [])) if snapshot.get('history_rows') is not None else 'None'}")
            st.caption(f"Debug — revenue_ttm: {snapshot.get('revenue_ttm')}")
            st.caption(f"Debug — gross_margin: {snapshot.get('gross_margin')}")
            st.caption(f"Debug — operating_margin: {snapshot.get('operating_margin')}")

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
                    st.plotly_chart(build_revenue_chart(snapshot), use_container_width=True)
                with c2:
                    st.plotly_chart(build_gross_margin_chart(snapshot), use_container_width=True)
                with c3:
                    st.plotly_chart(build_operating_margin_chart(snapshot), use_container_width=True)

                st.markdown('<div class="subsection-title">Lecture des fondamentaux</div>', unsafe_allow_html=True)
                st.info(generate_fundamental_commentary(snapshot))

                st.markdown('<div class="subsection-title">Tableau récapitulatif</div>', unsafe_allow_html=True)
                st.dataframe(build_fundamental_table(snapshot), use_container_width=True)

                st.markdown('<div class="subsection-title">Principaux chiffres du compte de résultat</div>', unsafe_allow_html=True)
                st.dataframe(build_income_statement_table(snapshot), use_container_width=True)
                st.info(generate_income_statement_commentary(snapshot))

            with tabs[2]:
                st.markdown('<div class="subsection-title">Valorisation</div>', unsafe_allow_html=True)
                valuation_df = build_valuation_table(snapshot)
                st.dataframe(valuation_df, use_container_width=True)

            with tabs[3]:
                st.markdown('<div class="subsection-title">Principaux risques</div>', unsafe_allow_html=True)
                st.markdown(build_risk_commentary(snapshot))

            with tabs[4]:
                st.markdown('<div class="subsection-title">Conclusion</div>', unsafe_allow_html=True)
                st.success(generate_investment_view(snapshot))
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
        edited = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)
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
                    st.plotly_chart(analytics["relative_chart"], use_container_width=True)
                else:
                    st.plotly_chart(analytics["cum_chart"], use_container_width=True)

                st.markdown('<div class="subsection-title">Composition du portefeuille</div>', unsafe_allow_html=True)
                st.dataframe(analytics["composition_df"], use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="subsection-title">Contribution à la performance</div>', unsafe_allow_html=True)
                    st.dataframe(analytics["contrib_perf_df"], use_container_width=True)
                with c2:
                    st.markdown('<div class="subsection-title">Contribution au risque</div>', unsafe_allow_html=True)
                    st.dataframe(analytics["contrib_risk_df"], use_container_width=True)

            with tabs[1]:
                st.markdown('<div class="subsection-title">Exposition sectorielle</div>', unsafe_allow_html=True)
                st.plotly_chart(analytics["sector_chart"], use_container_width=True)
                st.dataframe(analytics["sector_exposure"], use_container_width=True)

                st.markdown('<div class="subsection-title">Exposition géographique</div>', unsafe_allow_html=True)
                st.plotly_chart(analytics["region_chart"], use_container_width=True)
                st.dataframe(analytics["region_exposure"], use_container_width=True)

            with tabs[2]:
                st.markdown('<div class="subsection-title">Mesures de risque</div>', unsafe_allow_html=True)
                st.dataframe(analytics["risk_table"], use_container_width=True)

            with tabs[3]:
                st.markdown('<div class="subsection-title">Matrice de corrélation</div>', unsafe_allow_html=True)
                st.plotly_chart(analytics["corr_chart"], use_container_width=True)

            with tabs[4]:
                st.markdown('<div class="subsection-title">Frontière efficiente simulée</div>', unsafe_allow_html=True)
                frontier = simulate_efficient_frontier(prices, n_portfolios=2000)
                st.plotly_chart(frontier["chart"], use_container_width=True)
                st.dataframe(frontier["top_portfolios"], use_container_width=True)

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
