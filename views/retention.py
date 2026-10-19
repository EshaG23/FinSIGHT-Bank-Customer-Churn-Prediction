import plotly.graph_objects as go
import streamlit as st

import analytics
import theme
import ui


def concentration_tab(df, rates) -> None:
    plays = analytics.retention_plays(rates)
    ui.callout("<b>Read these as associations.</b> They describe who left in this dataset. They do not prove that a trait causes "
               "churn, and groups overlap, so the same customer can appear in several of them.")
    st.write("")
    if not plays.empty:
        cols = st.columns(len(plays))
        for col, row in zip(cols, plays.itertuples()):
            col.markdown(
                ui.kpi_card(f"{analytics.DIMENSION_TITLES[row.Dimension]}: {row.Segment}", f"{row.ChurnRate * 100:.0f}%",
                            f"churned. {row.Customers:,} customers, about {row.ExcessChurners:.0f} more leavers than average.",
                            "alert", "high"), unsafe_allow_html=True)
    st.write("")
    dim = st.selectbox("Explore a dimension", analytics.DIMENSIONS, format_func=lambda d: analytics.DIMENSION_TITLES[d], key="ret_dim")
    table = analytics.segment_chart_data(df, dim)
    pal = theme.palette()
    overall = analytics.overall_stats(df)["churn_rate"] * 100
    fig = go.Figure(go.Bar(x=table["Segment"].astype(str), y=table["ChurnRate"] * 100,
                           marker_color=[pal["churned"] if r * 100 > overall else pal["accent_2"] for r in table["ChurnRate"]],
                           customdata=table[["Customers", "Churned"]],
                           hovertemplate="%{x}<br>Churn rate %{y:.1f}%<br>%{customdata[1]:,} of %{customdata[0]:,} customers<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dash", line_color=pal["subtle"],
                  annotation_text=f"Overall {overall:.1f}%", annotation_font_color=pal["muted"])
    fig.update_layout(title_text=f"Churn rate by {analytics.DIMENSION_TITLES[dim].lower()}", yaxis_title="Churn rate (%)")
    with ui.card("ret-segment"):
        ui.plot(ui.style_fig(fig, 340, legend=False))
    if dim == "NumOfProducts":
        st.caption("Customers with 3 or 4 products are very few (326 in total) yet almost all of them left. Check why before using this as a rule.")


def distributions_tab(df) -> None:
    c1, c2 = st.columns(2)
    pal = theme.palette()
    for col, field, title in [(c1, "Age", "Age"), (c2, "CreditScore", "Credit score"),
                              (c1, "Balance", "Balance"), (c2, "EstimatedSalary", "Estimated salary")]:
        fig = go.Figure()
        for value, name, color in [(0, "Retained", pal["retained"]), (1, "Churned", pal["churned"])]:
            fig.add_histogram(x=df.loc[df.Exited == value, field], name=name, marker_color=color, opacity=0.7, nbinsx=30,
                              histnorm="percent", hovertemplate=f"{name}<br>{title} %{{x}}: %{{y:.1f}}% of group<extra></extra>")
        fig.update_layout(barmode="overlay", title_text=f"{title} distribution (share of each group)", yaxis_title="% of group")
        with col:
            with ui.card(f"ret-dist-{field}"):
                ui.plot(ui.style_fig(fig, 300))
    corr = df[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
               "EstimatedSalary", "Exited"]].corr()
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
                               colorscale=[[0, pal["accent_2"]], [0.5, pal["surface_2"]], [1, pal["churned"]]],
                               text=corr.round(2).values, texttemplate="%{text}", textfont=dict(color=pal["text"]), hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title_text="Correlation between numeric fields")
    fig.update_yaxes(autorange="reversed")
    with ui.card("ret-corr"):
        ui.plot(ui.style_fig(fig, 460, legend=False))


def plays_tab(rates) -> None:
    plays = analytics.retention_plays(rates)
    ui.callout("<b>These are hypotheses to test, not proven fixes.</b> The data shows where churn concentrates. "
               "Whether an action works needs a small pilot with a comparison group.", "warn")
    st.write("")
    ideas = {
        "NumOfProducts": "Single-product customers leave far more often than two-product customers. Test a second-product onboarding offer with a control group.",
        "IsActiveMember": "Inactive members churn at nearly twice the rate of active ones. Test re-engagement nudges after a quiet period.",
        "Age band": "Customers in their 40s and 50s churn most. Review whether products and service fit this life stage, and ask them directly.",
        "Geography": "Germany churns about twice as often as France or Spain. Look into local pricing, competitors and service quality before acting.",
        "Gender": "Women churn more than men here. Treat it as a prompt to check service differences, not a basis for targeting.",
        "Balance": "Customers holding a balance churn more than zero-balance ones. Review rates and offers for higher-balance accounts.",
    }
    if plays.empty:
        ui.empty_state("No clearly elevated segments", "The dataset shows no large group with a much higher churn rate.")
        return
    for row in plays.itertuples():
        st.markdown(
            f'<div class="card" style="margin-bottom:10px"><div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap">'
            f'{ui.badge(analytics.DIMENSION_TITLES[row.Dimension] + ": " + row.Segment, "high")}'
            f'<span class="model-line">{row.ChurnRate * 100:.1f}% churn across {row.Customers:,} customers</span></div>'
            f'<div style="font-size:.9rem;line-height:1.55">{ui.esc(ideas.get(row.Dimension, "Investigate why this group leaves more often."))}</div></div>',
            unsafe_allow_html=True)


def render() -> None:
    ui.page_header("Retention", "Retention Insights",
                   "Where churn concentrates in the customer base and what to test first.")
    df, rates = ui.get_dataset(), ui.get_rates()
    if df is None or rates is None:
        ui.empty_state("Dataset not available", "Place Churn_Modelling.csv in the data folder to see retention insights.")
        return
    t1, t2, t3 = st.tabs(["Where churn concentrates", "Distributions", "Suggested plays"])
    with t1:
        concentration_tab(df, rates)
    with t2:
        distributions_tab(df)
    with t3:
        plays_tab(rates)


render()
