# ============================================================
# Streamlit Dashboard — Decision Support Tool (v3, visually upgraded)
# ============================================================
# Upgrades over v2:
#   1. Always-visible portfolio KPI strip at the top
#   2. Stakeholder persona selector with tailored tip cards
#   3. SHAP "Why this price?" rendered as a Plotly horizontal bar chart
#   4. Tab 2: live adoption-rate slider + per-city uplift bar chart
#   5. Tab 3: per-model and per-city MAE bar charts (Plotly)
#   6. Cohesive coral/teal palette + cleaner typography hierarchy
# ============================================================
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Page configuration & theming (Airbnb-inspired palette)
# ============================================================
st.set_page_config(
    page_title="Airbnb Pricing Decision Tool",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color palette
CORAL    = "#FF5A5F"  # Airbnb red — primary accent
TEAL     = "#00A699"  # Airbnb teal — positive
DEEP     = "#484848"  # dark grey — text
SOFT     = "#767676"  # secondary text
WARN     = "#FC642D"  # Airbnb orange — overpriced / warning
SURFACE  = "#FFFFFF"
LIGHT    = "#F7F7F7"

# Pricing-status colors used everywhere
COLOR_UNDER = TEAL
COLOR_OVER  = CORAL
COLOR_FAIR  = "#5B8FF9"
STATUS_COLORS = {
    "Underpriced":  COLOR_UNDER,
    "Overpriced":   COLOR_OVER,
    "Fairly Priced": COLOR_FAIR,
}

# Streamlit theme overrides — works against the default light theme
st.markdown(f"""
<style>
    /* Page padding */
    .main .block-container {{
        padding-top: 1.2rem; padding-bottom: 2rem;
        max-width: 1400px;
    }}
    /* Heading typography */
    h1 {{ font-weight: 700 !important; letter-spacing: -0.02em; color: {DEEP} !important; }}
    h2 {{ font-weight: 600 !important; color: {DEEP} !important; }}
    h3 {{ font-weight: 600 !important; color: {DEEP} !important; }}

    /* KPI cards */
    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid #EBEBEB;
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    [data-testid="stMetric"] label {{
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: {SOFT} !important;
        font-weight: 600 !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: {DEEP} !important;
    }}

    /* Recommendation banner */
    .recommendation-banner {{
        padding: 22px 28px;
        border-radius: 14px;
        margin: 10px 0 18px 0;
        font-size: 1.05rem;
        line-height: 1.55;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    .banner-under  {{
        background: linear-gradient(135deg, #E8F8F5 0%, #D4F0EC 100%);
        border-left: 6px solid {COLOR_UNDER};
    }}
    .banner-over   {{
        background: linear-gradient(135deg, #FFEBEE 0%, #FFD8DC 100%);
        border-left: 6px solid {COLOR_OVER};
    }}
    .banner-fair   {{
        background: linear-gradient(135deg, #EBF1FE 0%, #DBE5FB 100%);
        border-left: 6px solid {COLOR_FAIR};
    }}
    .banner-blank  {{
        background: {LIGHT};
        border-left: 6px solid #BDBDBD;
    }}

    /* Persona tip cards */
    .persona-card {{
        padding: 14px 20px;
        border-radius: 10px;
        background: {LIGHT};
        border-left: 4px solid {CORAL};
        margin: 4px 0 18px 0;
        font-size: 0.95rem;
    }}
    .persona-card b {{ color: {CORAL}; }}

    /* Section labels above charts */
    .section-label {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {SOFT};
        font-weight: 600;
        margin: 18px 0 10px 0;
    }}

    /* Sidebar styling */
    [data-testid="stSidebar"] {{ background: {LIGHT}; }}
    [data-testid="stSidebar"] h2 {{ color: {CORAL} !important; }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        padding: 8px 18px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {LIGHT};
        color: {CORAL} !important;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Load model bundle
# ============================================================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "airbnb_price_model.pkl"

@st.cache_resource
def load_bundle():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

if not MODEL_PATH.exists():
    st.error("Model file not found. Please run the training notebook first.")
    st.stop()

bundle = load_bundle()
model = bundle["model"]
features = bundle["features"]
metrics = bundle["metrics"]
defaults = bundle["defaults"]
city_options_raw = bundle.get("city_options", [])
room_type_options_raw = bundle.get("room_type_options", [])
property_type_options_raw = bundle.get("property_type_options", [])
neighbourhood_by_city = bundle.get("neighbourhood_by_city", {})
location_reference = pd.DataFrame(bundle.get("location_reference", []))
amenity_features = bundle.get("amenity_features", [])
tag_labels = bundle.get("tag_labels", {})
categorical_features = bundle.get("categorical_features", [])
model_comparison = pd.DataFrame(bundle.get("model_comparison_test", []))
cv_results = pd.DataFrame(bundle.get("cv_results", []))
threshold_sensitivity = pd.DataFrame(bundle.get("threshold_sensitivity", []))
best_params = bundle.get("best_params", {})
naive_baseline_mae = bundle.get("naive_baseline_test_mae", None)
feature_names_out = bundle.get("preprocessor_feature_names_out", [])

# ============================================================
# Pre-compute portfolio statistics (used in top KPI strip)
# ============================================================
@st.cache_data
def load_portfolio_stats():
    """Load the test-set mispricing data once and return summary stats."""
    try:
        viz_df = pd.read_csv("mispricing_sample_output.csv")
        return viz_df
    except Exception:
        return pd.DataFrame()

viz_df_global = load_portfolio_stats()
if not viz_df_global.empty:
    n_total = len(viz_df_global)
    n_under = int((viz_df_global["pricing_status"] == "Underpriced").sum())
    n_over  = int((viz_df_global["pricing_status"] == "Overpriced").sum())
    underpriced_global = viz_df_global[viz_df_global["pricing_status"] == "Underpriced"].copy()
    underpriced_global["gap_per_night"] = (
        underpriced_global["predicted_price"] - underpriced_global["actual_price"]
    ).clip(lower=0)
    avg_gap = float(underpriced_global["gap_per_night"].mean()) if n_under > 0 else 0.0
    total_gbv_full_adoption = float(underpriced_global["gap_per_night"].sum() * 60)
else:
    n_total = n_under = n_over = 0
    avg_gap = 0.0
    total_gbv_full_adoption = 0.0


# ============================================================
# Helpers
# ============================================================
def with_all(options):
    options = [str(x) for x in options if pd.notna(x)]
    options = sorted(list(dict.fromkeys(options)))
    return ["All"] + options

def model_value(value, fallback="Unknown"):
    return fallback if value == "All" else value

def get_representative_coordinates(city, neighbourhood):
    default_lat = float(defaults.get("latitude", 40.7128))
    default_lon = float(defaults.get("longitude", -74.0060))
    if location_reference.empty:
        return default_lat, default_lon
    filtered = location_reference.copy()
    if city != "All":
        filtered = filtered[filtered["city"] == city]
    if neighbourhood != "All":
        filtered = filtered[filtered["neighbourhood"] == neighbourhood]
    if filtered.empty and city != "All":
        filtered = location_reference[location_reference["city"] == city]
    if filtered.empty:
        filtered = location_reference
    return float(filtered["latitude"].mean()), float(filtered["longitude"].mean())

def build_input_dataframe(input_row, features):
    X_new = pd.DataFrame([input_row])
    for col in features:
        if col not in X_new.columns:
            X_new[col] = np.nan
    return X_new[features]

@st.cache_resource
def build_shap_explainer(_model):
    try:
        import shap
        inner = _model.regressor_.named_steps["model"]
        return shap.TreeExplainer(inner)
    except Exception:
        return None

shap_explainer = build_shap_explainer(model)


# ============================================================
# SIDEBAR — persona selector + listing inputs
# ============================================================
PERSONAS = {
    "Host-Success Rep": {
        "icon": "📞",
        "tip": "<b>How to use this:</b> Use Tab 1 to get a defensible per-listing recommendation, then Tab 2 to pull your daily outreach queue. The SHAP bar chart gives you 3 ready-made talking points for every host call.",
    },
    "Marketplace Strategy": {
        "icon": "📈",
        "tip": "<b>How to use this:</b> The KPI strip above shows the addressable opportunity. Tab 2's adoption-rate slider lets you pressure-test commission projections. Tab 3 surfaces where the model is most/least reliable for segment-specific outreach plans.",
    },
    "Pricing Team": {
        "icon": "🔬",
        "tip": "<b>How to use this:</b> Tab 3 is your home — model comparison, per-city MAE, CV stability, and threshold sensitivity. The non-obvious finding is that <b>minimum_nights</b> is the second-most-important pricing feature (permutation importance $10.42 MAE).",
    },
}

with st.sidebar:
    st.header("👥 Persona")
    persona = st.radio(
        "Viewing as:",
        list(PERSONAS.keys()),
        index=0,
        help="Persona changes the tip card on each tab. All views remain accessible."
    )

    st.divider()
    st.header("📝 Listing Inputs")

    # Essential filters
    city_options = with_all(city_options_raw)
    default_city_index = city_options.index("New York") if "New York" in city_options else 0
    city = st.selectbox("City / Market", city_options, index=default_city_index)

    if city != "All":
        neighbourhood_base = neighbourhood_by_city.get(city, [])
    else:
        neighbourhood_base = []
        for values in neighbourhood_by_city.values():
            neighbourhood_base.extend(values)
    neighbourhood_options = with_all(neighbourhood_base if neighbourhood_base else ["Unknown"])
    neighbourhood = st.selectbox("Neighbourhood", neighbourhood_options)

    room_type = st.selectbox(
        "Room Type",
        with_all(room_type_options_raw),
        index=1 if len(room_type_options_raw) > 0 else 0
    )

    accommodates = st.number_input(
        "Accommodates", min_value=1, max_value=20,
        value=int(defaults.get("accommodates", 2))
    )

    actual_price = st.number_input(
        "Current Listed Price ($/night, optional)",
        min_value=0.0, value=0.0, step=5.0,
        help="Enter the host's current listed price to classify as Under/Fair/Overpriced."
    )

    with st.expander("🏠 Listing details"):
        property_type = st.selectbox(
            "Property Type",
            with_all(property_type_options_raw),
            index=1 if len(property_type_options_raw) > 0 else 0
        )
        bedrooms = st.number_input("Bedrooms", min_value=0.0, max_value=10.0,
                                   value=float(defaults.get("bedrooms", 1)), step=0.5)
        beds = st.number_input("Beds", min_value=0.0, max_value=20.0,
                               value=float(defaults.get("beds", 1)), step=0.5)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0,
                                    value=float(defaults.get("bathrooms", 1)), step=0.5)
        minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365,
                                          value=int(defaults.get("minimum_nights", 2)))
        availability_365 = st.slider("Availability (next 365 days)",
                                      min_value=0, max_value=365,
                                      value=int(defaults.get("availability_365", 180)))

    with st.expander("⭐ Reviews & host"):
        number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=2000,
                                             value=int(defaults.get("number_of_reviews", 10)))
        reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, max_value=50.0,
                                             value=float(defaults.get("reviews_per_month", 0.5)),
                                             step=0.1)
        calculated_host_listings_count = st.number_input(
            "Host Listing Count", min_value=1, max_value=500,
            value=int(defaults.get("calculated_host_listings_count", 1))
        )
        review_scores_rating = st.slider("Review Rating", min_value=0.0, max_value=100.0,
                                          value=float(defaults.get("review_scores_rating", 90)),
                                          step=1.0)
        instant_bookable = st.selectbox("Instant Bookable", ["All", "t", "f", "Unknown"])
        host_is_superhost = st.selectbox("Superhost", ["All", "t", "f", "Unknown"])

    with st.expander("🛋️ Amenities"):
        selected_amenities = {}
        for tag in amenity_features:
            label = tag_labels.get(tag, tag.replace("_", " ").title())
            default_value = True if tag in ["wifi", "kitchen"] else False
            selected_amenities[tag] = st.checkbox(label, value=default_value, key=f"amen_{tag}")

    st.divider()
    st.markdown("**Decision thresholds**")
    mispricing_threshold = st.slider(
        "Mispricing threshold (±%)", min_value=5, max_value=30, value=15, step=5,
        help="10% suits high-confidence segments (Boston/Seattle); 20–25% suits high-error segments (Austin/SF). See report §4.5."
    ) / 100
    adoption_rate = st.slider(
        "Host adoption rate (%)", min_value=5, max_value=75, value=25, step=5,
        help="Smart Pricing's reported adoption is in the low-teens; 25% is our central case for an explanation-driven recommendation system."
    ) / 100


# ============================================================
# Compute prediction (must run before tabs render)
# ============================================================
selected_lat, selected_lon = get_representative_coordinates(city, neighbourhood)

input_row = {
    "city": model_value(city),
    "neighbourhood": model_value(neighbourhood),
    "room_type": model_value(room_type),
    "property_type": model_value(property_type),
    "instant_bookable": model_value(instant_bookable, "Unknown"),
    "host_is_superhost": model_value(host_is_superhost, "Unknown"),
    "latitude": selected_lat,
    "longitude": selected_lon,
    "accommodates": accommodates,
    "bedrooms": bedrooms,
    "beds": beds,
    "bathrooms": bathrooms,
    "minimum_nights": minimum_nights,
    "number_of_reviews": number_of_reviews,
    "reviews_per_month": reviews_per_month,
    "review_activity": reviews_per_month,
    "calculated_host_listings_count": calculated_host_listings_count,
    "availability_365": availability_365,
    "availability_ratio": availability_365 / 365,
    "review_scores_rating": review_scores_rating,
    "data_year": int(defaults.get("data_year", 2023)),
}
for tag in amenity_features:
    input_row[tag] = int(selected_amenities.get(tag, False))

input_row["amenity_score"] = (
    int(selected_amenities.get("wifi", False)) * 1
    + int(selected_amenities.get("kitchen", False)) * 1
    + int(selected_amenities.get("air_conditioning", False)) * 1
    + int(selected_amenities.get("parking", False)) * 1
    + int(selected_amenities.get("pool", False)) * 2
    + int(selected_amenities.get("gym", False)) * 1.5
    + int(selected_amenities.get("pets_allowed", False)) * 1
)
input_row["has_luxury_amenities"] = int(
    selected_amenities.get("pool", False)
    or selected_amenities.get("gym", False)
    or selected_amenities.get("parking", False)
)

X_new = build_input_dataframe(input_row, features)
predicted_price = float(np.maximum(model.predict(X_new)[0], 1))
mae = float(metrics.get("MAE", 0))
low_price = max(0, predicted_price - mae)
high_price = predicted_price + mae


# ============================================================
# HEADER + PORTFOLIO KPI STRIP (always visible)
# ============================================================
title_col, persona_col = st.columns([3, 1])
with title_col:
    st.title("🏠 Airbnb Pricing Decision Tool")
    st.caption(
        "Decision-support for the host-success team • "
        "recommends a market-consistent nightly price • "
        "explains the recommendation in plain dollars."
    )
with persona_col:
    st.markdown(
        f'<div style="text-align:right; padding-top: 22px; color:{SOFT};">'
        f'<span style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;">Viewing as</span><br>'
        f'<span style="font-size:1.05rem; color:{DEEP}; font-weight:600;">'
        f'{PERSONAS[persona]["icon"]}  {persona}</span></div>',
        unsafe_allow_html=True,
    )

# Persona tip card
st.markdown(
    f'<div class="persona-card">{PERSONAS[persona]["tip"]}</div>',
    unsafe_allow_html=True,
)

# Portfolio KPI strip
st.markdown('<div class="section-label">Portfolio snapshot — test set</div>', unsafe_allow_html=True)
kp1, kp2, kp3, kp4 = st.columns(4)
kp1.metric(
    "Underpriced listings",
    f"{n_under:,}",
    delta=f"{n_under / max(n_total,1):.1%} of test set",
    delta_color="off",
    help=f"At the current {int(mispricing_threshold*100)}% threshold."
)
gbv_at_adoption = total_gbv_full_adoption * adoption_rate
kp2.metric(
    f"Annual GBV opportunity ({int(adoption_rate*100)}% adoption)",
    f"${gbv_at_adoption/1e6:,.2f}M" if gbv_at_adoption >= 1e6 else f"${gbv_at_adoption:,.0f}",
    help="Adjust the adoption-rate slider in the sidebar to pressure-test."
)
kp3.metric(
    "Avg gap / night (underpriced)",
    f"${avg_gap:,.0f}",
    help=f"Test MAE is ${mae:,.0f}, so a {int(mispricing_threshold*100)}%+ gap is signal, not noise."
)
kp4.metric(
    "Best model MAE",
    f"${mae:,.0f}",
    delta=f"−${(naive_baseline_mae - mae):,.0f} vs baseline" if naive_baseline_mae else None,
    help=f"Best model: {metrics.get('Best Model', 'N/A')}."
)

st.divider()


# ============================================================
# TABS
# ============================================================
tab_rec, tab_top, tab_model, tab_method = st.tabs([
    "🎯 Listing Recommendation",
    "📋 Top Underpriced (Pilot Targets)",
    "📊 Model Performance",
    "📚 Methodology & Limitations",
])


# ============================================================
# TAB 1 — Listing Recommendation
# ============================================================
with tab_rec:
    # Recommendation banner
    if actual_price > 0:
        mispricing_pct = (actual_price - predicted_price) / predicted_price
        if mispricing_pct <= -mispricing_threshold:
            uplift = predicted_price - actual_price
            banner_class = "banner-under"
            banner_text = (
                f"<b>📈 UNDERPRICED</b> &nbsp;—&nbsp; listed at ${actual_price:,.0f}, "
                f"recommended ${predicted_price:,.0f} "
                f"(<b>+${uplift:,.0f}/night opportunity</b>, {abs(mispricing_pct):.0%} below market)."
            )
        elif mispricing_pct >= mispricing_threshold:
            adjust = actual_price - predicted_price
            banner_class = "banner-over"
            banner_text = (
                f"<b>📉 OVERPRICED</b> &nbsp;—&nbsp; listed at ${actual_price:,.0f}, "
                f"recommended ${predicted_price:,.0f} "
                f"(<b>−${adjust:,.0f}/night to be competitive</b>, {mispricing_pct:.0%} above market)."
            )
        else:
            banner_class = "banner-fair"
            banner_text = (
                f"<b>✅ FAIRLY PRICED</b> &nbsp;—&nbsp; listed at ${actual_price:,.0f} "
                f"is within ±{int(mispricing_threshold*100)}% of recommended ${predicted_price:,.0f}. "
                "No action needed."
            )
        mispricing_pct_val = mispricing_pct
    else:
        banner_class = "banner-blank"
        banner_text = (
            f"<b>Recommended market price: ${predicted_price:,.0f}/night.</b> "
            "Enter the host's current listed price in the sidebar to classify."
        )
        mispricing_pct_val = None

    st.markdown(
        f'<div class="recommendation-banner {banner_class}">{banner_text}</div>',
        unsafe_allow_html=True,
    )

    # KPI row
    k1, k2, k3 = st.columns(3)
    k1.metric("Recommended Price", f"${predicted_price:,.0f} / night")
    k2.metric("Confidence Range", f"${low_price:,.0f} – ${high_price:,.0f}",
              help=f"±${mae:,.0f} (model's test-set MAE).")
    if actual_price > 0:
        delta_label = f"${actual_price - predicted_price:+,.0f} vs. market"
        k3.metric("Listed Price", f"${actual_price:,.0f} / night", delta=delta_label,
                  delta_color=("inverse" if mispricing_pct_val and mispricing_pct_val > 0 else "normal"))
    else:
        k3.metric("Listed Price", "—", help="Enter in sidebar to compare.")

    st.divider()

    # Two-column body — UPGRADED: SHAP as Plotly horizontal bar chart
    col_why, col_seg = st.columns([1.3, 1])

    with col_why:
        st.subheader("🔎 Why this price?")
        st.caption("Top contributors to *this listing's* recommendation, in dollar terms. Use these as host-call talking points.")

        contribs = []
        if shap_explainer is not None:
            try:
                preproc = model.regressor_.named_steps["preprocessor"]
                X_new_transformed = preproc.transform(X_new)
                if hasattr(X_new_transformed, "toarray"):
                    X_new_dense = X_new_transformed.toarray()
                else:
                    X_new_dense = X_new_transformed
                shap_vals = shap_explainer.shap_values(X_new_dense)[0]

                # --- Aggregate one-hot dummies back to their parent categorical ---
                # e.g. "cat__neighbourhood_Tribeca" -> parent="neighbourhood"
                # so all neighbourhood_* SHAP values sum into a single contribution
                # labeled with the listing's ACTUAL neighbourhood (e.g. "Tompkinsville").
                # Features that should be rolled up into a single "Location" bucket
                # rather than appearing as separate contributors on the host-call chart.
                LOCATION_FEATURES = {"latitude", "longitude", "neighbourhood", "city"}

                def _parent_feature(encoded_name):
                    stripped = encoded_name.replace("num__", "").replace("cat__", "")
                    # First: match against known categorical column names (longest-prefix wins)
                    for cat in sorted(categorical_features, key=len, reverse=True):
                        if stripped == cat or stripped.startswith(cat + "_"):
                            return "location" if cat in LOCATION_FEATURES else cat
                    # Second: numeric features — also fold lat/long into the location bucket
                    if stripped in LOCATION_FEATURES:
                        return "location"
                    return stripped

                shap_by_parent = {}
                for name, val in zip(feature_names_out, shap_vals):
                    parent = _parent_feature(name)
                    shap_by_parent[parent] = shap_by_parent.get(parent, 0.0) + float(val)

                # Label categoricals with the listing's actual selected value
                listing_values = {
                    "location": f"{neighbourhood}, {city}" if city != "All" else neighbourhood,
                    "room_type": room_type,
                    "property_type": property_type,
                }

                def _display_label(parent):
                    if parent in listing_values and listing_values[parent] not in (None, "", "All"):
                        if parent == "location":
                            return f"Location: {listing_values[parent]}"
                        pretty = parent.replace("_", " ").title()
                        return f"{pretty}: {listing_values[parent]}"
                    return parent.replace("_", " ").title()
                contribs = sorted(
                    [(_display_label(p), v) for p, v in shap_by_parent.items()],
                    key=lambda x: abs(x[1]), reverse=True
                )[:7]
            except Exception:
                contribs = []

        if contribs:
            shap_df = []
            for name, val in contribs:
                multiplicative = float(np.exp(val))
                dollar_effect = predicted_price * (1 - 1 / multiplicative)
                clean_name = (name.replace("num__", "").replace("cat__", "")
                              .replace("_", " ").title())
                clean_name = clean_name[:48]
                shap_df.append({
                    "feature": clean_name,
                    "dollar_effect": dollar_effect,
                    "direction": "Increases price" if val > 0 else "Decreases price",
                })
            shap_df = pd.DataFrame(shap_df)
            shap_df = shap_df.sort_values("dollar_effect")

            fig_shap = px.bar(
                shap_df, x="dollar_effect", y="feature",
                orientation="h", color="direction",
                color_discrete_map={
                    "Increases price": COLOR_UNDER,
                    "Decreases price": COLOR_OVER,
                },
                text=shap_df["dollar_effect"].apply(lambda x: f"${x:+,.0f}"),
            )
            fig_shap.update_traces(textposition="outside", textfont_size=12)
            fig_shap.update_layout(
                height=320,
                margin=dict(l=10, r=80, t=10, b=10),
                yaxis_title=None, xaxis_title="Effect on price ($/night)",
                showlegend=True,
                legend=dict(orientation="h", y=-0.18, x=0, title=None),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=DEEP, size=12),
            )
            fig_shap.update_xaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=True, zerolinecolor="#999")
            fig_shap.update_yaxes(showgrid=False)
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("SHAP explanations unavailable. Top drivers (heuristic):")
            fallback = []
            if city != "All":      fallback.append(f"• Location: {city}")
            if room_type != "All": fallback.append(f"• Room type: {room_type}")
            if accommodates >= 4:  fallback.append(f"• Larger capacity ({accommodates} guests)")
            sel_amen = [tag_labels.get(t, t)
                        for t in amenity_features if selected_amenities.get(t, False)]
            if sel_amen: fallback.append(f"• Amenities: {', '.join(sel_amen[:4])}")
            for line in fallback: st.write(line)

    with col_seg:
        st.subheader("🏘 Where this falls vs. comparable listings")
        try:
            similar_df = viz_df_global.copy()
            if city != "All" and "city" in similar_df.columns:
                similar_df = similar_df[similar_df["city"] == city]
            if room_type != "All" and "room_type" in similar_df.columns:
                similar_df = similar_df[similar_df["room_type"] == room_type]

            if len(similar_df) >= 5:
                # Mini histogram of comparable listings with our prediction marked
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=similar_df["actual_price"],
                    nbinsx=25,
                    marker_color=COLOR_FAIR,
                    opacity=0.65,
                    name="Comparable listings",
                ))
                fig_hist.add_vline(
                    x=predicted_price,
                    line_color=CORAL, line_width=3,
                    annotation_text=f"Recommended ${predicted_price:,.0f}",
                    annotation_position="top",
                    annotation_font_color=CORAL,
                )
                if actual_price > 0:
                    fig_hist.add_vline(
                        x=actual_price,
                        line_color=DEEP, line_width=2, line_dash="dash",
                        annotation_text=f"Listed ${actual_price:,.0f}",
                        annotation_position="bottom",
                    )
                fig_hist.update_layout(
                    height=320,
                    margin=dict(l=10, r=10, t=30, b=10),
                    showlegend=False,
                    xaxis_title="Listed price ($/night)",
                    yaxis_title="Comparable listings",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=DEEP, size=12),
                )
                fig_hist.update_xaxes(showgrid=True, gridcolor="#EEEEEE")
                fig_hist.update_yaxes(showgrid=True, gridcolor="#EEEEEE")
                st.plotly_chart(fig_hist, use_container_width=True)

                ks1, ks2 = st.columns(2)
                ks1.metric("Comparable listings", f"{len(similar_df):,}")
                ks2.metric("Their median price", f"${similar_df['actual_price'].median():,.0f}")
            else:
                st.info("Not enough comparable listings in the test sample for this filter set.")
        except Exception as e:
            st.info(f"Comparable-listings data unavailable: {e}")

    st.divider()
    st.markdown('<div class="section-label">Location context</div>', unsafe_allow_html=True)
    map_df = pd.DataFrame({"lat": [selected_lat], "lon": [selected_lon]})
    st.map(map_df, zoom=11, use_container_width=True)


# ============================================================
# TAB 2 — Top Underpriced (operational deliverable)
# ============================================================
with tab_top:
    st.subheader("📋 Top Underpriced Listings — Pilot Target List")
    st.caption(
        f"Sorted by estimated annual GBV uplift at 60 booked nights/year. "
        f"Adjust the adoption rate slider in the sidebar (currently {int(adoption_rate*100)}%) to update commission projections."
    )

    if not viz_df_global.empty:
        underpriced = viz_df_global[viz_df_global["pricing_status"] == "Underpriced"].copy()
        underpriced["price_gap_per_night"] = (
            underpriced["predicted_price"] - underpriced["actual_price"]
        ).clip(lower=0)
        underpriced["estimated_annual_uplift"] = underpriced["price_gap_per_night"] * 60
        underpriced["adjusted_uplift"] = underpriced["estimated_annual_uplift"] * adoption_rate
        underpriced["commission_uplift"] = underpriced["adjusted_uplift"] * 0.14

        if city != "All" and "city" in underpriced.columns:
            underpriced_view = underpriced[underpriced["city"] == city].copy()
        else:
            underpriced_view = underpriced.copy()

        # Live business impact KPIs (driven by the adoption slider)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Listings flagged", f"{len(underpriced_view):,}")
        c2.metric("Avg $/night gap", f"${underpriced_view['price_gap_per_night'].mean():,.0f}")
        gbv_total = underpriced_view['adjusted_uplift'].sum()
        c3.metric(
            f"Annual GBV uplift @ {int(adoption_rate*100)}%",
            f"${gbv_total/1e6:,.2f}M" if gbv_total >= 1e6 else f"${gbv_total:,.0f}",
        )
        c4.metric(
            f"Commission uplift @ {int(adoption_rate*100)}%",
            f"${underpriced_view['commission_uplift'].sum():,.0f}",
            help="GBV uplift × 14% Airbnb commission rate."
        )

        st.divider()

        # Per-city uplift bar chart (UPGRADE)
        if "city" in underpriced.columns:
            st.markdown('<div class="section-label">Annual commission uplift by market</div>', unsafe_allow_html=True)
            city_impact = (
                underpriced.groupby("city")
                .agg(
                    listings=("price_gap_per_night", "size"),
                    avg_gap=("price_gap_per_night", "mean"),
                    commission=("commission_uplift", "sum"),
                )
                .reset_index()
                .sort_values("commission", ascending=True)
            )
            fig_city = px.bar(
                city_impact, x="commission", y="city", orientation="h",
                color="commission", color_continuous_scale=["#FFE5E7", CORAL],
                text=city_impact["commission"].apply(lambda x: f"${x/1e3:,.0f}K"),
                hover_data={"listings": True, "avg_gap": ":.0f", "commission": ":,.0f"},
            )
            fig_city.update_traces(textposition="outside")
            fig_city.update_layout(
                height=320,
                margin=dict(l=10, r=80, t=10, b=10),
                xaxis_title=f"Commission uplift @ {int(adoption_rate*100)}% adoption",
                yaxis_title=None,
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=DEEP, size=12),
            )
            fig_city.update_xaxes(showgrid=True, gridcolor="#EEEEEE")
            st.plotly_chart(fig_city, use_container_width=True)

        st.divider()

        # Top-50 table for the host-success team
        st.markdown('<div class="section-label">Top 50 outreach targets (current filter)</div>', unsafe_allow_html=True)
        underpriced_sorted = underpriced_view.sort_values("estimated_annual_uplift", ascending=False).head(50)
        st.dataframe(
            underpriced_sorted[
                ["city", "neighbourhood", "room_type", "property_type",
                 "actual_price", "predicted_price",
                 "price_gap_per_night", "estimated_annual_uplift",
                 "adjusted_uplift", "commission_uplift"]
            ].round(0).rename(columns={
                "actual_price": "Listed",
                "predicted_price": "Recommended",
                "price_gap_per_night": "Gap/night",
                "estimated_annual_uplift": "GBV uplift (full)",
                "adjusted_uplift": f"GBV uplift (@{int(adoption_rate*100)}%)",
                "commission_uplift": f"Commission (@{int(adoption_rate*100)}%)",
            }),
            use_container_width=True,
            height=400,
        )
        st.download_button(
            "⬇ Export top-50 list (CSV)",
            data=underpriced_sorted.to_csv(index=False),
            file_name="top_50_underpriced_listings.csv",
            mime="text/csv",
        )
    else:
        st.warning("Top-listings data unavailable. Re-run the notebook to generate mispricing_sample_output.csv.")


# ============================================================
# TAB 3 — Model Performance
# ============================================================
with tab_model:
    st.subheader("Model comparison (test set)")

    # Per-model MAE bar chart (UPGRADE)
    if not model_comparison.empty and "MAE" in model_comparison.columns and "Model" in model_comparison.columns:
        cmp = model_comparison.copy().sort_values("MAE", ascending=True)
        fig_mae = px.bar(
            cmp, x="MAE", y="Model", orientation="h",
            color="MAE", color_continuous_scale=[TEAL, "#F4A261", CORAL],
            text=cmp["MAE"].apply(lambda x: f"${x:,.2f}"),
        )
        fig_mae.update_traces(textposition="outside")
        fig_mae.update_layout(
            height=320, margin=dict(l=10, r=80, t=10, b=10),
            xaxis_title="Test MAE ($/night) — lower is better",
            yaxis_title=None, coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=DEEP, size=12),
        )
        fig_mae.update_xaxes(showgrid=True, gridcolor="#EEEEEE")
        st.plotly_chart(fig_mae, use_container_width=True)

        with st.expander("📋 Full metrics table"):
            display_cmp = model_comparison.copy()
            for col in ["MAE", "RMSE", "R2_log", "R2_price", "Adj_R2", "MAPE (%)"]:
                if col in display_cmp.columns:
                    display_cmp[col] = display_cmp[col].round(3)
            st.dataframe(display_cmp, use_container_width=True)

    if best_params:
        with st.expander("⚙ Selected model hyperparameters"):
            st.json(best_params)

    st.divider()

    # Per-city MAE bar chart (NEW UPGRADE)
    st.subheader("Where the model is most/least reliable")
    st.caption(
        "Operationally: Boston/Seattle warrant tighter 10% mispricing thresholds; "
        "Austin/SF warrant looser 20–25%. Tune via the sidebar slider."
    )
    if not viz_df_global.empty:
        city_err = viz_df_global.copy()
        city_err["abs_error"] = (city_err["actual_price"] - city_err["predicted_price"]).abs()
        city_err_agg = city_err.groupby("city").agg(
            mae=("abs_error", "mean"), n=("abs_error", "size")
        ).reset_index().sort_values("mae", ascending=True)

        fig_city_mae = px.bar(
            city_err_agg, x="city", y="mae",
            color="mae", color_continuous_scale=[TEAL, CORAL],
            text=city_err_agg["mae"].apply(lambda x: f"${x:,.0f}"),
            hover_data={"n": True, "mae": ":.2f"},
        )
        fig_city_mae.update_traces(textposition="outside")
        fig_city_mae.update_layout(
            height=340, margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="Test MAE ($/night)", xaxis_title=None,
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=DEEP, size=12),
        )
        fig_city_mae.update_yaxes(showgrid=True, gridcolor="#EEEEEE")
        st.plotly_chart(fig_city_mae, use_container_width=True)

    st.divider()

    # CV stability
    st.subheader("Cross-validation stability (3-fold)")
    if not cv_results.empty:
        cv_display = cv_results.round(3)
        st.dataframe(cv_display, use_container_width=True)
        st.caption(
            "Low CV_MAE_Std relative to CV_MAE_Mean means the model is stable "
            "across data splits and not overfit to a particular subset."
        )

    st.divider()

    # Threshold sensitivity
    st.subheader("Threshold sensitivity (defending the 15% choice)")
    if not threshold_sensitivity.empty:
        st.dataframe(threshold_sensitivity, use_container_width=True)
        st.caption(
            "At 5–10% the average gap is small enough to be confused with model noise. "
            "At 20%+ we miss too many actionable cases. 15% balances coverage and signal strength."
        )

    st.divider()

    # Actual vs Predicted scatter
    st.subheader("Actual vs Predicted (test set)")
    if not viz_df_global.empty:
        fig_scatter = px.scatter(
            viz_df_global.sample(min(3000, len(viz_df_global)), random_state=42),
            x="actual_price", y="predicted_price",
            color="pricing_status",
            color_discrete_map=STATUS_COLORS,
            hover_data=["city", "neighbourhood", "room_type", "property_type"],
            opacity=0.55,
        )
        max_v = float(viz_df_global["actual_price"].max())
        fig_scatter.add_shape(type="line", line=dict(dash="dash", color=DEEP),
                              x0=0, y0=0, x1=max_v, y1=max_v)
        fig_scatter.update_layout(
            height=480,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=DEEP, size=12),
            legend=dict(orientation="h", y=-0.15, x=0),
        )
        fig_scatter.update_xaxes(showgrid=True, gridcolor="#EEEEEE")
        fig_scatter.update_yaxes(showgrid=True, gridcolor="#EEEEEE")
        st.plotly_chart(fig_scatter, use_container_width=True)


# ============================================================
# TAB 4 — Methodology & Limitations
# ============================================================
with tab_method:
    st.subheader("How this tool works")
    st.markdown("""
We trained a regression model on ~25,000 cleaned Airbnb listings spanning NYC, Boston,
Seattle, and broader US markets across 2019, 2020, and 2023. The model takes ~25 features
per listing — location (city, neighbourhood, lat/long), room/property type, capacity,
review activity, and amenity tags — and outputs a recommended nightly price.

**Selection process.** Per the M3 module's regression algorithm ladder, we benchmarked a
naive segment-median baseline, Linear Regression (M3 baseline), Random Forest (bagging),
Gradient Boosting (boosting), and XGBoost (M3-recommended boosting). The strongest family
was hyperparameter-tuned with `GridSearchCV` on the validation set. Final test-set metrics
were computed once on a held-out test partition the model never saw during selection.

**Why MAE primary.** Per M3's context-driven metric selection: dollar-per-night error is
what a host-success rep defends on a phone call. Squared metrics overweight luxury outliers
we do not target with this recommendation system. RMSE / MAPE / R² / Adj R² are
reported as secondary metrics for completeness.
    """)

    st.subheader("⚠ Known limitations")
    st.markdown(r"""
- **Listed price ≠ realized booking revenue — the largest single limitation.**
  We model what hosts list, not what they earn. Cleaning fees, surcharges, and length-of-stay
  discounts are absent. Recommendations are advisory only until booking-level data is available.
- **Non-stationary training data with survivorship bias.**
  2020 reflects pandemic-specific economics; public scrapes capture only listings active at
  scrape time. Reported \$58.15 test MAE is likely optimistic against 2026 deployment.
- **The "\$179K commission upside" is a midpoint.**
  Adoption rate is the most consequential and least empirically supported assumption.
  Realistic range is approximately \$50K–\$360K depending on adoption (10–50%) and booked nights.
- **Methodological caveats.**
  Validation is in-distribution random split, not temporal; Adj R² of 0.324 is honest
  about the ~67% unexplained variance from listing-quality factors not in our data.
  SHAP attributions are model reasoning, not causal effects.
- **Fairness and disparate impact.**
  Pricing models trained on neighbourhood data can encode patterns correlated with protected
  characteristics. The pilot must not proceed in any market without a prior fairness audit.
- **"Do no harm" guardrail.**
  Aggregate uplift can mask individual harm — before scaling, also report the fraction
  of treatment-arm hosts experiencing year-over-year revenue declines vs. control.
    """)

    st.subheader("📊 Operational guidance")
    st.markdown("""
- **Use:** prioritize host-success outreach to underpriced listings flagged in Tab 2,
  with the SHAP-derived reasons from Tab 1 as talking points.
- **Don't:** use this for automatic price changes without host approval. This is a
  recommendation tool, not an automated pricing engine.
- **Cadence:** refresh the model quarterly; monitor rolling 30-day MAE and retrain if it
  degrades by >15%.
    """)

st.divider()
st.caption(
    f"Best model: {metrics.get('Best Model', 'N/A')} • "
    f"MAE: ${metrics.get('MAE', 0):.0f} • "
    f"Adj R²: {metrics.get('Adj_R2', metrics.get('R2_price', 0)):.3f} • "
    f"n_test = {metrics.get('n_test', '—')}"
)
