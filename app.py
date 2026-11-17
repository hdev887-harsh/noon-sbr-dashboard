"""
Noon E-Commerce Analytics Dashboard — SBR (Data Analytics)
Group: Harsh, Hunar, Saachi (CEO), Maansi, Keshav, Devansh

NOTE: The dataset is SYNTHETIC (generated with a fixed seed) for academic
illustration. It is modelled on Noon's UAE/KSA/Egypt marketplace but does not
contain real Noon data.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Noon Analytics Dashboard", page_icon="🛒", layout="wide")

NOON_YELLOW = "#FEEE00"
INK = "#1F1F1F"


# --------------------------------------------------------------------------
# 1. DATA
# --------------------------------------------------------------------------
@st.cache_data
def make_data(n_orders: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    countries = ["UAE", "KSA", "Egypt"]
    cities = {
        "UAE": ["Dubai", "Abu Dhabi", "Sharjah"],
        "KSA": ["Riyadh", "Jeddah", "Dammam"],
        "Egypt": ["Cairo", "Alexandria", "Giza"],
    }
    categories = {
        "Electronics": (950, 0.09),
        "Fashion": (180, 0.22),
        "Beauty": (120, 0.08),
        "Home & Kitchen": (260, 0.10),
        "Grocery (Minutes)": (85, 0.02),
        "Toys": (140, 0.07),
    }
    channels = ["App", "Web"]
    delivery_types = ["noon Minutes (15-min)", "Express (next-day)", "Standard"]
    payments = ["Card", "Cash on Delivery", "Apple Pay", "BNPL (Tabby/Tamara)"]

    n_customers = 3500
    cust_ids = np.arange(1, n_customers + 1)
    cust_country = rng.choice(countries, n_customers, p=[0.5, 0.35, 0.15])
    cust_age = rng.choice(["18-24", "25-34", "35-44", "45+"], n_customers, p=[0.24, 0.40, 0.24, 0.12])
    cust_prime = rng.random(n_customers) < 0.28  # noon One members

    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    # seasonality: Ramadan (Mar), White/Yellow Friday (late Nov), DSF (Dec)
    w = np.ones(len(dates))
    w[(dates.month == 3)] *= 1.5
    w[(dates.month == 11) & (dates.day >= 20)] *= 3.0
    w[(dates.month == 12)] *= 1.4
    w /= w.sum()

    cust = rng.choice(cust_ids, n_orders, p=None)
    idx = cust - 1
    order_date = rng.choice(dates, n_orders, p=w)
    cat_names = list(categories)
    cat = rng.choice(cat_names, n_orders, p=[0.2, 0.22, 0.14, 0.14, 0.2, 0.1])
    base_price = np.array([categories[c][0] for c in cat])
    base_return = np.array([categories[c][1] for c in cat])

    sale_event = ((pd.DatetimeIndex(order_date).month == 11) & (pd.DatetimeIndex(order_date).day >= 20))
    discount = np.clip(rng.normal(0.12, 0.06, n_orders) + sale_event * 0.18, 0, 0.7)
    qty = rng.integers(1, 4, n_orders)
    gmv = np.round(base_price * rng.lognormal(0, 0.35, n_orders) * qty, 2)
    net = np.round(gmv * (1 - discount), 2)

    dtype = np.where(cat == "Grocery (Minutes)", delivery_types[0],
                     rng.choice(delivery_types[1:], n_orders, p=[0.45, 0.55]))
    promised = np.select([dtype == delivery_types[0], dtype == delivery_types[1]], [0.25, 1], 3)
    delay = rng.exponential(0.3, n_orders) * np.where(cust_country[idx] == "Egypt", 1.8, 1.0)
    actual = promised + delay * np.where(dtype == delivery_types[0], 0.15, 1)
    late = actual > promised + np.where(dtype == delivery_types[0], 0.1, 0.5)

    ret_prob = base_return + late * 0.06 + (discount > 0.35) * 0.04
    returned = rng.random(n_orders) < ret_prob
    rating = np.clip(np.round(rng.normal(4.3, 0.6, n_orders) - late * 0.9 - returned * 0.8), 1, 5)

    df = pd.DataFrame({
        "order_id": np.arange(100001, 100001 + n_orders),
        "order_date": pd.to_datetime(order_date),
        "customer_id": cust,
        "country": cust_country[idx],
        "city": [rng.choice(cities[c]) for c in cust_country[idx]],
        "age_group": cust_age[idx],
        "noon_one_member": cust_prime[idx],
        "channel": rng.choice(channels, n_orders, p=[0.78, 0.22]),
        "payment_method": rng.choice(payments, n_orders, p=[0.42, 0.28, 0.18, 0.12]),
        "category": cat,
        "quantity": qty,
        "gmv_aed": gmv,
        "discount_pct": np.round(discount, 3),
        "net_revenue_aed": net,
        "sale_event": np.where(sale_event, "Yellow Friday", "Regular"),
        "delivery_type": dtype,
        "promised_days": promised,
        "actual_days": np.round(actual, 2),
        "late_delivery": late,
        "returned": returned,
        "rating": rating.astype(int),
    })
    return df.sort_values("order_date").reset_index(drop=True)


df_all = make_data()

# --------------------------------------------------------------------------
# 2. SIDEBAR FILTERS
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🛒 Filters")
countries = st.sidebar.multiselect("Country", sorted(df_all.country.unique()), default=sorted(df_all.country.unique()))
cats = st.sidebar.multiselect("Category", sorted(df_all.category.unique()), default=sorted(df_all.category.unique()))
channel = st.sidebar.multiselect("Channel", ["App", "Web"], default=["App", "Web"])
d0, d1 = df_all.order_date.min().date(), df_all.order_date.max().date()
date_range = st.sidebar.date_input("Order date range", (d0, d1), min_value=d0, max_value=d1)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start, end = pd.to_datetime(d0), pd.to_datetime(d1)

df = df_all[
    df_all.country.isin(countries)
    & df_all.category.isin(cats)
    & df_all.channel.isin(channel)
    & df_all.order_date.between(start, end)
]

st.sidebar.markdown("---")
st.sidebar.download_button("⬇️ Download dataset (CSV)", df_all.to_csv(index=False), "noon_synthetic_orders_2025.csv", "text/csv")
st.sidebar.caption("Synthetic dataset (seed 42) built for the SBR assignment — not real Noon data.")

# --------------------------------------------------------------------------
# 3. HEADER + KPIs
# --------------------------------------------------------------------------
st.markdown(
    f"<div style='background:{NOON_YELLOW};padding:18px 24px;border-radius:10px;margin-bottom:12px'>"
    f"<h1 style='color:{INK};margin:0'>Noon E-Commerce Analytics Dashboard</h1>"
    f"<p style='color:{INK};margin:4px 0 0'>SBR · Data Analytics · Group: Harsh, Hunar, Saachi (CEO), Maansi, Keshav, Devansh</p></div>",
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("No orders match these filters.")
    st.stop()

orders_per_cust = df.groupby("customer_id").size()
repeat_rate = (orders_per_cust > 1).mean()
k = st.columns(6)
k[0].metric("Orders", f"{len(df):,}")
k[1].metric("Net revenue (AED)", f"{df.net_revenue_aed.sum()/1e6:,.2f}M")
k[2].metric("Avg order value", f"AED {df.net_revenue_aed.mean():,.0f}")
k[3].metric("Repeat-customer rate", f"{repeat_rate:.0%}")
k[4].metric("On-time delivery", f"{1 - df.late_delivery.mean():.0%}")
k[5].metric("Return rate", f"{df.returned.mean():.1%}")

tabs = st.tabs([
    "📈 Overview",
    "PS1 · Customer retention",
    "PS2 · Delivery & returns",
    "PS3 · Discount effectiveness",
    "🗂 Data",
])
tpl = dict(template="plotly_white", color_discrete_sequence=[INK, "#F5B700", "#3A86FF", "#8338EC", "#FF006E", "#2A9D8F"])

# ---- Overview ----
with tabs[0]:
    c1, c2 = st.columns([2, 1])
    monthly = df.set_index("order_date").resample("MS").net_revenue_aed.sum().reset_index()
    c1.plotly_chart(px.line(monthly, x="order_date", y="net_revenue_aed", markers=True,
                            title="Monthly net revenue (AED)", **tpl), width="stretch")
    c2.plotly_chart(px.pie(df, names="country", values="net_revenue_aed", hole=0.5,
                           title="Revenue share by country", **tpl), width="stretch")
    c3, c4 = st.columns(2)
    bycat = df.groupby("category", as_index=False).net_revenue_aed.sum().sort_values("net_revenue_aed")
    c3.plotly_chart(px.bar(bycat, x="net_revenue_aed", y="category", orientation="h",
                           title="Net revenue by category", **tpl), width="stretch")
    c4.plotly_chart(px.histogram(df, x="payment_method", color="channel", barmode="group",
                                 title="Orders by payment method & channel", **tpl), width="stretch")

# ---- PS1 Retention ----
with tabs[1]:
    st.subheader("Problem statement 1: Low repeat purchase among first-time customers")
    st.caption("How many customers come back, and do noon One members, age groups or countries retain better?")
    cust = df.groupby("customer_id").agg(
        orders=("order_id", "count"), spend=("net_revenue_aed", "sum"),
        member=("noon_one_member", "first"), age=("age_group", "first"), country=("country", "first"),
        avg_rating=("rating", "mean"))
    cust["segment"] = pd.cut(cust.orders, [0, 1, 3, 6, 999], labels=["One-time", "2-3 orders", "4-6 orders", "Loyal (7+)"])
    c1, c2 = st.columns(2)
    seg = cust.segment.value_counts().reindex(["One-time", "2-3 orders", "4-6 orders", "Loyal (7+)"]).reset_index()
    seg.columns = ["segment", "customers"]
    c1.plotly_chart(px.bar(seg, x="segment", y="customers", title="Customers by purchase frequency", **tpl), width="stretch")
    mem = cust.groupby("member").apply(lambda g: pd.Series({"repeat_rate": (g.orders > 1).mean(), "avg_spend": g.spend.mean()})).reset_index()
    mem["member"] = mem.member.map({True: "noon One member", False: "Non-member"})
    c2.plotly_chart(px.bar(mem, x="member", y="repeat_rate", text_auto=".0%", title="Repeat rate: members vs non-members", **tpl), width="stretch")
    c3, c4 = st.columns(2)
    age = cust.groupby(["age", "country"]).orders.mean().reset_index()
    c3.plotly_chart(px.bar(age, x="age", y="orders", color="country", barmode="group", title="Avg orders per customer by age & country", **tpl), width="stretch")
    c4.plotly_chart(px.box(cust.reset_index(), x="segment", y="avg_rating", title="Avg rating by customer segment", **tpl), width="stretch")
    st.info("**Insight to discuss:** customers who rate their experience lower order less often. "
            "Recommendation direction: a first-to-second-order nudge (welcome voucher, noon One trial) targeted at one-time buyers.")

# ---- PS2 Delivery & returns ----
with tabs[2]:
    st.subheader("Problem statement 2: Late deliveries are driving returns and low ratings")
    st.caption("Where is Noon missing its delivery promise, and how does that affect returns and ratings?")
    c1, c2 = st.columns(2)
    ot = df.groupby(["delivery_type", "country"]).late_delivery.mean().reset_index()
    ot["on_time"] = 1 - ot.late_delivery
    c1.plotly_chart(px.bar(ot, x="delivery_type", y="on_time", color="country", barmode="group", text_auto=".0%",
                           title="On-time delivery by service & country", **tpl), width="stretch")
    rr = df.groupby(["category", "late_delivery"]).returned.mean().reset_index()
    rr["late_delivery"] = rr.late_delivery.map({True: "Late", False: "On time"})
    c2.plotly_chart(px.bar(rr, x="category", y="returned", color="late_delivery", barmode="group", text_auto=".0%",
                           title="Return rate: late vs on-time orders", **tpl), width="stretch")
    c3, c4 = st.columns(2)
    rt = df.groupby("late_delivery").rating.mean().reset_index()
    rt["late_delivery"] = rt.late_delivery.map({True: "Late", False: "On time"})
    c3.plotly_chart(px.bar(rt, x="late_delivery", y="rating", text_auto=".2f", title="Average rating", range_y=[0, 5], **tpl), width="stretch")
    city = df.groupby("city").late_delivery.mean().sort_values(ascending=False).reset_index()
    c4.plotly_chart(px.bar(city, x="city", y="late_delivery", text_auto=".0%", title="Late-delivery rate by city", **tpl), width="stretch")
    st.info("**Insight to discuss:** late orders are returned more often and rated lower; Egyptian cities lag the Gulf. "
            "Recommendation direction: last-mile capacity in the weakest cities and realistic delivery promises at checkout.")

# ---- PS3 Discount ----
with tabs[3]:
    st.subheader("Problem statement 3: Are deep discounts (Yellow Friday) eroding profitability?")
    st.caption("Do bigger discounts bring proportionally more revenue, or just more returns?")
    df3 = df.assign(discount_band=pd.cut(df.discount_pct, [-0.01, 0.1, 0.2, 0.35, 1], labels=["0-10%", "10-20%", "20-35%", "35%+"]))
    c1, c2 = st.columns(2)
    band = df3.groupby("discount_band", observed=True).agg(orders=("order_id", "count"), return_rate=("returned", "mean"),
                                                           aov=("net_revenue_aed", "mean")).reset_index()
    c1.plotly_chart(px.bar(band, x="discount_band", y="orders", title="Orders by discount band", **tpl), width="stretch")
    c2.plotly_chart(px.line(band, x="discount_band", y="return_rate", markers=True, title="Return rate by discount band", **tpl), width="stretch")
    c3, c4 = st.columns(2)
    ev = df.groupby(["sale_event", "category"]).agg(gmv=("gmv_aed", "sum"), net=("net_revenue_aed", "sum")).reset_index()
    ev["discount_given"] = ev.gmv - ev.net
    c3.plotly_chart(px.bar(ev, x="category", y="discount_given", color="sale_event", barmode="group",
                           title="Discount value given away (AED)", **tpl), width="stretch")
    c4.plotly_chart(px.bar(band, x="discount_band", y="aov", text_auto=".0f", title="Average order value (AED) by discount band", **tpl), width="stretch")
    st.info("**Insight to discuss:** discounts above 35% bring volume but carry the highest return rate. "
            "Recommendation direction: cap blanket discounts and move to targeted, category-specific offers.")

# ---- Data ----
with tabs[4]:
    st.subheader("Filtered order data")
    st.dataframe(df, width="stretch", height=480)
    st.caption(f"{len(df):,} of {len(df_all):,} orders shown. Columns: " + ", ".join(df.columns))
