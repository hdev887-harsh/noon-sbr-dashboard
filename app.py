"""
Noon E-Commerce Analytics Dashboard — SBR (Data Analytics)
Group: Harsh, Hunar, Saachi (CEO), Maansi, Keshav, Devansh

NOTE: The datasets are SYNTHETIC (generated with a fixed seed) for academic
illustration. They are modelled on Noon's UAE/KSA/Egypt marketplace but do not
contain real Noon data.

Problem statements
  PS2: Experience failures (slow refunds, late deliveries, weak support) hurt
       some customers more than others.
  PS3: Customers lapse, and engagement (marketing messages) is badly timed.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Noon Analytics Dashboard", page_icon="🛒", layout="wide")

NOON_YELLOW = "#FEEE00"
INK = "#1F1F1F"
SNAPSHOT = pd.Timestamp("2025-12-31")


# --------------------------------------------------------------------------
# 1. DATA (customers -> orders simulated sequentially, then marketing messages)
# --------------------------------------------------------------------------
@st.cache_data
def make_data(n_customers: int = 4000, seed: int = 42):
    rng = np.random.default_rng(seed)
    cities = {"UAE": ["Dubai", "Abu Dhabi", "Sharjah"],
              "KSA": ["Riyadh", "Jeddah", "Dammam"],
              "Egypt": ["Cairo", "Alexandria", "Giza"]}
    categories = {"Electronics": 950, "Fashion": 180, "Beauty": 120,
                  "Home & Kitchen": 260, "Grocery (Minutes)": 85, "Toys": 140}
    cat_names, cat_p = list(categories), [0.2, 0.22, 0.14, 0.14, 0.2, 0.1]
    cat_return = {"Electronics": 0.07, "Fashion": 0.20, "Beauty": 0.06,
                  "Home & Kitchen": 0.08, "Grocery (Minutes)": 0.02, "Toys": 0.06}
    payments = ["Card", "Cash on Delivery", "Apple Pay", "BNPL (Tabby/Tamara)"]

    cust = pd.DataFrame({"customer_id": np.arange(1, n_customers + 1)})
    cust["country"] = rng.choice(["UAE", "KSA", "Egypt"], n_customers, p=[0.5, 0.35, 0.15])
    cust["city"] = [rng.choice(cities[c]) for c in cust.country]
    cust["age_group"] = rng.choice(["18-24", "25-34", "35-44", "45+"], n_customers, p=[0.24, 0.40, 0.24, 0.12])
    cust["noon_one_member"] = rng.random(n_customers) < 0.28
    cust["payment_pref"] = [rng.choice(payments, p=[0.25, 0.55, 0.12, 0.08]) if c == "Egypt"
                            else rng.choice(payments, p=[0.45, 0.22, 0.21, 0.12]) for c in cust.country]
    cust["active_hour"] = rng.choice(np.arange(8, 24), n_customers,
                                     p=np.array([1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 3, 4, 5, 5, 4, 2]) / 36)
    acq_days = rng.integers(0, 330, n_customers)
    cust["first_order_date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(acq_days, unit="D")

    rows = []
    oid = 100001
    for c in cust.itertuples(index=False):
        t, k = c.first_order_date, 1
        while t <= SNAPSHOT:
            cat = rng.choice(cat_names, p=cat_p)
            dtype = "noon Minutes" if cat == "Grocery (Minutes)" else rng.choice(["Express", "Standard"], p=[0.45, 0.55])
            pay = c.payment_pref if rng.random() < 0.8 else rng.choice(payments)
            p_late = {"UAE": 0.10, "KSA": 0.14, "Egypt": 0.27}[c.country] * (0.6 if dtype == "noon Minutes" else 1)
            late = rng.random() < p_late
            returned = rng.random() < cat_return[cat] + 0.05 * late
            refund_days = np.nan
            if returned:
                base = 9 if pay == "Cash on Delivery" else 4
                refund_days = round(base * rng.lognormal(0, 0.45) * (1.3 if c.country == "Egypt" else 1), 1)
            slow_refund = bool(returned and refund_days > 7)
            support = rng.random() < 0.08 + 0.45 * late + 0.55 * returned
            res_hours = round(rng.lognormal(np.log(20 if c.country != "Egypt" else 34), 0.7), 1) if support else np.nan
            weak_support = bool(support and res_hours > 48)
            failure = late or slow_refund or weak_support
            csat = np.clip(round(rng.normal(4.4, 0.5) - 0.9 * late - 1.0 * slow_refund - 1.1 * weak_support), 1, 5)
            gmv = round(categories[cat] * rng.lognormal(0, 0.35) * rng.integers(1, 3), 2)
            disc = float(np.clip(rng.normal(0.12, 0.06), 0, 0.6))
            rows.append((oid, t, c.customer_id, k, cat, dtype, pay, gmv, round(gmv * (1 - disc), 2),
                         late, returned, refund_days, slow_refund, support, res_hours, weak_support,
                         failure, int(csat), rng.choice(["App", "Web"], p=[0.78, 0.22])))
            oid += 1
            # Who gets hurt: sensitivity to a failure differs by segment
            sens = (1 + 0.8 * (k == 1) + 0.5 * (pay == "Cash on Delivery") + 0.3 * (c.country == "Egypt")
                    + 0.3 * (c.age_group == "45+") - 0.6 * c.noon_one_member)
            p_continue = 0.80 + 0.08 * c.noon_one_member - (0.22 * sens if failure else 0)
            if rng.random() > max(p_continue, 0.05):
                break
            t = t + pd.Timedelta(days=int(rng.exponential(28 if c.noon_one_member else 40)) + 1)
            k += 1

    orders = pd.DataFrame(rows, columns=[
        "order_id", "order_date", "customer_id", "order_number", "category", "delivery_type", "payment_method",
        "gmv_aed", "net_revenue_aed", "late_delivery", "returned", "refund_days", "slow_refund",
        "support_contact", "support_resolution_hours", "weak_support", "experience_failure", "csat", "channel"])
    orders = orders.merge(cust[["customer_id", "country", "city", "age_group", "noon_one_member"]], on="customer_id")
    orders = orders.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    orders["ordered_again"] = orders.groupby("customer_id").order_id.shift(-1).notna()
    orders["customer_type"] = np.where(orders.order_number == 1, "First order", "Repeat order")

    # customer status at snapshot
    last = orders.groupby("customer_id").agg(last_order=("order_date", "max"), orders=("order_id", "count"),
                                             failures=("experience_failure", "sum"),
                                             last_failed=("experience_failure", "last"))
    cust = cust.merge(last, on="customer_id")
    cust["days_since_last_order"] = (SNAPSHOT - cust.last_order).dt.days
    cust["status"] = pd.cut(cust.days_since_last_order, [-1, 30, 90, 9999],
                            labels=["Active (≤30d)", "Cooling (31-90d)", "Lapsed (>90d)"])

    # marketing messages
    odates = orders.groupby("customer_id").order_date.apply(lambda s: s.values).to_dict()
    ofail = orders.groupby("customer_id").experience_failure.apply(lambda s: s.values).to_dict()
    msgs = []
    mid = 500001
    for c in cust.itertuples(index=False):
        n = rng.poisson(9)
        span = (SNAPSHOT - c.first_order_date).days
        for d in np.sort(rng.integers(0, max(span, 1), n)):
            send_date = c.first_order_date + pd.Timedelta(days=int(d))
            ods = odates[c.customer_id]
            prior = ods[ods <= np.datetime64(send_date)]
            if len(prior) == 0:
                continue
            last_o = pd.Timestamp(prior[-1])
            dsl = (send_date - last_o).days
            after_failure = bool(ofail[c.customer_id][len(prior) - 1]) and dsl <= 5
            hour = int(rng.choice([10, 13, 20], p=[0.35, 0.25, 0.4])) if rng.random() < 0.75 else int(rng.choice(np.arange(8, 24)))
            in_window = abs(hour - c.active_hour) <= 2
            channel = rng.choice(["Push", "Email", "SMS", "WhatsApp"], p=[0.45, 0.30, 0.15, 0.10])
            p = 0.02 + 0.045 * in_window
            p += (0.05 if 14 <= dsl <= 45 else 0) + (0.02 if 46 <= dsl <= 90 else 0)
            p *= 0.35 if dsl < 7 else 1
            p *= 0.4 if dsl > 120 else 1
            p *= 0.3 if after_failure else 1
            p *= {"Push": 1, "Email": 0.7, "SMS": 0.8, "WhatsApp": 1.4}[channel]
            opened = rng.random() < min(0.15 + 0.25 * in_window + 0.6 * p, 0.95)
            converted = opened and rng.random() < min(p * 2.2, 0.9)
            unsub = rng.random() < (0.004 + 0.02 * (dsl < 7) + 0.05 * after_failure + 0.01 * (not in_window))
            msgs.append((mid, send_date, hour, c.customer_id, c.country, channel, dsl, in_window,
                         after_failure, opened, converted, unsub))
            mid += 1
    messages = pd.DataFrame(msgs, columns=[
        "message_id", "send_date", "send_hour", "customer_id", "country", "channel", "days_since_last_order",
        "in_active_window", "sent_within_5d_of_failure", "opened", "converted", "unsubscribed"])
    messages["recency_bucket"] = pd.cut(messages.days_since_last_order, [-1, 6, 13, 30, 45, 90, 120, 9999],
                                        labels=["0-6d", "7-13d", "14-30d", "31-45d", "46-90d", "91-120d", "120d+"])
    messages["timing"] = np.select(
        [messages.sent_within_5d_of_failure, messages.days_since_last_order < 7,
         messages.days_since_last_order > 90, ~messages.in_active_window],
        ["Right after a failure", "Too soon (<7d)", "Too late (>90d)", "Wrong hour"], "Well timed")
    return cust, orders, messages


customers_all, orders_all, messages_all = make_data()

# --------------------------------------------------------------------------
# 2. SIDEBAR FILTERS
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🛒 Filters")
countries = st.sidebar.multiselect("Country", ["UAE", "KSA", "Egypt"], default=["UAE", "KSA", "Egypt"])
cats = st.sidebar.multiselect("Category", sorted(orders_all.category.unique()), default=sorted(orders_all.category.unique()))
member_f = st.sidebar.radio("noon One membership", ["All", "Members", "Non-members"], horizontal=True)

orders = orders_all[orders_all.country.isin(countries) & orders_all.category.isin(cats)]
customers = customers_all[customers_all.country.isin(countries)]
messages = messages_all[messages_all.country.isin(countries)]
if member_f != "All":
    flag = member_f == "Members"
    orders = orders[orders.noon_one_member == flag]
    customers = customers[customers.noon_one_member == flag]
    messages = messages[messages.customer_id.isin(customers.customer_id)]

st.sidebar.markdown("---")
st.sidebar.download_button("⬇️ Orders dataset (CSV)", orders_all.to_csv(index=False), "noon_orders_2025.csv", "text/csv")
st.sidebar.download_button("⬇️ Customers dataset (CSV)", customers_all.to_csv(index=False), "noon_customers_2025.csv", "text/csv")
st.sidebar.download_button("⬇️ Marketing messages (CSV)", messages_all.to_csv(index=False), "noon_messages_2025.csv", "text/csv")
st.sidebar.caption("Synthetic datasets (seed 42) built for the SBR assignment — not real Noon data.")

# --------------------------------------------------------------------------
# 3. HEADER + KPIs
# --------------------------------------------------------------------------
st.markdown(
    f"<div style='background:{NOON_YELLOW};padding:18px 24px;border-radius:10px;margin-bottom:12px'>"
    f"<h1 style='color:{INK};margin:0'>Noon E-Commerce Analytics Dashboard</h1>"
    f"<p style='color:{INK};margin:4px 0 0'>SBR · Data Analytics · Group: Harsh, Hunar, Saachi (CEO), Maansi, Keshav, Devansh</p></div>",
    unsafe_allow_html=True,
)
if orders.empty or customers.empty:
    st.warning("No data matches these filters.")
    st.stop()

k = st.columns(3) + st.columns(3)
k[0].metric("Orders", f"{len(orders):,}", border=True)
k[1].metric("Net revenue (AED)", f"{orders.net_revenue_aed.sum()/1e6:,.2f}M", border=True)
k[2].metric("Customers", f"{customers.customer_id.nunique():,}", border=True)
k[3].metric("Orders with an experience failure", f"{orders.experience_failure.mean():.1%}", border=True)
k[4].metric("Lapsed customers (>90 days)", f"{(customers.status == 'Lapsed (>90d)').mean():.0%}", border=True)
k[5].metric("Marketing message conversion", f"{messages.converted.mean():.2%}", border=True)

tabs = st.tabs(["📈 Overview", "PS2 · Experience failures", "PS3 · Lapse & engagement timing", "🗂 Data"])
tpl = dict(template="plotly_white", color_discrete_sequence=[INK, "#F5B700", "#3A86FF", "#8338EC", "#FF006E", "#2A9D8F"])
pct = dict(tickformat=".0%")

# ---- Overview ----
with tabs[0]:
    c1, c2 = st.columns([2, 1])
    monthly = orders.set_index("order_date").resample("MS").net_revenue_aed.sum().reset_index()
    c1.plotly_chart(px.line(monthly, x="order_date", y="net_revenue_aed", markers=True,
                            title="Monthly net revenue (AED)", **tpl), width="stretch")
    c2.plotly_chart(px.pie(orders, names="country", values="net_revenue_aed", hole=0.5,
                           title="Revenue share by country", **tpl), width="stretch")
    c3, c4 = st.columns(2)
    bycat = orders.groupby("category", as_index=False).net_revenue_aed.sum().sort_values("net_revenue_aed")
    c3.plotly_chart(px.bar(bycat, x="net_revenue_aed", y="category", orientation="h",
                           title="Net revenue by category", **tpl), width="stretch")
    st_counts = customers.status.value_counts().reindex(["Active (≤30d)", "Cooling (31-90d)", "Lapsed (>90d)"]).reset_index()
    st_counts.columns = ["status", "customers"]
    c4.plotly_chart(px.bar(st_counts, x="status", y="customers", title="Customer base by status (31 Dec 2025)", **tpl), width="stretch")

# ---- PS2 Experience failures ----
with tabs[1]:
    st.subheader("Problem statement 2: Experience failures hurt some customers more than others")
    st.caption("Slow refunds (>7 days), late deliveries and weak support (>48h to resolve). "
               "Who experiences them most, and whose loyalty do they damage most?")
    seg_label = st.selectbox("Compare customers by", ["Customer type", "Payment method", "Country", "noon One membership", "Age group"])
    seg_col = {"Customer type": "customer_type", "Payment method": "payment_method", "Country": "country",
               "noon One membership": "member_label", "Age group": "age_group"}[seg_label]
    o = orders.assign(member_label=np.where(orders.noon_one_member, "noon One member", "Non-member"))

    c1, c2 = st.columns(2)
    fr = o.groupby(seg_col)[["late_delivery", "slow_refund", "weak_support"]].mean().reset_index().melt(
        id_vars=seg_col, var_name="failure", value_name="rate")
    fr["failure"] = fr.failure.map({"late_delivery": "Late delivery", "slow_refund": "Slow refund", "weak_support": "Weak support"})
    f1 = px.bar(fr, x=seg_col, y="rate", color="failure", barmode="group", text_auto=".1%",
                title=f"How often each failure happens, by {seg_label.lower().replace('noon one', 'noon One')}", **tpl)
    f1.update_yaxes(**pct)
    c1.plotly_chart(f1, width="stretch")

    ra = o.groupby([seg_col, "experience_failure"]).ordered_again.mean().reset_index()
    ra["experience"] = ra.experience_failure.map({True: "Had a failure", False: "Smooth order"})
    f2 = px.bar(ra, x=seg_col, y="ordered_again", color="experience", barmode="group", text_auto=".0%",
                title="Share who ordered again after this order", **tpl)
    f2.update_yaxes(**pct)
    c2.plotly_chart(f2, width="stretch")

    piv = ra.pivot(index=seg_col, columns="experience", values="ordered_again")
    piv["Loyalty drop (pts)"] = (piv["Smooth order"] - piv["Had a failure"]) * 100
    piv = piv.sort_values("Loyalty drop (pts)", ascending=False).reset_index()
    c3, c4 = st.columns(2)
    f3 = px.bar(piv, x="Loyalty drop (pts)", y=seg_col, orientation="h", text_auto=".1f",
                title="Who gets hurt most: drop in reorder rate after a failure (percentage points)", **tpl)
    f3.update_yaxes(categoryorder="total ascending")
    c3.plotly_chart(f3, width="stretch")
    refunds = o[o.returned]
    c4.plotly_chart(px.box(refunds, x="payment_method", y="refund_days", color="country",
                           title="Refund time (days) by payment method & country", **tpl), width="stretch")

    c5, c6 = st.columns(2)
    sup = o[o.support_contact].copy()
    sup["resolution_band"] = pd.cut(sup.support_resolution_hours, [0, 12, 24, 48, 96, 9999],
                                    labels=["<12h", "12-24h", "24-48h", "48-96h", "96h+"])
    sb = sup.groupby("resolution_band", observed=True).agg(csat=("csat", "mean"), reorder=("ordered_again", "mean")).reset_index()
    c5.plotly_chart(px.bar(sb, x="resolution_band", y="csat", text_auto=".2f", range_y=[0, 5],
                           title="Support resolution time vs CSAT (1-5)", **tpl), width="stretch")
    nf = o.assign(n_fail=o[["late_delivery", "slow_refund", "weak_support"]].sum(axis=1)).groupby("n_fail").agg(
        reorder=("ordered_again", "mean"), orders=("order_id", "count")).reset_index()
    nf["n_fail"] = nf.n_fail.astype(str) + " failure(s)"
    f6 = px.bar(nf, x="n_fail", y="reorder", text_auto=".0%", title="Failures stack up: reorder rate by number of failures in one order", **tpl)
    f6.update_yaxes(**pct)
    c6.plotly_chart(f6, width="stretch")

    worst = piv.iloc[0]
    st.info(f"**Insight:** by {seg_label.lower().replace('noon one', 'noon One')}, **{worst[seg_col]}** customers lose the most loyalty after a bad experience "
            f"(reorder rate falls {worst['Loyalty drop (pts)']:.0f} points). First-time buyers, Cash-on-Delivery payers and Egypt "
            "are the most fragile, and COD refunds are the slowest. noon One members are the least affected. "
            "**Recommendation direction:** prioritise service recovery for fragile segments, such as instant store-credit refunds for COD "
            "and a priority support queue for first orders.")

# ---- PS3 Lapse & engagement ----
with tabs[2]:
    st.subheader("Problem statement 3: Customers lapse, and engagement is badly timed")
    st.caption("A customer is lapsed after 90+ days without an order. Engagement = push, email, SMS and WhatsApp messages. "
               "A message counts as well timed if it reaches the customer at the right point after their last order and near their usual active hour.")
    c1, c2 = st.columns(2)
    cu = customers.assign(member_label=np.where(customers.noon_one_member, "noon One member", "Non-member"),
                          last_order_failed=np.where(customers.last_failed, "Last order had a failure", "Last order was smooth"))
    lap = cu.groupby(["country", "member_label"]).status.apply(lambda s: (s == "Lapsed (>90d)").mean()).reset_index(name="lapse_rate")
    f1 = px.bar(lap, x="country", y="lapse_rate", color="member_label", barmode="group", text_auto=".0%",
                title="Lapse rate by country & membership", **tpl)
    f1.update_yaxes(**pct)
    c1.plotly_chart(f1, width="stretch")
    lf = cu.groupby("last_order_failed").status.apply(lambda s: (s == "Lapsed (>90d)").mean()).reset_index(name="lapse_rate")
    f2 = px.bar(lf, x="last_order_failed", y="lapse_rate", text_auto=".0%", title="Lapse rate vs how the last order went", **tpl)
    f2.update_yaxes(**pct)
    c2.plotly_chart(f2, width="stretch")

    c3, c4 = st.columns(2)
    tm = messages.timing.value_counts(normalize=True).reindex(
        ["Well timed", "Wrong hour", "Too soon (<7d)", "Too late (>90d)", "Right after a failure"]).reset_index()
    tm.columns = ["timing", "share"]
    c3.plotly_chart(px.pie(tm, names="timing", values="share", hole=0.5, title="How messages are timed today", **tpl), width="stretch")
    tc = messages.groupby("timing").agg(conversion=("converted", "mean"), unsubscribe=("unsubscribed", "mean")).reset_index().melt(
        id_vars="timing", var_name="metric", value_name="rate")
    f4 = px.bar(tc, x="timing", y="rate", color="metric", barmode="group", text_auto=".1%",
                title="Conversion and unsubscribe rate by timing", **tpl)
    f4.update_yaxes(**pct)
    c4.plotly_chart(f4, width="stretch")

    c5, c6 = st.columns(2)
    rb = messages.groupby("recency_bucket", observed=True).converted.mean().reset_index()
    f5 = px.line(rb, x="recency_bucket", y="converted", markers=True,
                 title="Conversion by days since the customer's last order", **tpl)
    f5.update_yaxes(**pct)
    c5.plotly_chart(f5, width="stretch")
    heat = messages.assign(hour_gap=(messages.send_hour - messages.customer_id.map(customers_all.set_index("customer_id").active_hour)).abs().clip(upper=6))
    hg = heat.groupby(["channel", "hour_gap"]).converted.mean().reset_index()
    hg["hour_gap"] = hg.hour_gap.map(lambda h: "6h+" if h >= 6 else f"{int(h)}h")
    c6.plotly_chart(px.density_heatmap(hg, x="hour_gap", y="channel", z="converted", histfunc="avg", text_auto=".1%",
                                       color_continuous_scale=["#FFFFFF", NOON_YELLOW, INK],
                                       title="Conversion by channel and hours away from the customer's active hour"),
                    width="stretch")

    badly = 1 - (messages.timing == "Well timed").mean()
    wt = messages[messages.timing == "Well timed"].converted.mean()
    bt = messages[messages.timing != "Well timed"].converted.mean()
    st.info(f"**Insight:** {badly:.0%} of messages are badly timed. Well-timed messages convert at {wt:.1%} against {bt:.1%} for the rest. "
            "Messages sent right after a failure have the highest unsubscribe rate. Conversion peaks 14-45 days after the last order "
            "and collapses after 120 days. **Recommendation direction:** trigger win-back messages at 14-45 days, send them in each "
            "customer's active hour, and hold promotions for 5 days after a failure (send a service-recovery message instead).")

# ---- Data ----
with tabs[3]:
    which = st.radio("Dataset", ["Orders", "Customers", "Marketing messages"], horizontal=True)
    data = {"Orders": orders, "Customers": customers, "Marketing messages": messages}[which]
    st.dataframe(data, width="stretch", height=480)
    st.caption(f"{len(data):,} rows (filtered). Columns: " + ", ".join(data.columns))
