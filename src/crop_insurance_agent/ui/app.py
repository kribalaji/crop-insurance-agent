"""Streamlit UI for submitting, reviewing and inspecting claims.

Talks only to the FastAPI backend (API_URL, default http://127.0.0.1:8000), so
approvals go through the same validation and locking as any other client.

    uv run streamlit run src/crop_insurance_agent/ui/app.py
"""

from datetime import date
from typing import Any

import streamlit as st

from crop_insurance_agent.ui.client import ApiClient, ApiError

STATUS_BADGE = {
    "awaiting_approval": "🟡 awaiting approval",
    "completed": "🟢 completed",
    "failed": "🔴 failed",
    "stalled": "🟠 stalled",
}


def api() -> ApiClient:
    # Tests inject a client backed by the FastAPI app itself.
    if "api" not in st.session_state:
        st.session_state["api"] = ApiClient.from_env()
    return st.session_state["api"]


def show_error(exc: ApiError) -> None:
    if isinstance(exc.detail, dict) and exc.detail.get("status") == "stalled":
        st.error(f"Claim stalled: {exc.detail.get('detail')}")
        st.info(exc.detail.get("hint", ""))
    else:
        st.error(f"{exc.status_code}: {exc.detail}")


def rs(x: float | None) -> str:
    return "—" if x is None else f"Rs {x:,.0f}"


def render_claim(view: dict[str, Any]) -> None:
    st.subheader(f"Claim {view['claim_id']} — {STATUS_BADGE[view['status']]}")

    if err := view.get("error"):
        st.error(f"Run failed at **{err['agent']}** ({err['kind']}): {err['message']}")
    if view["status"] == "stalled":
        st.warning("An unexpected error stopped this claim. Fix the cause, then retry.")
        if st.button("Retry", key=f"retry-{view['claim_id']}"):
            try:
                api().retry(view["claim_id"])
                st.rerun()
            except ApiError as exc:
                show_error(exc)

    if adv := view.get("advisory"):
        st.markdown("#### Message to the farmer")
        st.info(f"**{adv['headline']}**\n\n{adv['explanation']}")
        if adv["generated_by"] == "template":
            st.caption(
                "Template wording: the LLM text failed the number check. " + " ".join(adv["notes"])
            )

    if claim := view.get("claim"):
        appr = claim["approval"]
        st.markdown("#### Decision")
        cols = st.columns(3)
        cols[0].metric("Status", claim["status"])
        cols[1].metric("Payout", rs(claim["payout_amount"]))
        cols[2].metric("Reviewer", appr["reviewer"])
        if appr.get("comment"):
            st.caption(f"Reviewer comment: {appr['comment']}")

    if w := view.get("weather"):
        st.markdown("#### Weather")
        r, g = w["rainfall"], w["gdu"]
        cols = st.columns(4)
        cols[0].metric("Rainfall", f"{r['actual_mm']:g} mm", f"usual {r['normal_mm']:g} mm")
        cols[1].metric("Deficit", "—" if r["deficit_pct"] is None else f"{r['deficit_pct']:g}%")
        cols[2].metric("IMD category", r["category"] or "—")
        cols[3].metric(
            "GDU vs normal", "—" if g["deviation_pct"] is None else f"{g['deviation_pct']:+g}%"
        )
        st.caption(
            f"{w['source'].upper()} grid point ({w['grid_latitude']}, {w['grid_longitude']}), "
            f"{w['season_start']} to {w['season_end']}, "
            f"baseline {len(w['baseline_years'])} seasons. " + " ".join(w["notes"])
        )

    if p := view.get("policy"):
        st.markdown(f"#### Policy `{p['policy_id']}`")
        st.write(
            f"Crop covered: **{'yes' if p['crop_covered'] else 'no'}** · "
            f"Triggered: **{'yes' if p['triggered'] else 'no'}** · "
            f"Sum insured: **{rs(p['sum_insured_per_ha'])}/ha**"
        )
        st.dataframe(
            [
                {
                    "cover": e["trigger"]["name"],
                    "condition": f"{e['trigger']['metric']} {e['trigger']['comparator']} "
                    f"{e['trigger']['threshold']:g}",
                    "observed": e["observed"],
                    "met": e["met"],
                    "quote verified": e["quote_verified"],
                }
                for e in p["evaluations"]
            ],
            hide_index=True,
        )
        with st.expander("Policy quotes and retrieved excerpts"):
            for e in p["evaluations"]:
                st.markdown(
                    f"**{e['trigger']['name']}** (p. {e['trigger']['page']}): "
                    f"> {e['trigger']['source_quote']}"
                )
            for c in p["citations"]:
                st.caption(f"p. {c['page']}: {c['text']}")
        for note in p["notes"]:
            st.caption(note)


def render_review(view: dict[str, Any]) -> None:
    pending = view["pending_review"]
    est = pending["estimate"]
    st.markdown(f"### {view['claim_id']} — {est['policy_id']}")
    for a in pending["assumptions"]:
        st.warning(a, icon="⚠️")
    if est["needs_manual_assessment"]:
        st.error("Some triggered covers could not be priced; the amount may be understated.")
    cols = st.columns(3)
    cols[0].metric("Estimated payout", rs(est["payout_amount"]))
    cols[1].metric("Of sum insured", f"{est['total_payout_pct']:g}%")
    cols[2].metric("Area", f"{est['area_ha']:g} ha")
    st.dataframe(
        [
            {
                "cover": c["name"],
                "observed": c["observed"],
                "trigger": c["threshold"],
                "exit": c["exit_threshold"],
                "method": c["method"],
                "payout %": c["payout_pct"],
            }
            for c in est["covers"]
        ],
        hide_index=True,
    )
    for n in est["notes"]:
        st.caption(n)

    with st.form(f"decide-{view['claim_id']}"):
        reviewer = st.text_input("Reviewer", key=f"reviewer-{view['claim_id']}")
        comment = st.text_input("Comment (optional)", key=f"comment-{view['claim_id']}")
        approve = st.form_submit_button("Approve", type="primary")
        reject = st.form_submit_button("Reject")
    if approve or reject:
        if not reviewer.strip():
            st.error("Enter the reviewer's name.")
            return
        try:
            done = api().decide(view["claim_id"], approve, reviewer.strip(), comment.strip())
            st.success(f"Claim {done['claim_id']} {STATUS_BADGE[done['status']]}")
        except ApiError as exc:
            show_error(exc)


def submit_tab() -> None:
    try:
        policies = api().policies()
    except Exception as exc:  # backend down: say so instead of a traceback
        st.error(f"Cannot reach the API: {exc}")
        return
    with st.form("submit"):
        c1, c2 = st.columns(2)
        claim_id = c1.text_input("Claim id (blank for automatic)")
        policy_id = c2.selectbox("Policy", policies)
        lat = c1.number_input("Latitude", value=10.79, format="%.4f")
        lon = c2.number_input("Longitude", value=79.14, format="%.4f")
        start = c1.date_input("Season start", value=date(2023, 6, 1))
        end = c2.date_input("Season end", value=date(2023, 9, 30))
        crop = c1.text_input("Crop", value="rice")
        source = c2.selectbox("Weather source", ["era5", "imd"])
        area = c1.number_input("Insured area (ha)", min_value=0.01, value=2.0)
        submitted = st.form_submit_button("Submit claim", type="primary")
    if not submitted:
        return
    body: dict[str, Any] = {
        "weather": {
            "latitude": lat,
            "longitude": lon,
            "season_start": start.isoformat(),
            "season_end": end.isoformat(),
            "crop": crop,
            "source": source,
        },
        "policy_id": policy_id,
        "area_ha": area,
    }
    if claim_id.strip():
        body["claim_id"] = claim_id.strip()
    with st.spinner("Running weather and policy agents..."):
        try:
            view = api().submit(body)
        except ApiError as exc:
            show_error(exc)
            return
    st.success(f"Claim {view['claim_id']}: {STATUS_BADGE[view['status']]}")
    if view["status"] == "awaiting_approval":
        st.info("Open the Review queue tab to approve or reject it.")
    render_claim(view)


def review_tab() -> None:
    waiting = api().claims("awaiting_approval")
    if not waiting:
        st.info("No claims are waiting for approval.")
        return
    for summary in waiting:
        render_review(api().claim(summary["claim_id"]))
        st.divider()


def claims_tab() -> None:
    status = st.selectbox(
        "Status", ["all", *STATUS_BADGE], format_func=lambda s: STATUS_BADGE.get(s, s)
    )
    rows = api().claims(None if status == "all" else status)
    if not rows:
        st.info("No claims.")
        return
    st.dataframe(
        [
            {
                "claim": r["claim_id"],
                "status": STATUS_BADGE[r["status"]],
                "policy": r["policy_id"],
                "payout": rs(r["payout_amount"]),
                "error": r["error"],
            }
            for r in rows
        ],
        hide_index=True,
    )
    chosen = st.selectbox("Open claim", [r["claim_id"] for r in rows])
    if chosen:
        render_claim(api().claim(chosen))


def main() -> None:
    st.set_page_config(page_title="Crop insurance claims", page_icon="🌾", layout="wide")
    st.title("🌾 Crop insurance claims")
    st.caption("Sample policies are illustrative. Every payout needs a human reviewer.")
    submit, review, claims = st.tabs(["Submit claim", "Review queue", "All claims"])
    with submit:
        submit_tab()
    with review:
        review_tab()
    with claims:
        claims_tab()


main()
