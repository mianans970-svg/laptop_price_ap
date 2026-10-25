import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import laptop_utils  # noqa: F401  (the saved model references laptop_utils.safe_expm1)

BASE = Path(__file__).parent
MODEL_PATH = BASE / "laptop_price_model.joblib"
META_PATH = BASE / "model_meta.json"

st.set_page_config(page_title="Laptop Price Predictor (PKR)", page_icon="💻", layout="centered")


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


model, meta = load_assets()
cats, nums = meta["categories"], meta["numeric"]

st.title("💻 Laptop Price Predictor")
st.caption(
    f"Predicts the market price in Pakistani Rupees (PKR). Model: **{meta['best_model']}** "
    f"| 10-fold CV RMSE ≈ Rs {meta['cv_rmse_pkr']:,.0f} | R² ≈ {meta['cv_r2']:.2f}"
)

st.subheader("Laptop specifications")
c1, c2 = st.columns(2)
with c1:
    company = st.selectbox("Brand", cats["Company"])
    type_name = st.selectbox("Type", cats["TypeName"])
    cpu_family = st.selectbox("Processor family", cats["Cpu_Family"])
    gpu_brand = st.selectbox("Graphics card brand", cats["Gpu_Brand"])
    opsys = st.selectbox("Operating system", cats["OpSys"])
    touch = st.radio("Touchscreen", ["No", "Yes"], horizontal=True)
    ips = st.radio("IPS panel", ["No", "Yes"], horizontal=True)
with c2:
    inches = st.number_input("Screen size (inches)", nums["Inches"]["min"], nums["Inches"]["max"], nums["Inches"]["default"], 0.1)
    res = st.selectbox("Screen resolution", meta["resolutions"], index=meta["resolutions"].index(meta["default_resolution"]))
    cpu_ghz = st.number_input("CPU speed (GHz)", nums["Cpu_Speed_GHz"]["min"], nums["Cpu_Speed_GHz"]["max"], nums["Cpu_Speed_GHz"]["default"], 0.1)
    ram = st.selectbox("RAM (GB)", meta["ram_options"], index=meta["ram_options"].index(8))
    ssd = st.selectbox("SSD / Flash storage (GB)", meta["ssd_options"], index=meta["ssd_options"].index(256))
    hdd = st.selectbox("HDD / Hybrid storage (GB)", meta["hdd_options"], index=0)
    weight = st.number_input("Weight (kg)", nums["Weight_kg"]["min"], nums["Weight_kg"]["max"], nums["Weight_kg"]["default"], 0.05)

if st.button("Predict price", type="primary", use_container_width=True):
    w, h = (int(v) for v in res.split("x"))
    row = pd.DataFrame([{
        "Company": company, "TypeName": type_name, "Inches": inches,
        "Screen_Width": w, "Screen_Height": h, "Touchscreen": touch, "IPS_Panel": ips,
        "Cpu_Family": cpu_family, "Cpu_Speed_GHz": cpu_ghz, "Ram_GB": ram,
        "SSD_GB": ssd, "HDD_GB": hdd, "Gpu_Brand": gpu_brand, "OpSys": opsys, "Weight_kg": weight,
    }])[meta["features"]]
    pkr = float(model.predict(row)[0])
    low, high = pkr * (1 - meta["cv_mape"]), pkr * (1 + meta["cv_mape"])
    st.success(f"Estimated price: **Rs {pkr:,.0f}**")
    st.write(f"Typical range (± average model error of {meta['cv_mape']*100:.0f}%): Rs {low:,.0f} – Rs {high:,.0f}")
    st.caption(f"≈ €{pkr / meta['eur_to_pkr']:,.0f} at 1 EUR = {meta['eur_to_pkr']} PKR")

st.divider()
st.caption("Trained on the Kaggle laptop price dataset (1,275 unique laptops). Prices converted from EUR to PKR at the rate shown above. Estimates are indicative only.")
