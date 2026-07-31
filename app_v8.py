import json
import re
import ast
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
import streamlit as st

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="SD Model Calibrator v10 (Robust)", layout="wide")
st.title("🎯 System Dynamics Calibrator")
st.caption("เครื่องมือวิเคราะห์ข้อมูลและปรับจูนพารามิเตอร์เข้ากับข้อมูลจริง")
st.markdown("---")

# ==========================================
# 2. Session State Initialization
# ==========================================
if "df_data" not in st.session_state:
    st.session_state.df_data = None
if "model_json" not in st.session_state:
    st.session_state.model_json = None
if "raw_model_json" not in st.session_state:
    st.session_state.raw_model_json = None
if "mapping" not in st.session_state:
    st.session_state.mapping = {}
if "opt_results" not in st.session_state:
    st.session_state.opt_results = None
if "timeseries" not in st.session_state:
    # ⚡ ใหม่: เก็บ time-series ภายนอกที่ใช้อ้างอิงในสูตร (เช่น T_obs(t), DO_obs(t) ของโมเดลปลา)
    # แยกจาก df_data ซึ่งเป็นข้อมูลเป้าหมายที่ใช้ "เทียบ/คำนวณ RMSE" คนละหน้าที่กัน
    st.session_state.timeseries = {}

# ==========================================
# 2.5 ฟังก์ชันคณิตศาสตร์ + ตัวตรวจสอบสูตรแบบปลอดภัย (พอร์ตมาจาก sys-sim v7)
# ==========================================
def _mod(a, b): return a % b
def _ite(cond, a, b): return a if cond else b

MATH_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "min": min, "max": max, "abs": abs, "round": round, "mod": _mod,
    "if_then_else": _ite,
}
RESERVED_NAMES = set(MATH_FUNCS.keys()) | {"t"}

# ⚡ AST whitelist — หัวใจของการปิดช่องโหว่ sandbox-escape ของ eval() (พบและแก้แล้วใน v7)
# การ "ไม่รวม" ast.Attribute ในนี้คือสิ่งที่บล็อกโค้ดแบบ
# "().__class__.__bases__[0].__subclasses__()" ที่หา subprocess.Popen มารันคำสั่งระบบได้
# แม้ตั้ง __builtins__=None แล้วก็ตาม (ยืนยันด้วย stress test จริงมาก่อนแล้ว)
_ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Call, ast.Name, ast.Load, ast.Constant, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)

def validate_formula_syntax(formula, extra_allowed_calls=None):
    """ตรวจสอบว่าสูตรถูกไวยากรณ์ และไม่มีโครงสร้างอันตราย (attribute access ฯลฯ) ก่อนใช้งานจริง
    คืนค่า (True, None) ถ้าผ่าน หรือ (False, ข้อความ error) ถ้าไม่ผ่าน"""
    formula_str = str(formula).replace('^', '**')
    try:
        tree = ast.parse(formula_str, mode='eval')
    except SyntaxError as e:
        return False, f"สูตรผิดไวยากรณ์ (Syntax Error): {e.msg} — ตำแหน่งประมาณ {e.text.strip() if e.text else ''}"
    except Exception as e:
        return False, f"สูตรมีปัญหา: {e}"

    allowed_calls = set(MATH_FUNCS.keys()) | (set(extra_allowed_calls) if extra_allowed_calls else set())
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            return False, f"ไม่อนุญาตให้ใช้ {type(node).__name__} ในสูตร (เพื่อความปลอดภัย)"
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return False, "เรียกใช้ฟังก์ชันได้เฉพาะชื่อฟังก์ชันตรงๆ เท่านั้น"
            if node.func.id not in allowed_calls:
                return False, f"ไม่รู้จักฟังก์ชัน `{node.func.id}` — ใช้ได้เฉพาะฟังก์ชันที่รองรับเท่านั้น"

    try:
        compile(formula_str, '<string>', 'eval')
    except Exception as e:
        return False, f"สูตรมีปัญหา: {e}"
    return True, None

def build_lookup_functions(timeseries_inputs):
    """สร้างฟังก์ชัน interpolation จาก time series ที่โหลดไว้ (เช่น T_obs(t), Ia(t))"""
    lookups = {}
    for name, data in timeseries_inputs.items():
        t_arr = np.array(data["time"], dtype=float)
        v_arr = np.array(data["value"], dtype=float)
        order = np.argsort(t_arr)
        t_arr, v_arr = t_arr[order], v_arr[order]
        kind = data.get("kind", "linear")
        try:
            f = interp1d(t_arr, v_arr, kind=kind, bounds_error=False,
                         fill_value=(v_arr[0], v_arr[-1]))
        except ValueError:
            # ⚡ กันกรณี cubic แต่ข้อมูลไม่ถึง 4 จุด — fallback เป็น linear แทนไม่ให้แอปพัง
            f = interp1d(t_arr, v_arr, kind="linear", bounds_error=False, fill_value=(v_arr[0], v_arr[-1]))
        lookups[name] = (lambda fn: (lambda tt: float(fn(tt))))(f)
    return lookups

def build_base_context(parameters, timeseries_inputs):
    ctx = dict(MATH_FUNCS)
    ctx.update(build_lookup_functions(timeseries_inputs))
    ctx.update(parameters)
    return ctx

# ==========================================
# 3. RK4 Engine Core (Explosion-Proof)
# ==========================================
def compute_derivatives(t, stocks, compiled_flows, edges, base_context):
    flow_rates = {}
    eval_context = base_context.copy()
    eval_context["t"] = t
    eval_context.update(stocks)

    for f_name, compiled_code in compiled_flows.items():
        try:
            rate = eval(compiled_code, {"__builtins__": None, "np": np}, eval_context)
            if np.isnan(rate) or np.isinf(rate):
                rate = 0.0
        except:
            rate = 0.0
        flow_rates[f_name] = float(rate)

    derivatives = {s_name: 0.0 for s_name in stocks.keys()}
    for edge in edges:
        if edge["type"] == "inflow" and edge["to"] in derivatives:
            derivatives[edge["to"]] += flow_rates.get(edge["from"], 0.0)
        if edge["type"] == "outflow" and edge["from"] in derivatives:
            derivatives[edge["from"]] -= flow_rates.get(edge["to"], 0.0)
    return derivatives

def run_rk4_simulation(model_data, custom_params, custom_stocks, time_grid, timeseries=None):
    dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0
    steps = len(time_grid)

    current_stocks = {name: float(val) for name, val in custom_stocks.items()}
    history = {name: [val] for name, val in current_stocks.items()}

    compiled_flows = {}
    for f_name, flow_data in model_data.get("flows", {}).items():
        try:
            formula_str = flow_data["formula"].replace("^", "**")
            compiled_flows[f_name] = compile(formula_str, "<string>", "eval")
        except SyntaxError:
            compiled_flows[f_name] = compile("0.0", "<string>", "eval")

    # ⚡ ใช้ build_base_context เดียวกับ v7: มี min/max/abs/round/mod/if_then_else
    # ครบ และรองรับ CSV time-series lookup (เช่น T_obs(t)) ที่โมเดลปลาต้องใช้
    base_context = build_base_context(custom_params, timeseries or {})

    for step in range(steps - 1):
        current_t = time_grid[step]
        k1 = compute_derivatives(current_t, current_stocks, compiled_flows, model_data.get("edges", []), base_context)
        
        stocks_k2 = {s: current_stocks[s] + 0.5 * dt * k1[s] for s in current_stocks}
        k2 = compute_derivatives(current_t + 0.5 * dt, stocks_k2, compiled_flows, model_data.get("edges", []), base_context)
        
        stocks_k3 = {s: current_stocks[s] + 0.5 * dt * k2[s] for s in current_stocks}
        k3 = compute_derivatives(current_t + 0.5 * dt, stocks_k3, compiled_flows, model_data.get("edges", []), base_context)
        
        stocks_k4 = {s: current_stocks[s] + dt * k3[s] for s in current_stocks}
        k4 = compute_derivatives(current_t + dt, stocks_k4, compiled_flows, model_data.get("edges", []), base_context)

        for s in current_stocks.keys():
            next_val = current_stocks[s] + (dt / 6.0) * (k1[s] + 2.0 * k2[s] + 2.0 * k3[s] + k4[s])
            # Safeguard: ป้องกันกราฟระเบิดจากสมการที่ตั้งค่าผิด
            if np.isnan(next_val) or np.isinf(next_val) or abs(next_val) > 1e12:
                next_val = 0.0 
            current_stocks[s] = next_val
            history[s].append(next_val)

    return history

# ==========================================
# 4. Error Calculation (NaN-Safe)
# ==========================================
def calculate_normalized_rmse(df_data, time_col, mapping, sim_history, time_grid):
    total_norm_rmse = 0.0
    mapped_count = 0

    for s_name, csv_col in mapping.items():
        if csv_col and csv_col != "-- ไม่ใช้งาน --" and csv_col in df_data.columns:
            # กรองเอาเฉพาะแถวที่มีทั้ง เวลา และ ข้อมูลจริง เท่านั้น
            valid_idx = df_data[csv_col].notna() & df_data[time_col].notna()
            actual = df_data.loc[valid_idx, csv_col].values
            t_values = df_data.loc[valid_idx, time_col].values

            if len(actual) == 0:
                continue

            simulated = np.interp(t_values, time_grid, sim_history.get(s_name, np.zeros_like(time_grid)))
            
            rmse = np.sqrt(np.mean((actual - simulated) ** 2))
            std_actual = np.std(actual) if np.std(actual) > 1e-6 else 1.0
            norm_rmse = rmse / std_actual

            total_norm_rmse += norm_rmse
            mapped_count += 1

    return (total_norm_rmse / mapped_count) if mapped_count > 0 else 999.0

# ==========================================
# 5. Sidebar & File Uploaders
# ==========================================
st.sidebar.header("📁 โหลดไฟล์ข้อมูล & โมเดล")

if st.sidebar.button("🗑️ ล้างข้อมูล / เริ่มต้นใหม่ (Clear All)", type="primary", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")

# ==========================================
# ⚡ ใหม่: Section 0 — โหลด Time Series Input (ตัวแปรภายนอกในสูตร เช่น T_obs(t))
# แยกจาก "1. โหลดข้อมูลจริง (CSV)" ด้านล่าง ซึ่งเป็นข้อมูลเป้าหมายที่ใช้เทียบ/คำนวณ RMSE
# ต้องอัปโหลดก่อนโหลด Model JSON ถ้าสูตรในโมเดลมีการอ้างอิงตัวแปรพวกนี้ (เช่น Ia(t), T_obs(t))
# ==========================================
with st.sidebar.expander("0. โหลด Time Series Input (ตัวแปรภายนอกในสูตร)", expanded=False):
    st.caption("สำหรับตัวแปรที่ถูกเรียกในสูตรแบบ `T_obs(t)`, `Ia(t)` ฯลฯ — อัปโหลดทีละไฟล์ต่อ 1 ตัวแปร")
    ts_csv_file = st.file_uploader("เลือกไฟล์ CSV", type=["csv"], key="ts_csv_uploader_v8")

    if ts_csv_file is not None:
        try:
            df_ts_raw = pd.read_csv(ts_csv_file)
            cols = list(df_ts_raw.columns)
            ts_time_col = st.selectbox("คอลัมน์เวลา (time)", cols, index=0, key="ts_time_col_v8")
            ts_val_col = st.selectbox("คอลัมน์ค่าตัวแปร (value)", cols, index=min(1, len(cols) - 1), key="ts_val_col_v8")
            ts_name_input = st.text_input("ตั้งชื่อตัวแปร (เช่น T_obs)", value="", key="ts_name_input_v8").strip()
            ts_kind = st.radio("วิธี Interpolation", ["linear", "cubic"], horizontal=True, key="ts_kind_v8")

            if st.button("✅ สร้าง Lookup Function", use_container_width=True, key="ts_create_btn_v8") and ts_name_input:
                ts_name_clean = re.sub(r'\W', '_', ts_name_input)
                if ts_name_clean in RESERVED_NAMES:
                    st.error(f"⚠️ '{ts_name_clean}' เป็นชื่อสงวน (ฟังก์ชันคณิตศาสตร์)")
                else:
                    t_vals = pd.to_numeric(df_ts_raw[ts_time_col], errors="coerce")
                    v_vals = pd.to_numeric(df_ts_raw[ts_val_col], errors="coerce")
                    valid_mask = t_vals.notna() & v_vals.notna()
                    t_clean = t_vals[valid_mask].tolist()
                    v_clean = v_vals[valid_mask].tolist()
                    if len(t_clean) < 2:
                        st.error("⚠️ ต้องมีข้อมูลอย่างน้อย 2 จุดที่เป็นตัวเลขถูกต้อง")
                    elif ts_kind == "cubic" and len(t_clean) < 4:
                        st.error(f"⚠️ ข้อมูลมีแค่ {len(t_clean)} จุด แต่ cubic ต้องการอย่างน้อย 4 จุด — เลือก linear แทน")
                    else:
                        st.session_state.timeseries[ts_name_clean] = {"time": t_clean, "value": v_clean, "kind": ts_kind}
                        st.success(f"เพิ่ม `{ts_name_clean}(t)` สำเร็จ!")
                        st.rerun()
        except Exception as e:
            st.error(f"⚠️ อ่านไฟล์ไม่สำเร็จ: {e}")

    if st.session_state.timeseries:
        st.markdown("**ตัวแปรที่ใช้งานอยู่:**")
        for ts_n, ts_d in st.session_state.timeseries.items():
            c1, c2 = st.columns([3, 1])
            c1.write(f"`{ts_n}(t)` — {len(ts_d['time'])} จุด • {ts_d['kind']}")
            if c2.button("ลบ", key=f"del_ts_v8_{ts_n}"):
                del st.session_state.timeseries[ts_n]
                st.rerun()

st.sidebar.markdown("---")

uploaded_csv = st.sidebar.file_uploader("1. โหลดข้อมูลจริง (CSV)", type=["csv"])
if uploaded_csv is not None:
    try:
        df_temp = pd.read_csv(uploaded_csv)
        if df_temp.empty:
            st.sidebar.error("ไฟล์ CSV ว่างเปล่า!")
        else:
            # Auto-Clean: บังคับแปลงข้อความแปลกๆ เป็นตัวเลข (ถ้าแปลงไม่ได้จะเป็น NaN)
            df_temp = df_temp.apply(pd.to_numeric, errors='coerce')
            st.session_state.df_data = df_temp
            st.sidebar.success("อัปโหลดและเตรียมข้อมูล CSV สำเร็จ!")
    except Exception as e:
        st.sidebar.error(f"อ่านไฟล์ CSV ไม่สำเร็จ: {e}")

uploaded_json = st.sidebar.file_uploader("2. โหลดโมเดล (JSON)", type=["json"])
if uploaded_json is not None:
    try:
        model_data = json.load(uploaded_json)
        if "stocks" not in model_data or "parameters" not in model_data:
            st.sidebar.warning("ไฟล์ JSON ขาดคีย์ที่จำเป็น ('stocks', 'parameters')")
            st.session_state.raw_model_json = None
        else:
            # ⚡ เก็บไฟล์ที่ parse ได้ไว้ก่อนเสมอ (ไม่ว่าจะปลอดภัย/ครบ Time Series หรือยัง)
            # เพื่อให้ panel ตรวจสอบความเข้ากันได้ด้านล่างแสดงผลได้ทันที และ "เช็คซ้ำอัตโนมัติ"
            # ทุกรอบที่หน้าเว็บ rerun โดยไม่ต้องอัปโหลดไฟล์ JSON ซ้ำ หลังอัปโหลด Time Series ที่ขาดเพิ่ม
            st.session_state.raw_model_json = model_data
    except Exception as e:
        st.sidebar.error(f"อ่านไฟล์ JSON ไม่สำเร็จ: {e}")
        st.session_state.raw_model_json = None

# ==========================================
# ⚡ ใหม่: ตรวจสอบความเข้ากันได้ของโมเดล vs Time Series อัตโนมัติ (เช็คซ้ำได้ทุกรอบ rerun)
# เห็นทันทีว่าขาด Time Series ตัวไหน โดยไม่ต้องรอ Phase 2 และไม่ต้องอัปโหลด JSON ซ้ำ
# ==========================================
if st.session_state.get("raw_model_json") is not None:
    with st.expander("🔍 ตรวจสอบความเข้ากันได้ของโมเดล (Compatibility Check)", expanded=True):
        m = st.session_state.raw_model_json
        stocks_names = set(m.get("stocks", {}).keys())
        params_names = set(m.get("parameters", {}).keys())
        ts_names = set(st.session_state.timeseries.keys())

        cc1, cc2 = st.columns(2)
        cc1.markdown(f"**📦 Stocks ({len(stocks_names)}):** " + (", ".join(f"`{s}`" for s in sorted(stocks_names)) or "-"))
        cc2.markdown(f"**⚪ Parameters ({len(params_names)}):** " + (", ".join(f"`{p}`" for p in sorted(params_names)) or "-"))

        # ไล่หาทุก token ในทุกสูตรที่ "ไม่ใช่" stock/parameter/ฟังก์ชันคณิตศาสตร์ที่รู้จัก
        # เพื่อระบุว่าตัวไหนคือ CSV time-series ที่สูตรต้องการ
        all_referenced = set()
        for f_name, f_info in m.get("flows", {}).items():
            tokens = set(re.findall(r'\b[a-zA-Z_]\w*\b', f_info.get("formula", "")))
            all_referenced |= (tokens - stocks_names - params_names - RESERVED_NAMES - {"np"})

        matched_ts = sorted(all_referenced & ts_names)
        missing_ts = sorted(all_referenced - ts_names)

        if matched_ts:
            st.success("✅ Time Series ที่โหลดไว้ตรงกับสูตรครบ: " + ", ".join(f"`{t}(t)`" for t in matched_ts))
        if missing_ts:
            st.error("❌ สูตรอ้างอิงตัวแปรเหล่านี้ แต่ยังไม่ได้อัปโหลด Time Series: " + ", ".join(f"`{t}`" for t in missing_ts))
            st.caption("👉 ไปที่เมนู '0. โหลด Time Series Input' ในแถบซ้าย อัปโหลด CSV แล้วตั้งชื่อตัวแปรให้ตรงกับที่แจ้งด้านบน — พอครบแล้วหน้านี้จะเช็คซ้ำให้เองอัตโนมัติ ไม่ต้องอัปโหลด JSON ใหม่")
        if not matched_ts and not missing_ts:
            st.info("โมเดลนี้ไม่ได้ใช้ CSV Time Series เลย (ใช้แค่ stock/parameter ปกติ)")

        # ⚡ เช็คความปลอดภัย/ไวยากรณ์ของสูตรทุกครั้งที่ rerun (ใช้ ts_names ปัจจุบัน)
        formula_errors = []
        for f_name, f_info in m.get("flows", {}).items():
            ok, err = validate_formula_syntax(f_info.get("formula", ""), extra_allowed_calls=ts_names)
            if not ok:
                formula_errors.append(f"`{f_name}`: {err}")

        if formula_errors:
            st.error("⚠️ ยังโหลดโมเดลนี้ไม่ได้ เพราะพบสูตรที่ไม่ปลอดภัยหรือผิดไวยากรณ์:")
            for e in formula_errors:
                st.write(f"- {e}")
            st.session_state.model_json = None
        elif st.session_state.df_data is None:
            st.session_state.model_json = m  # ปลอดภัยแล้ว พร้อมใช้งาน แต่ยังขาดข้อมูลจริง
            st.warning("⚠️ โมเดลพร้อมแล้ว แต่ยังไม่พบไฟล์ข้อมูลจริง (CSV) — อัปโหลดที่ '1. โหลดข้อมูลจริง (CSV)' เพื่อให้ Phase 2 แสดงผล")
        else:
            st.session_state.model_json = m
            st.success("✅ พร้อมสำหรับ Phase 2 แล้ว — เลื่อนลงไปดูด้านล่างได้เลย")

# ==========================================
# 6. Phase 1: Data Exploration
# ==========================================
if st.session_state.df_data is not None:
    st.subheader("📊 Phase 1: สำรวจข้อมูลจริง (Data Explorer)")
    df = st.session_state.df_data

    # เช็คว่ามีค่า NaN เกิดขึ้นหรือไม่ (จากการ Clean Data)
    nan_counts = df.isna().sum()
    invalid_cols = nan_counts[nan_counts > 0]
    
    if not invalid_cols.empty:
        st.warning("⚠️ **ตรวจพบข้อมูลไม่สมบูรณ์ (หรือข้อความขยะ) ในไฟล์ CSV:** ระบบได้ทำการแปลงข้อมูลดังกล่าวเป็นค่าว่าง (NaN) เพื่อให้วิเคราะห์ต่อได้")
        for col, count in invalid_cols.items():
            st.write(f"- คอลัมน์ `{col}`: พบค่าว่างจำนวน **{count}** แถว")
        st.markdown("---")

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        time_col = st.selectbox("เลือกคอลัมน์ที่เป็น เวลา (Time / X-axis):", df.columns)
        
        # ป้องกันคอลัมน์เวลาว่างเปล่าล้วน
        if df[time_col].dropna().empty:
            st.error(f"⚠️ คอลัมน์ '{time_col}' ไม่มีข้อมูลตัวเลขเลย! กรุณาเลือกคอลัมน์อื่น")
            st.stop()

    with col_t2:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        possible_y = [c for c in numeric_cols if c != time_col]
        selected_cols = st.multiselect(
            "เลือกคอลัมน์ที่ต้องการดู Scatter Plot & สถิติ:",
            possible_y, default=possible_y[:2] if possible_y else []
        )

    if selected_cols:
        col_fig, col_stats = st.columns([3, 2])
        with col_fig:
            fig_exp = go.Figure()
            for col in selected_cols:
                # Plotly จะข้ามจุดที่เป็น NaN อัตโนมัติเวลาแสดง Markers
                fig_exp.add_trace(go.Scatter(x=df[time_col], y=df[col], mode="markers+lines", name=col, connectgaps=False))
            fig_exp.update_layout(xaxis_title=time_col, yaxis_title="Value", height=300, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_exp, use_container_width=True)

        with col_stats:
            st.markdown("**📈 Descriptive Stats**")
            stats_df = df[selected_cols].agg(["min", "max", "mean", "std"]).T.reset_index()
            stats_df.columns = ["Column", "Min", "Max", "Average", "Std Dev"]
            st.dataframe(stats_df.style.format({"Min": "{:.2f}", "Max": "{:.2f}", "Average": "{:.2f}", "Std Dev": "{:.2f}"}), use_container_width=True, hide_index=True)
    st.markdown("---")

# ==========================================
# 7. Phase 2: Model Calibration
# ==========================================
if st.session_state.df_data is not None and st.session_state.model_json is not None:
    st.subheader("⚙️ Phase 2: ปรับจูนโมเดลเข้ากับข้อมูล (Model Calibration)")
    model = st.session_state.model_json
    df = st.session_state.df_data

# --- Step 2.1: Mapping & Equations ---
    with st.expander("🔗 Step 2.1: จับคู่ตัวแปร และแสดงสมการโมเดล", expanded=True):
        map_cols = st.columns(min(len(model.get("stocks", {})), 4))
        csv_options = ["-- ไม่ใช้งาน --"] + list(df.columns)

        for i, (s_name, s_init) in enumerate(model.get("stocks", {}).items()):
            target_col = map_cols[i % max(1, len(map_cols))]
            with target_col:
                selected_map = st.selectbox(f"📦 Stock `{s_name}` ⟷ CSV:", csv_options, key=f"map_{s_name}")
                if selected_map != "-- ไม่ใช้งาน --":
                    st.session_state.mapping[s_name] = selected_map
                elif s_name in st.session_state.mapping:
                    del st.session_state.mapping[s_name]

        st.markdown("---")
        st.markdown("**📐 สมการของโมเดล (Model Equations):**")
        eq_col1, eq_col2 = st.columns(2)
        
        with eq_col1:
            st.markdown("*สูตรอัตราการเพิ่ม/ลด (Formulae):*")
            for f_name, f_info in model.get("flows", {}).items():
                # เปลี่ยนมาใช้ st.code หรือ Markdown ธรรมดา ป้องกันบั๊ก LaTeX
                st.markdown(f"`{f_name} = {f_info['formula']}`")

        with eq_col2:
            st.markdown("*สมการเชิงอนุพันธ์ (Differential Equations):*")
            for s_name in model.get("stocks", {}):
                # ดึง Inflow (เข้า)
                inflows = [e["from"] for e in model.get("edges", []) if e.get("to") == s_name and e.get("type") == "inflow"]
                
                # 🐞 แก้ Bug แล้ว: ดึง Outflow (ออก) ต้องใช้ e["to"] เพื่อระบุชื่อ Flow ปลายทาง
                outflows = [e["to"] for e in model.get("edges", []) if e.get("from") == s_name and e.get("type") == "outflow"]
                
                rhs_terms = []
                if inflows: 
                    rhs_terms.append(" + ".join(inflows))
                if outflows: 
                    rhs_terms.append("- (" + " + ".join(outflows) + ")")
                
                rhs = " ".join(rhs_terms) if rhs_terms else "0"
                
                # แสดงผลเป็นข้อความธรรมดา
                st.markdown(f"`d({s_name})/dt = {rhs}`")
                
    # --- Step 2.2: Manual Range Finder ---
    st.markdown("### 🎛️ Step 2.2: ปรับแต่ง Bounds และดูผลลัพธ์ (Manual Tune)")
    col_sliders, col_plots = st.columns([2, 3])
    tuned_params = {}
    tuned_stocks = {}
    param_bounds_dict = {}
    fit_flags = {}  # ⚡ ใหม่: True = เอาเข้า Auto-Fit, False = ตรึงค่าไว้คงที่ (ไม่ปรับ)

    with col_sliders:
        st.markdown("##### ⚪ กำหนดขอบเขต Parameter")
        st.caption("☑️ เลือก 'Fit' เฉพาะ parameter ที่ไม่แน่นอน/ไม่ทราบค่าแน่ชัด — ค่าคงที่ทางชีวภาพ/กายภาพ "
                    "เช่น อุณหภูมิขีดจำกัด, เลขชี้กำลัง ไม่ควรเลือก เพราะ optimizer อาจบิดเบือนค่าที่มี "
                    "และอาจเกิด overfitting")

        # ⚡ หัวคอลัมน์ "Fit / Min / Max" แสดงครั้งเดียวด้านบน แทนที่จะให้ checkbox แต่ละแถว
        # มี label "Fit" ซ้ำๆ กัน (ซึ่งทำให้ตัวหนังสือตกบรรทัดในคอลัมน์แคบๆ)
        h_fit, h_min, h_max = st.columns([1, 2, 2])
        h_fit.markdown("**Fit**")
        h_min.markdown("**Min**")
        h_max.markdown("**Max**")

        for p_name, p_val in model.get("parameters", {}).items():
            fit_c, sub_c1, sub_c2 = st.columns([1, 2, 2])
            fit_flags[p_name] = fit_c.checkbox(
                "Fit", value=False, key=f"fit_flag_{p_name}",
                label_visibility="collapsed",
                help=f"รวม `{p_name}` เข้า Auto-Fit หรือไม่"
            )
            p_min = sub_c1.number_input(f"Min `{p_name}`", value=0.0, format="%.4f", key=f"b_min_{p_name}")
            p_max = sub_c2.number_input(f"Max `{p_name}`", value=max(1.0, float(p_val) * 2.0), format="%.4f", key=f"b_max_{p_name}")

            if p_min >= p_max: p_max = p_min + 0.001
            param_bounds_dict[p_name] = (p_min, p_max)
            step_val = 0.0001 if (p_max - p_min) <= 1.0 else (0.01 if (p_max - p_min) <= 10.0 else 1.0)
            default_val = float(np.clip(p_val, p_min, p_max))

            if fit_flags[p_name]:
                tuned_params[p_name] = st.slider(f"`{p_name}`", min_value=float(p_min), max_value=float(p_max), value=default_val, step=step_val, format="%.4f", key=f"slider_p_{p_name}")
            else:
                # ⚡ parameter ที่ไม่ fit แสดงเป็นค่าคงที่ (แก้ไขได้แต่ไม่ใช่ slider ที่จะสื่อว่า optimizer ปรับได้)
                tuned_params[p_name] = st.number_input(f"`{p_name}` (คงที่ — ไม่ fit)", value=float(p_val), format="%.4f", key=f"fixed_val_{p_name}")

        st.markdown("##### 📦 ปรับค่าเริ่มต้น Initial Stocks")
        for s_name, s_init in model.get("stocks", {}).items():
            # 1. คำนวณค่า Min/Max เบื้องต้นอ้างอิงจากข้อมูล (Smart Default)
            if s_name in st.session_state.mapping:
                mapped_col = st.session_state.mapping[s_name]
                clean_s_data = df[mapped_col].dropna()
                default_s_min = float(clean_s_data.min()) if not clean_s_data.empty else 0.0
                default_s_max = float(clean_s_data.max() * 1.5) if not clean_s_data.empty else max(100.0, float(s_init) * 2.0)
            else:
                default_s_min, default_s_max = 0.0, max(100.0, float(s_init) * 2.0)

            if default_s_min >= default_s_max: 
                default_s_max = default_s_min + 10.0

            # 2. สร้างช่องกรอก Min / Max ให้ผู้ใช้ปรับแต่งเองได้
            sub_c1, sub_c2 = st.columns(2)
            s_min_val = sub_c1.number_input(f"Min `{s_name}`", value=default_s_min, format="%.2f", key=f"b_min_s_{s_name}")
            s_max_val = sub_c2.number_input(f"Max `{s_name}`", value=default_s_max, format="%.2f", key=f"b_max_s_{s_name}")

            # ป้องกัน Error กรณีผู้ใช้ตั้งค่า Min มากกว่า Max
            if s_min_val >= s_max_val: 
                s_max_val = s_min_val + 0.01

            # 3. กำหนดค่าเริ่มต้นและ Step ของ Slider
            s_val = float(np.clip(float(s_init), s_min_val, s_max_val))
            step_val = 0.01 if (s_max_val - s_min_val) <= 10.0 else 1.0

            # 4. วาด Slider
            tuned_stocks[s_name] = st.slider(
                f"Initial `{s_name}`", 
                min_value=float(s_min_val), 
                max_value=float(s_max_val), 
                value=s_val, 
                step=step_val, 
                key=f"slider_s_{s_name}"
            )

    # Calculate Time Grid Safely (Ignore NaNs in time_col)
    clean_time = df[time_col].dropna()
    t_min = float(clean_time.min()) if not clean_time.empty else 0.0
    t_max = float(clean_time.max()) if not clean_time.empty else 10.0
    
    # Cap steps to prevent memory freeze
    num_steps = min(len(clean_time) * 5 if not clean_time.empty else 50, 2000)
    num_steps = max(num_steps, 50)
    time_grid = np.linspace(t_min, t_max, num_steps)

    manual_history = run_rk4_simulation(model, tuned_params, tuned_stocks, time_grid, st.session_state.timeseries)
    current_norm_rmse = calculate_normalized_rmse(df, time_col, st.session_state.mapping, manual_history, time_grid)

    with col_plots:
        st.markdown(f"##### 📈 กราฟเปรียบเทียบ (Norm RMSE: `{current_norm_rmse:.4f}`)")
        if len(st.session_state.mapping) == 0:
            st.warning("⚠️ โปรดจับคู่ตัวแปรใน Step 2.1 เพื่อคำนวณ Error")

        all_stocks = list(model.get("stocks", {}).keys())
        if all_stocks:
            fig_sub = make_subplots(rows=len(all_stocks), cols=1, shared_xaxes=True, subplot_titles=[f"Stock: {s}" for s in all_stocks])

            for i, s_name in enumerate(all_stocks):
                fig_sub.add_trace(go.Scatter(x=time_grid, y=manual_history[s_name], mode="lines", name=f"{s_name} (Manual)", line=dict(color='orange')), row=i + 1, col=1)
                if s_name in st.session_state.mapping:
                    csv_col = st.session_state.mapping[s_name]
                    # กรองเอาเฉพาะข้อมูลที่ไม่ใช่ NaN มาพล็อต
                    valid_data = df[[time_col, csv_col]].dropna()
                    fig_sub.add_trace(go.Scatter(x=valid_data[time_col], y=valid_data[csv_col], mode="markers", name=f"{csv_col} (Data)", marker=dict(size=7, symbol="circle", color='blue')), row=i + 1, col=1)

            fig_sub.update_layout(height=max(250, 250 * len(all_stocks)), margin=dict(l=10, r=10, t=30, b=10), showlegend=True)
            st.plotly_chart(fig_sub, use_container_width=True)

    # --- Step 2.3: Auto Optimization ---
    st.markdown("---")
    st.markdown("### ⚡ Step 2.3: Auto Optimization Toolbox")
    col_opt1, col_opt2 = st.columns([2, 3])
    with col_opt1:
        opt_alg = st.selectbox("เลือกอัลกอริทึม Auto-Fit:", ["L-BFGS-B (Fast/Default)", "SLSQP (Constrained)", "Differential Evolution (Global Search)"])

    with col_opt2:
        st.write("")
        st.write("")
        if st.button("🚀 เริ่มต้นรัน Auto-Fit (Optimize)", type="primary", use_container_width=True):
            fit_param_names = [p for p, flag in fit_flags.items() if flag]
            fixed_param_values = {p: v for p, v in tuned_params.items() if p not in fit_param_names}

            if len(st.session_state.mapping) == 0:
                st.error("❌ โปรดจับคู่ตัวแปรใน Step 2.1 ก่อน")
            elif len(fit_param_names) == 0:
                st.error("❌ โปรดติ๊ก '☑️ Fit' อย่างน้อย 1 parameter ก่อนรัน Auto-Fit (parameter ที่ไม่ติ๊กจะถูกตรึงไว้คงที่)")
            else:
                param_names = fit_param_names
                initial_guess = [tuned_params[p] for p in param_names]
                bounds_list = [param_bounds_dict[p] for p in param_names]

                def objective_func(x):
                    curr_p = dict(zip(param_names, x))
                    curr_p.update(fixed_param_values)  # ⚡ เติม parameter ที่ตรึงไว้กลับเข้าไปทุกครั้งที่ประเมินผล
                    sim_h = run_rk4_simulation(model, curr_p, tuned_stocks, time_grid, st.session_state.timeseries)
                    return calculate_normalized_rmse(df, time_col, st.session_state.mapping, sim_h, time_grid)

                with st.spinner(f"🤖 กำลังคำนวณปรับจูน {len(fit_param_names)} parameter ({', '.join(fit_param_names)})... (อาจใช้เวลาสักครู่)"):
                    try:
                        if "L-BFGS-B" in opt_alg:
                            res = minimize(objective_func, x0=initial_guess, method="L-BFGS-B", bounds=bounds_list)
                        elif "SLSQP" in opt_alg:
                            res = minimize(objective_func, x0=initial_guess, method="SLSQP", bounds=bounds_list)
                        else:
                            res = differential_evolution(objective_func, bounds=bounds_list, seed=42)
                        
                        best_params = dict(zip(param_names, res.x))
                        best_params.update(fixed_param_values)  # ⚡ ให้ export ผลลัพธ์ได้ครบทุก parameter ไม่ใช่แค่ตัวที่ fit
                        st.session_state.opt_results = {"best_params": best_params, "best_stocks": tuned_stocks, "final_score": res.fun, "fitted_names": fit_param_names}
                        st.rerun()
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดขณะปรับจูน: {e}")

    # --- Results Rendering ---
    if st.session_state.opt_results is not None:
        st.success(f"✨ ปรับจูนสำเร็จ! Final Norm RMSE: `{st.session_state.opt_results['final_score']:.4f}`")
        opt_p = st.session_state.opt_results["best_params"]
        opt_s = st.session_state.opt_results["best_stocks"]
        opt_history = run_rk4_simulation(model, opt_p, opt_s, time_grid, st.session_state.timeseries)
        
        fig_opt = make_subplots(rows=len(all_stocks), cols=1, shared_xaxes=True, subplot_titles=[f"Stock: {s}" for s in all_stocks])
        for i, s_name in enumerate(all_stocks):
            fig_opt.add_trace(go.Scatter(x=time_grid, y=opt_history[s_name], mode="lines", name=f"{s_name} (Optimized)", line=dict(color='green', width=3)), row=i + 1, col=1)
            if s_name in st.session_state.mapping:
                csv_col = st.session_state.mapping[s_name]
                valid_data = df[[time_col, csv_col]].dropna()
                fig_opt.add_trace(go.Scatter(x=valid_data[time_col], y=valid_data[csv_col], mode="markers", name=f"{csv_col} (Data)", marker=dict(size=7, symbol="circle", color='blue')), row=i + 1, col=1)
        
        fig_opt.update_layout(height=max(250, 250 * len(all_stocks)), margin=dict(l=10, r=10, t=30, b=10), showlegend=True)
        st.plotly_chart(fig_opt, use_container_width=True)

        fitted_names = st.session_state.opt_results.get("fitted_names", list(opt_p.keys()))
        st.markdown("##### 🎯 ค่าพารามิเตอร์ที่เหมาะสมที่สุด (Optimal Values):")
        st.caption(f"🟢 Fit แล้ว: {', '.join(fitted_names) if fitted_names else '-'}  •  ⚪ ตรึงคงที่: {', '.join(p for p in opt_p if p not in fitted_names) or '-'}")
        m_cols = st.columns(max(1, min(len(opt_p), 4)))
        for i, (p_name, p_val) in enumerate(opt_p.items()):
            with m_cols[i % len(m_cols)]:
                p_init = model.get("parameters", {}).get(p_name, 0.0)
                tag = "🟢 Fit" if p_name in fitted_names else "⚪ คงที่"
                st.metric(label=f"{tag} `{p_name}`", value=f"{p_val:.4f}", delta=f"{p_val - p_init:+.4f} จากเดิม" if p_name in fitted_names else None)

    # --- Step 2.4: Export ---
    if st.session_state.opt_results is not None:
        st.markdown("---")
        st.markdown("### 📥 Step 2.4: ดาวน์โหลดผลลัพธ์")
        
        import re # นำเข้าไลบรารีสำหรับค้นหาชื่อตัวแปรในสูตร
        
        opt_p = st.session_state.opt_results["best_params"]
        opt_s = st.session_state.opt_results["best_stocks"]

        # 🧹 จัดเรียงโครงสร้าง JSON ใหม่ให้เหมือน sd_model.json
        calibrated_json = {
            "stocks": opt_s,
            "parameters": opt_p,
            "flows": model.get("flows", {})
        }
        
        # 1. แยกเส้น Physical Flow (inflow, outflow) ของเดิมเก็บไว้
        original_edges = model.get("edges", [])
        physical_edges = [e for e in original_edges if e.get("type") in ["inflow", "outflow"]]
        
        # 2. สร้างเส้น Information Flow อัตโนมัติจากการแกะสูตรคณิตศาสตร์
        information_edges = []
        all_variables = list(opt_s.keys()) + list(opt_p.keys())
        
        for f_name, f_info in calibrated_json["flows"].items():
            formula = f_info.get("formula", "")
            # ดึงคำทั้งหมดที่เป็นตัวอักษรหรือมี underscore ออกจากสูตร
            words_in_formula = set(re.findall(r'[a-zA-Z_]\w*', formula))
            
            for word in words_in_formula:
                if word in all_variables:
                    information_edges.append({
                        "from": word,
                        "to": f_name,
                        "type": "information"
                    })
        
        # 3. ประกอบรวมกัน โดยให้เส้น information ขึ้นก่อนตามรูปแบบ sd_model.json
        calibrated_json["edges"] = information_edges + physical_edges

        # แปลงเป็นไฟล์ดาวน์โหลด
        json_bytes = json.dumps(calibrated_json, indent=4).encode("utf-8")

        comp_df = pd.DataFrame({"Time": df[time_col]})
        opt_history = run_rk4_simulation(model, opt_p, opt_s, time_grid, st.session_state.timeseries)
        for s_name, csv_col in st.session_state.mapping.items():
            if csv_col and csv_col != "-- ไม่ใช้งาน --":
                comp_df[f"{s_name}_Actual"] = df[csv_col]
                comp_df[f"{s_name}_Model"] = np.interp(df[time_col], time_grid, opt_history[s_name])
        csv_bytes = comp_df.to_csv(index=False).encode("utf-8")

        txt = f"SD Model Calibration Report\nFinal Norm RMSE: {st.session_state.opt_results['final_score']:.6f}\n\n[Optimal Parameters]\n"
        for p, v in opt_p.items(): txt += f"- {p}: {v:.6f}\n"
        txt_bytes = txt.encode("utf-8")

        c1, c2, c3 = st.columns(3)
        c1.download_button("📄 โหลด JSON (สำหรับ sys-sim)", data=json_bytes, file_name="calibrated_model.json", mime="application/json", use_container_width=True)
        c2.download_button("📊 โหลด CSV (เทียบผล Model vs Data)", data=csv_bytes, file_name="fit_data.csv", mime="text/csv", use_container_width=True)
        c3.download_button("📜 โหลด Text Summary", data=txt_bytes, file_name="summary.txt", mime="text/plain", use_container_width=True)