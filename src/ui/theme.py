# --------------- STYLING ---------------
PALETTE = {
    "red": "#cc1f1f",
    "dark_red": "#a81818",
    "amber": "#e08a1e",
    "green": "#2f9e63",
    "page_bg": "#f1f1f1",
    "card_bg": "#ffffff",
    "card_border": "#e6e6e6",
    "subtle_border": "#f1f1f1",
    "thead_bg": "#fafafa",
    "ink": "#1a1a1a",
    "ink_soft": "#888888",
    "ink_faint": "#aaaaaa",
}
FONT_MAIN = '"Noto Sans SC", system-ui, sans-serif'
DEFAULT_MARGIN = dict(l=10, r=10, t=10, b=10)

# --------------- HELPER ---------------
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


# --------------- CSS (root vars generated from PALETTE, rest is static) ---------------
_ROOT_VARS = "\n".join(f"    --{k.replace('_', '-')}: {v};" for k, v in PALETTE.items())
_ROOT_VARS += f"\n    --font-main: {FONT_MAIN};"

_PILL_OK_BG = hex_to_rgba(PALETTE["green"], 0.12)
_PILL_WARN_BG = hex_to_rgba(PALETTE["amber"], 0.14)
_PILL_FAIL_BG = hex_to_rgba(PALETTE["red"], 0.12)

CSS = """
<style>
:root {
""" + _ROOT_VARS + """
}

html, body, [class*="css"] {
    font-family: var(--font-main);
    color: var(--ink);
}

.stApp {
    background-color: var(--page-bg);
}

[data-testid="stHeader"] {
    background-color: transparent;
}

/* ---------- App header ---------- */
.ri-header {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin-bottom: 1.25rem;
}
.ri-header .ri-title {
    font-size: 4rem;
    font-weight: 700;
    color: var(--dark-red);
}
.ri-header .ri-subtitle-en {
    font-size: 1.5rem;
    color: var(--ink-soft);
}
.ri-header .ri-subtitle-zh {
    font-size: 1rem;
    color: var(--ink-faint);
}

/* ---------- Section header ---------- */
.ri-section-header {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin: 1.6rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--card-border);
}
.ri-section-header .ri-section-en {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--ink);
}
.ri-section-header .ri-section-zh {
    font-size: 0.85rem;
    color: var(--ink-faint);
}

/* ---------- Card ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .ri-card-title) {
    background-color: var(--card-bg);
    border: 1px solid var(--card-border) !important;
    padding: 0.9rem 1rem;
    margin-bottom: 0.9rem;
}

.ri-card-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.4rem;
}

.ri-card-title .ri-card-name {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--ink);
}

.ri-card-title .ri-card-desc {
    font-size: 1rem;
    color: var(--ink-soft);
    margin-top: 1px;
}

/* ---------- Status pill ---------- */
.ri-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 999px;
    white-space: nowrap;
}
.ri-pill-ok { background-color: __PILL_OK_BG__; color: var(--green); }
.ri-pill-warn { background-color: __PILL_WARN_BG__; color: var(--amber); }
.ri-pill-fail { background-color: __PILL_FAIL_BG__; color: var(--red); }
.ri-pill-neutral { background-color: var(--thead-bg); color: var(--ink-soft); }

/* ---------- Dataframe tweaks ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--subtle-border);
    overflow: hidden;
    background-color: var(--page-bg);
}

/* ---------- Metric tweaks ---------- */
[data-testid="stMetric"] {
    background-color: var(--card-bg);
    border: 1px solid var(--card-border);
    padding: 0.6rem 0.9rem;
}

[data-testid="stMetric"] p {
    color: var(--dark-red);
}

[data-testid="stMetricLabel"] {
    color: var(--ink-soft);
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background-color: var(--dark-red);
    border-right: 1px solid var(--card-border);
    color: var(--card-bg);
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: var(--card-bg);
}

/* ---------- Sidebar: Slider (track, thumb, and value labels) ---------- */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background-color: rgba(255, 255, 255, 0.25);
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background-color: var(--card-bg);
    border-color: var(--card-bg);
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderThumbValue"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBar"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderThumbValue"] p,
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBar"] p {
    color: var(--card-bg) !important;
}

/* ---------- Sidebar: Number/Text input ---------- */
[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background-color: rgba(255, 255, 255, 0.92);
    color: var(--ink);
    border-radius: 6px;
}

/* ---------- Sidebar: Selectbox ---------- */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div {
    background-color: rgba(255, 255, 255, 0.92);
    border-radius: 6px;
}

/* ---------- Tab bar ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid var(--card-border);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: var(--ink-soft);
    font-weight: 600;
    padding: 0.6rem 1rem;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--dark-red);
    border-bottom: 3px solid var(--dark-red);
}

/* ---------- Chart container ---------- */
[data-testid="stPlotlyChart"] {
    background-color: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 0.6rem;
}

</style>
"""
CSS = (CSS
    .replace("__PILL_OK_BG__", _PILL_OK_BG)
    .replace("__PILL_WARN_BG__", _PILL_WARN_BG)
    .replace("__PILL_FAIL_BG__", _PILL_FAIL_BG)
)


def inject(st_module) -> None:
    st_module.markdown(CSS, unsafe_allow_html=True)
