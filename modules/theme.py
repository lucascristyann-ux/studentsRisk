import streamlit as st
import plotly.express as px

NAVY='#13243A'
NAVY_2='#1B3553'
TEAL='#0A807A'
TEAL_DARK='#086760'
BLUE='#2E6FB2'
MUTED='#6B7C90'
BG='#F4F7FB'
BORDER='#DCE5EF'
PALETTE=[TEAL,BLUE,'#E0A44B','#7667B2','#3998A4','#D87569']

def setup():
    px.defaults.template='plotly_white'
    px.defaults.color_discrete_sequence=PALETTE
    st.markdown('''<style>
    :root{
        --navy:#13243A;
        --navy2:#1B3553;
        --teal:#0A807A;
        --teal-dark:#086760;
        --blue:#2E6FB2;
        --muted:#6B7C90;
        --bg:#F4F7FB;
        --card:#FFFFFF;
        --border:#DCE5EF;
        --soft:#EEF4F8;
    }

    /* Main application */
    .stApp{
        background:
          radial-gradient(circle at 92% 4%, rgba(10,128,122,.06), transparent 28%),
          linear-gradient(180deg,#F8FAFD 0%,#F3F7FB 100%);
        color:var(--navy);
    }
    .block-container{
        max-width:1480px;
        padding-top:1.5rem;
        padding-bottom:4rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"]{
        background:linear-gradient(180deg,#10243B 0%,#142D48 55%,#10243B 100%)!important;
        border-right:1px solid rgba(255,255,255,.05);
    }
    [data-testid="stSidebar"] > div:first-child{
        padding-top:.6rem;
    }
    [data-testid="stSidebar"] *{
        color:#EAF2F9!important;
    }
    [data-testid="stSidebar"] hr{
        border-color:rgba(255,255,255,.14)!important;
        margin:1rem 0!important;
    }
    [data-testid="stSidebar"] .sidebar-brand{
        padding:16px 14px 12px;
        margin:0 0 10px;
        border:1px solid rgba(255,255,255,.09);
        border-radius:16px;
        background:linear-gradient(135deg,rgba(255,255,255,.08),rgba(255,255,255,.025));
        box-shadow:0 10px 28px rgba(0,0,0,.10);
    }
    [data-testid="stSidebar"] .sidebar-brand .brand-kicker{
        font-size:10px;
        letter-spacing:.17em;
        font-weight:800;
        color:#8ED8D2!important;
        margin-bottom:6px;
    }
    [data-testid="stSidebar"] .sidebar-brand .brand-title{
        font-size:20px;
        font-weight:800;
        color:#FFFFFF!important;
        line-height:1.1;
    }
    [data-testid="stSidebar"] .sidebar-brand .brand-sub{
        margin-top:6px;
        color:#BFD0DF!important;
        font-size:11px;
        line-height:1.4;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"]{
        background:rgba(255,255,255,.065);
        border:1px solid rgba(255,255,255,.09);
        border-radius:15px;
        padding:10px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section{
        background:rgba(255,255,255,.96)!important;
        border-radius:12px!important;
        border:0!important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section *{
        color:#27415E!important;
    }

    [data-testid="stSidebar"] [role="radiogroup"]{
        gap:5px;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label{
        padding:10px 11px;
        border-radius:11px;
        margin:2px 0;
        border:1px solid transparent;
        transition:all .18s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover{
        background:rgba(255,255,255,.075);
        border-color:rgba(255,255,255,.07);
        transform:translateX(2px);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
        background:linear-gradient(90deg,#176A73,#227C82)!important;
        border-color:rgba(255,255,255,.12)!important;
        box-shadow:0 8px 20px rgba(0,0,0,.12);
    }

    /* Typography */
    h1,h2,h3,h4{
        color:var(--navy)!important;
        letter-spacing:-.025em;
    }
    h1{font-size:2rem!important;font-weight:800!important}
    h2{font-size:1.5rem!important;font-weight:780!important}
    h3{font-size:1.17rem!important;font-weight:750!important}
    p,li{line-height:1.55}
    [data-testid="stCaptionContainer"]{
        color:#718196!important;
    }

    /* Hero */
    .hero{
        position:relative;
        overflow:hidden;
        padding:27px 30px;
        border-radius:22px;
        background:
          radial-gradient(circle at 86% 22%,rgba(72,183,175,.25),transparent 28%),
          linear-gradient(110deg,#13243A 0%,#18465A 58%,#1E6670 100%);
        color:#fff;
        margin:0 0 22px;
        box-shadow:0 14px 36px rgba(19,36,58,.13);
        border:1px solid rgba(255,255,255,.08);
    }
    .hero:after{
        content:'';
        position:absolute;
        width:190px;height:190px;
        border-radius:50%;
        border:1px solid rgba(255,255,255,.08);
        right:-62px;top:-84px;
    }
    .hero .eyebrow{
        font-size:11px;
        font-weight:800;
        letter-spacing:.19em;
        color:#9BE0DA;
        margin-bottom:8px;
    }
    .hero .heading{
        font-size:29px;
        font-weight:820;
        line-height:1.16;
        color:white;
    }
    .hero .sub{
        color:#D4E6EF;
        font-size:14px;
        margin-top:9px;
        max-width:760px;
    }

    /* Metrics */
    [data-testid="stMetric"]{
        position:relative;
        background:rgba(255,255,255,.96)!important;
        border:1px solid var(--border);
        border-radius:18px;
        padding:19px 20px 18px;
        box-shadow:0 7px 24px rgba(19,36,58,.055);
        min-height:118px;
        transition:transform .16s ease, box-shadow .16s ease;
    }
    [data-testid="stMetric"]:before{
        content:'';
        position:absolute;
        left:0;top:16px;bottom:16px;
        width:3px;border-radius:3px;
        background:linear-gradient(180deg,var(--teal),#50B8AF);
    }
    [data-testid="stMetric"]:hover{
        transform:translateY(-1px);
        box-shadow:0 10px 28px rgba(19,36,58,.075);
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *{
        color:#61748A!important;
        font-weight:700!important;
        opacity:1!important;
        font-size:.88rem!important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *{
        color:var(--navy)!important;
        font-size:2rem!important;
        font-weight:820!important;
        opacity:1!important;
    }

    /* Charts and dataframes */
    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"]{
        background:rgba(255,255,255,.98);
        border:1px solid var(--border);
        border-radius:18px;
        padding:10px;
        box-shadow:0 5px 20px rgba(19,36,58,.045);
    }

    /* Tabs */
    [data-testid="stTabs"] button{
        font-weight:720;
        color:#60738A!important;
    }
    [data-testid="stTabs"] button[aria-selected="true"]{
        color:var(--teal)!important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"]{
        background-color:var(--teal)!important;
    }

    /* Expanders */
    [data-testid="stExpander"]{
        border:1px solid var(--border)!important;
        border-radius:14px!important;
        background:#FFFFFF!important;
        box-shadow:0 3px 12px rgba(19,36,58,.03);
    }

    /* Buttons */
    .stButton>button{
        border-radius:10px!important;
        font-weight:720!important;
        min-height:2.65rem;
        border:1px solid #C9D7E5!important;
        transition:all .16s ease!important;
    }
    .stButton>button:hover{
        border-color:var(--teal)!important;
        color:var(--teal)!important;
        transform:translateY(-1px);
    }
    .stButton>button[kind="primary"]{
        background:linear-gradient(90deg,var(--teal-dark),var(--teal))!important;
        border-color:var(--teal)!important;
        color:white!important;
        box-shadow:0 6px 16px rgba(10,128,122,.16);
    }
    .stButton>button[kind="primary"]:hover{
        color:white!important;
        box-shadow:0 8px 20px rgba(10,128,122,.22);
    }

    /* Notices */
    [data-testid="stAlert"]{
        border-radius:14px!important;
        border-width:1px!important;
    }

    /* Inputs */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div{
        border-radius:10px!important;
        border-color:#CEDAE6!important;
    }

    /* Research section separators */
    hr{
        border-color:#DEE7F0!important;
        margin:1.7rem 0!important;
    }

    /* Download buttons */
    [data-testid="stDownloadButton"] button{
        border-radius:10px!important;
        font-weight:700!important;
    }

    @media(max-width:700px){
        .hero{padding:21px 20px;border-radius:17px}
        .hero .heading{font-size:22px}
        .block-container{padding:1rem .8rem 3rem}
        [data-testid="stMetric"]{min-height:100px}
    }
    </style>''',unsafe_allow_html=True)

def hero(title, subtitle):
    st.markdown(
        f'<div class="hero">'
        f'<div class="eyebrow">ADSSU · OSAS WELLNESS INTELLIGENCE</div>'
        f'<div class="heading">{title}</div>'
        f'<div class="sub">{subtitle}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

def polish(fig,height=420):
    fig.update_layout(
        template='plotly_white',
        height=height,
        margin=dict(l=24,r=26,t=58,b=34),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family='Arial',color=NAVY,size=12),
        title_font=dict(size=17,color=NAVY,family='Arial'),
        legend_title_text='',
        hoverlabel=dict(bgcolor='white',font_color=NAVY,bordercolor='#D7E1EC'),
        colorway=PALETTE
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#EAF0F6',
        zeroline=False,
        linecolor='#D9E3ED',
        tickfont=dict(color='#65788E')
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        linecolor='#D9E3ED',
        tickfont=dict(color='#65788E')
    )
    return fig
