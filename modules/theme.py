import streamlit as st
import plotly.express as px

# Warm academic palette inspired by the supplied dashboard reference.
NAVY='#3B281F'
NAVY_2='#563A2A'
TEAL='#91A66B'
TEAL_DARK='#657A35'
BLUE='#7F9FA8'
MUTED='#7D7068'
BG='#F3EEE8'
BORDER='#DED3C8'
ORANGE='#E88920'
CORAL='#C96F5B'
PALETTE=[TEAL,'#DFA06F',BLUE,'#A28DC3','#D58B7C','#BBA079']

def setup():
    px.defaults.template='plotly_white'
    px.defaults.color_discrete_sequence=PALETTE
    st.markdown('''<style>
    :root{
        --espresso:#3B281F;
        --espresso2:#563A2A;
        --cream:#F3EEE8;
        --ivory:#FBF8F4;
        --card:#FFFFFF;
        --olive:#82994A;
        --olive-dark:#657A35;
        --orange:#E88920;
        --lavender:#8877D3;
        --coral:#C96F5B;
        --muted:#7D7068;
        --border:#DED3C8;
        --soft:#F7F2EC;
    }

    /* Main workspace */
    .stApp{
        background:
          radial-gradient(circle at 91% 3%,rgba(232,137,32,.075),transparent 27%),
          radial-gradient(circle at 72% 26%,rgba(136,119,211,.045),transparent 25%),
          linear-gradient(180deg,#F8F4EF 0%,#F1EBE4 100%);
        color:var(--espresso);
    }
    .block-container{
        max-width:1480px;
        padding-top:1.45rem;
        padding-bottom:4rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"]{
        background:
          radial-gradient(circle at 10% 4%,rgba(255,255,255,.06),transparent 22%),
          linear-gradient(180deg,#4A2E21 0%,#3D271E 55%,#352219 100%)!important;
        border-right:1px solid rgba(255,255,255,.06);
    }
    [data-testid="stSidebar"] > div:first-child{padding-top:.55rem}
    [data-testid="stSidebar"] *{color:#F8F0E9!important}
    [data-testid="stSidebar"] hr{
        border-color:rgba(255,255,255,.13)!important;
        margin:1rem 0!important;
    }

    [data-testid="stSidebar"] .sidebar-brand{
        padding:17px 15px 14px;
        margin:0 0 10px;
        border:1px solid rgba(255,255,255,.10);
        border-radius:18px;
        background:linear-gradient(145deg,rgba(255,255,255,.09),rgba(255,255,255,.025));
        box-shadow:0 12px 28px rgba(20,10,4,.15);
    }
    [data-testid="stSidebar"] .sidebar-brand .brand-kicker{
        font-size:9.5px;
        letter-spacing:.16em;
        font-weight:800;
        color:#E9CFAF!important;
        margin-bottom:7px;
    }
    [data-testid="stSidebar"] .sidebar-brand .brand-title{
        font-size:20px;
        font-weight:830;
        color:#FFFFFF!important;
        line-height:1.08;
    }
    [data-testid="stSidebar"] .sidebar-brand .brand-sub{
        margin-top:7px;
        color:#DACCC2!important;
        font-size:11px;
        line-height:1.45;
    }

    /* Survey dataset uploader: soft accent card */
    [data-testid="stSidebar"] [data-testid="stFileUploader"]{
        position:relative;
        overflow:hidden;
        background:
          radial-gradient(circle at 92% 8%,rgba(255,255,255,.20),transparent 32%),
          linear-gradient(135deg,#9EAC7B 0%,#B8A17F 100%);
        border:1px solid rgba(255,255,255,.22);
        border-radius:18px;
        padding:11px;
        box-shadow:0 9px 22px rgba(30,17,9,.12);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"]:before{
        content:'';
        position:absolute;
        width:72px;height:72px;
        border-radius:50%;
        right:-30px;bottom:-34px;
        background:rgba(255,255,255,.10);
        pointer-events:none;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section{
        background:rgba(255,253,249,.96)!important;
        border-radius:14px!important;
        border:1px solid rgba(255,255,255,.58)!important;
        box-shadow:0 3px 10px rgba(66,45,31,.06);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section *{
        color:#52392B!important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button{
        border-radius:10px!important;
        border:1px solid #C8B59F!important;
        background:#F8F2EB!important;
        color:#5B4234!important;
        font-weight:700!important;
    }

    /* Current dataset status card */
    [data-testid="stSidebar"] .dataset-status{
        margin:10px 0 14px;
        padding:13px 14px;
        border-radius:15px;
        background:rgba(255,255,255,.075);
        border:1px solid rgba(255,255,255,.11);
        box-shadow:0 7px 18px rgba(20,10,4,.10);
    }
    [data-testid="stSidebar"] .dataset-status-label{
        font-size:9px;
        letter-spacing:.15em;
        font-weight:800;
        color:#DCC29E!important;
        margin-bottom:5px;
    }
    [data-testid="stSidebar"] .dataset-status-name{
        font-size:12px;
        font-weight:760;
        color:#FFFFFF!important;
        line-height:1.35;
        word-break:break-word;
    }
    [data-testid="stSidebar"] .dataset-status-meta{
        margin-top:5px;
        font-size:11px;
        color:#D8CBC1!important;
    }
    [data-testid="stSidebar"] .dataset-status-state{
        margin-top:8px;
        font-size:10.5px;
        color:#E6DAD0!important;
        display:flex;
        align-items:center;
        gap:7px;
    }
    [data-testid="stSidebar"] .dataset-status-state span{
        width:8px;
        height:8px;
        border-radius:50%;
        background:#A7BD72;
        box-shadow:0 0 0 3px rgba(167,189,114,.14);
        display:inline-block;
    }

    /* Navigation */
    [data-testid="stSidebar"] [role="radiogroup"]{gap:5px}
    [data-testid="stSidebar"] [role="radiogroup"] label{
        padding:10px 11px;
        border-radius:12px;
        margin:2px 0;
        border:1px solid transparent;
        transition:all .18s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover{
        background:rgba(255,255,255,.07);
        border-color:rgba(255,255,255,.08);
        transform:translateX(2px);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
        background:linear-gradient(90deg,#A55E26,#C8762F)!important;
        border-color:rgba(255,255,255,.13)!important;
        box-shadow:0 8px 19px rgba(25,12,5,.18);
    }

    /* Typography */
    h1,h2,h3,h4{
        color:var(--espresso)!important;
        letter-spacing:-.025em;
    }
    h1{font-size:2rem!important;font-weight:820!important}
    h2{font-size:1.5rem!important;font-weight:790!important}
    h3{font-size:1.17rem!important;font-weight:760!important}
    p,li{line-height:1.57}
    [data-testid="stCaptionContainer"]{color:#86786F!important}

    /* Page hero */
    .hero{
        position:relative;
        overflow:hidden;
        padding:27px 30px;
        border-radius:23px;
        background:
          radial-gradient(circle at 84% 18%,rgba(232,137,32,.18),transparent 27%),
          radial-gradient(circle at 96% 90%,rgba(130,153,74,.13),transparent 23%),
          linear-gradient(115deg,#493025 0%,#5B3D2B 57%,#6B4932 100%);
        color:#fff;
        margin:0 0 22px;
        box-shadow:0 15px 38px rgba(61,39,30,.14);
        border:1px solid rgba(255,255,255,.09);
    }
    .hero:before{
        content:'';
        position:absolute;
        width:115px;height:115px;
        border-radius:50%;
        background:rgba(255,255,255,.025);
        right:34px;bottom:-62px;
    }
    .hero:after{
        content:'';
        position:absolute;
        width:190px;height:190px;
        border-radius:50%;
        border:1px solid rgba(255,255,255,.08);
        right:-70px;top:-96px;
    }
    .hero .eyebrow{
        font-size:10.5px;
        font-weight:800;
        letter-spacing:.19em;
        color:#F0C690;
        margin-bottom:8px;
    }
    .hero .heading{
        font-size:29px;
        font-weight:830;
        line-height:1.16;
        color:white;
    }
    .hero .sub{
        color:#EFE5DE;
        font-size:14px;
        margin-top:9px;
        max-width:780px;
    }

    /* Metric cards */
    [data-testid="stMetric"]{
        position:relative;
        overflow:hidden;
        background:rgba(255,255,255,.97)!important;
        border:1px solid var(--border);
        border-radius:19px;
        padding:19px 20px 18px;
        box-shadow:0 7px 24px rgba(59,40,31,.055);
        min-height:116px;
        transition:transform .16s ease,box-shadow .16s ease;
    }
    [data-testid="stMetric"]:before{
        content:'';
        position:absolute;
        left:0;top:0;bottom:0;
        width:4px;
        background:linear-gradient(180deg,var(--orange),#E5B04C);
    }
    [data-testid="stMetric"]:after{
        content:'';
        position:absolute;
        width:62px;height:62px;
        border-radius:50%;
        background:rgba(232,137,32,.05);
        right:-22px;top:-20px;
    }
    [data-testid="stMetric"]:hover{
        transform:translateY(-1px);
        box-shadow:0 11px 29px rgba(59,40,31,.075);
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *{
        color:#7B6E66!important;
        font-weight:700!important;
        opacity:1!important;
        font-size:.87rem!important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *{
        color:var(--espresso)!important;
        font-size:2rem!important;
        font-weight:830!important;
        opacity:1!important;
    }

    /* Result surfaces */
    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"]{
        background:rgba(255,255,255,.985);
        border:1px solid var(--border);
        border-radius:19px;
        padding:10px;
        box-shadow:0 5px 20px rgba(59,40,31,.045);
    }

    /* Tabs */
    [data-testid="stTabs"] button{
        font-weight:720;
        color:#7A6E66!important;
    }
    [data-testid="stTabs"] button[aria-selected="true"]{
        color:#A45F29!important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"]{
        background-color:#C8752E!important;
    }

    /* Expanders */
    [data-testid="stExpander"]{
        border:1px solid var(--border)!important;
        border-radius:15px!important;
        background:#FFFFFF!important;
        box-shadow:0 3px 12px rgba(59,40,31,.03);
    }

    /* Buttons */
    .stButton>button{
        border-radius:11px!important;
        font-weight:730!important;
        min-height:2.65rem;
        background:#FFFDFB!important;
        border:1px solid #D8CBC0!important;
        color:#55392B!important;
        transition:all .16s ease!important;
    }
    .stButton>button:hover{
        border-color:#C8752E!important;
        color:#A45F29!important;
        transform:translateY(-1px);
        box-shadow:0 5px 13px rgba(77,47,32,.07);
    }
    .stButton>button[kind="primary"]{
        background:linear-gradient(90deg,#A65E25,#D07C30)!important;
        border-color:#C8752E!important;
        color:white!important;
        box-shadow:0 7px 17px rgba(166,94,37,.18);
    }
    .stButton>button[kind="primary"]:hover{
        color:white!important;
        box-shadow:0 9px 21px rgba(166,94,37,.24);
    }

    /* Alerts */
    [data-testid="stAlert"]{
        border-radius:14px!important;
        border-width:1px!important;
    }

    /* Inputs */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div{
        border-radius:11px!important;
        border-color:#D9CEC4!important;
        background:#FFFDFC!important;
    }

    /* Separators */
    hr{
        border-color:#E2D8CF!important;
        margin:1.7rem 0!important;
    }

    /* Downloads */
    [data-testid="stDownloadButton"] button{
        border-radius:11px!important;
        font-weight:710!important;
    }

    @media(max-width:700px){
        .hero{padding:21px 20px;border-radius:18px}
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
        plot_bgcolor='#FFFCF9',
        font=dict(family='Arial',color=NAVY,size=12),
        title_font=dict(size=17,color=NAVY,family='Arial'),
        legend_title_text='',
        hoverlabel=dict(bgcolor='#FFFDFB',font_color=NAVY,bordercolor='#D8CBC0'),
        colorway=PALETTE,
        bargap=.28,
        hovermode='closest',
        legend=dict(
            bgcolor='rgba(255,255,255,.72)',
            bordercolor='rgba(222,211,200,.72)',
            borderwidth=1
        )
    )

    # Softer graph shapes without changing any data or calculations.
    for trace in fig.data:
        if trace.type in ('bar','histogram'):
            trace.update(opacity=.88,marker_line_width=0)
            if trace.type=='bar':
                try:
                    trace.update(marker_cornerradius=7)
                except Exception:
                    pass
        elif trace.type in ('scatter','scattergl'):
            try:
                trace.update(line=dict(width=2.4))
            except Exception:
                pass
        elif trace.type=='pie':
            try:
                existing_hole=float(trace.hole or 0)
                trace.update(
                    hole=max(existing_hole,.38),
                    sort=False,
                    marker_line=dict(color='#FFF9F4',width=2)
                )
            except Exception:
                pass

    fig.update_xaxes(
        showgrid=True,
        gridcolor='#F1EAE3',
        zeroline=False,
        linecolor='#E2D7CC',
        tickfont=dict(color='#7D7068'),
        ticks='outside',
        tickcolor='#D8CDC3'
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        linecolor='#E2D7CC',
        tickfont=dict(color='#7D7068'),
        ticks='outside',
        tickcolor='#D8CDC3'
    )
    return fig
