import hashlib
import streamlit as st
from modules.core import load_data,enrich
from modules.theme import setup,hero
from modules import overview,eda,wellness,clustering,prediction,interventions

st.set_page_config(page_title='ADSSU | Wellness Intelligence',page_icon='✦',layout='wide',initial_sidebar_state='expanded')
setup()
st.sidebar.markdown(
    '''<div class="sidebar-brand">
        <div class="brand-kicker">AGUSAN DEL SUR STATE UNIVERSITY</div>
        <div class="brand-title">Wellness Intelligence</div>
        <div class="brand-sub">Office of Student Affairs & Services · Research Analytics Workspace</div>
    </div>''',
    unsafe_allow_html=True
)
st.sidebar.divider()
st.sidebar.markdown('#### DATASET')
st.sidebar.caption('Upload or replace the authorized de-identified survey file.')
upload=st.sidebar.file_uploader(
    'Upload / Replace Dataset',
    type='csv',
    key='survey_dataset_uploader',
    help='Use authorized de-identified CSV only. Public hosting is not suitable for sensitive wellness data.'
)
if upload is not None:
    st.session_state['survey_raw']=upload.getvalue()
    st.session_state['survey_filename']=upload.name

if 'survey_raw' not in st.session_state:
    hero('Student Wellness Intelligence','Analytics, behavioral insights and evidence-informed program planning')
    st.info('Upload an authorized de-identified ADSSU-CWI CSV from the sidebar to begin.')
    st.warning('Research prototype only. Use institution-approved authentication and private hosting before processing sensitive student responses.')
    st.stop()

raw=st.session_state['survey_raw']
if upload is None:
    st.sidebar.caption('Using the dataset kept in this current Streamlit session.')
fingerprint=hashlib.sha256(raw).hexdigest()
if st.session_state.get('dataset_fingerprint')!=fingerprint:
    st.session_state.dataset_fingerprint=fingerprint
    for k in ['cluster_ready','sig_ready','train_ready','chapter3_feature_ready','chapter3_model_ready','chapter3_intervention_ready','chapter3_best_model','chapter3_model_comparison','chapter3_kmeans_ready','chapter3_stats_ready','chapter3_preprocess_ready']:
        st.session_state.pop(k,None)
try:
    df=load_data(raw)
    a=enrich(df)
except Exception as e:
    st.error(f'Unable to read dataset: {e}')
    st.stop()

dataset_name=st.session_state.get('survey_filename','ADSSU-CWI dataset')
st.sidebar.markdown(
    f'''<div class="dataset-status">
        <div class="dataset-status-label">CURRENT DATASET</div>
        <div class="dataset-status-name">{dataset_name}</div>
        <div class="dataset-status-meta">{len(df):,} records loaded</div>
        <div class="dataset-status-state"><span></span> Ready for analysis</div>
    </div>''',
    unsafe_allow_html=True
)

pages={'Executive Dashboard':overview,'Data Exploration':eda,'Wellness Profile':wellness,'Behavioral Segments':clustering,'Predictive Analytics':prediction,'Intervention Planning':interventions}
st.sidebar.markdown('#### ANALYTICS WORKSPACE')
st.sidebar.caption('Navigate by research and analysis function')
page=st.sidebar.radio('Navigation',list(pages),label_visibility='collapsed')
st.sidebar.divider()
st.sidebar.caption('Screening indicators are not diagnoses. Small groups are suppressed. No individual referrals are automated.')
subtitles={'Executive Dashboard':'Institution-wide indicators and survey participation','Data Exploration':'Response quality, distributions and survey relationships','Wellness Profile':'WHO-5 and PHQ-9 screening summaries','Behavioral Segments':'Unsupervised patterns and student behavioral profiles','Predictive Analytics':'Survey predictors, model comparison and validation','Intervention Planning':'College-level program demand and OSAS service planning'}
hero(page,subtitles[page])
pages[page].render(df,a)
