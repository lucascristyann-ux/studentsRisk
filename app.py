import hashlib
import streamlit as st
from modules.core import load_data,enrich
from modules.theme import setup,hero
from modules.storage import is_configured,load_latest_dataset,save_dataset
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

# Restore the latest private dataset automatically after a fresh app/session start.
storage_error=None
if 'survey_raw' not in st.session_state and is_configured():
    try:
        saved=load_latest_dataset()
    except Exception as e:
        saved=None
        storage_error=str(e)
    if saved:
        st.session_state['survey_raw']=saved['raw']
        st.session_state['survey_filename']=saved['filename']
        st.session_state['survey_source']='Private storage'
        st.session_state['persistent_object_path']=saved['object_path']

upload=st.sidebar.file_uploader(
    'Upload / Replace Dataset',
    type='csv',
    key='survey_dataset_uploader',
    help='Use an authorized de-identified CSV. A validated upload is saved to the private OSAS storage bucket.'
)

if upload is not None:
    candidate=upload.getvalue()
    candidate_fp=hashlib.sha256(candidate).hexdigest()
    if st.session_state.get('last_seen_upload_fingerprint')!=candidate_fp:
        st.session_state['last_seen_upload_fingerprint']=candidate_fp
        try:
            # Validate before replacing the active or persistent dataset.
            load_data(candidate)
        except Exception as e:
            st.sidebar.error(f'Dataset not accepted: {e}')
        else:
            st.session_state['survey_raw']=candidate
            st.session_state['survey_filename']=upload.name
            st.session_state['survey_source']='Current session'
            if is_configured():
                try:
                    saved=save_dataset(candidate,upload.name)
                except Exception as e:
                    st.sidebar.warning('Dataset loaded for this session, but private storage could not save it.')
                    st.sidebar.error(f'Private storage error: {str(e)[:180]}')
                    st.session_state['storage_save_error']=str(e)
                else:
                    st.session_state['survey_source']='Private storage'
                    st.session_state['persistent_object_path']=saved['object_path']
                    st.session_state.pop('storage_save_error',None)
                    st.sidebar.markdown(
                        '<div class="storage-success"><span class="storage-success-dot"></span><span>Dataset validated and saved privately.</span></div>',
                        unsafe_allow_html=True
                    )

if 'survey_raw' not in st.session_state:
    hero('Student Wellness Intelligence','Analytics, behavioral insights and evidence-informed program planning')
    st.info('Upload an authorized de-identified ADSSU-CWI CSV from the sidebar to begin.')
    if storage_error:
        st.warning('Private storage is configured but the saved dataset could not be loaded. Upload a valid CSV to continue.')
        st.caption(f'Storage error: {storage_error[:220]}')
    else:
        st.warning('Use institution-approved access controls before processing sensitive student responses.')
    st.stop()

raw=st.session_state['survey_raw']
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
dataset_source=st.session_state.get('survey_source','Current session')
st.sidebar.markdown(
    f'''<div class="dataset-status">
        <div class="dataset-status-label">CURRENT DATASET</div>
        <div class="dataset-status-name">{dataset_name}</div>
        <div class="dataset-status-meta">{len(df):,} records loaded · {dataset_source}</div>
        <div class="dataset-status-state"><span></span> Ready for analysis</div>
    </div>''',
    unsafe_allow_html=True
)
if is_configured():
    st.sidebar.caption('Private storage is connected. The latest saved dataset loads automatically after app restarts.')
else:
    st.sidebar.caption('Private storage is not configured. Uploads remain session-only.')

pages={'Executive Dashboard':overview,'Data Exploration':eda,'Wellness Profile':wellness,'Behavioral Segments':clustering,'Predictive Analytics':prediction,'Intervention Planning':interventions}
st.sidebar.markdown('#### ANALYTICS WORKSPACE')
st.sidebar.caption('Navigate by research and analysis function')
page=st.sidebar.radio('Navigation',list(pages),label_visibility='collapsed')
st.sidebar.divider()
st.sidebar.caption('Screening indicators are not diagnoses. Small groups are suppressed. No individual referrals are automated.')
subtitles={'Executive Dashboard':'Institution-wide indicators and survey participation','Data Exploration':'Response quality, distributions and survey relationships','Wellness Profile':'WHO-5 and PHQ-9 screening summaries','Behavioral Segments':'Unsupervised patterns and student behavioral profiles','Predictive Analytics':'Survey predictors, model comparison and validation','Intervention Planning':'College-level program demand and OSAS service planning'}
hero(page,subtitles[page])
pages[page].render(df,a)
