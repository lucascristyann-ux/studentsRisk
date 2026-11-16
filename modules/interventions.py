import pandas as pd
import streamlit as st
import plotly.express as px
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#0F766E", "#2563EB", "#D97706", "#7C3AED", "#0891B2"]
from .core import services,protected_table,download_csv,chapter3_research_scores,CHAPTER3_TIERS
from .theme import polish,TEAL

def render(df,a):
    st.markdown('### Student support program demand')
    st.info('Primary measure: respondents explicitly selecting a service. This is expressed demand, not confirmed eligibility. A respondent may select multiple services.')
    s=services(df)
    if s.empty:st.warning('No service selections found.');return
    n=a.College.value_counts()
    counts=s.groupby(a.College).sum()
    safe=protected_table(counts,n)
    options=['All colleges']+safe.index.tolist()
    selected=st.selectbox('College',options)
    view=safe if selected=='All colleges' else safe.loc[[selected]]
    st.dataframe(view.style.format(na_rep='Suppressed',precision=0),use_container_width=True)
    st.plotly_chart(polish(px.imshow(view,aspect='auto',color_continuous_scale='Teal',title='Requested services by college · small counts hidden'),max(420,70*len(view))),use_container_width=True)
    download_csv(view,'adssu_suppressed_program_demand.csv','Download suppressed program matrix')
    st.subheader('Service request rates by college')
    rates=counts.div(n,axis=0)*100
    rates=rates.loc[safe.index].where(safe.notna())
    st.dataframe(rates.round(1).style.format(na_rep='Suppressed'),use_container_width=True)
    st.subheader('Suggested OSAS program pathways')
    interventions=pd.DataFrame([
      ('Academic coaching','Academic engagement challenges','Study skills, adviser consultations, tutoring'),
      ('Food and financial support','Self-reported food or financial strain','Confidential assessment for scholarships, pantry and transport support'),
      ('Counseling and mental health','Low WHO-5 screening or voluntary request','Offer confidential professional assessment and counseling'),
      ('Peer belonging','Limited social support','Peer mentoring and facilitated student groups'),
      ('Digital wellness','Sleep disruption or digital stress','Digital boundaries and sleep education'),
      ('Housing support','Housing instability response','Safe housing referral and assistance'),
      ('Service awareness','Help-seeking knowledge gaps','OSAS orientation and confidential access information'),
      ('Urgent safety response','Direct safety disclosures','Restricted, prompt human-led triage under approved protocol')],columns=['Program','Exploratory signal','Recommended action'])
    st.dataframe(interventions,hide_index=True,use_container_width=True)
    st.warning('This is an aggregate planning dashboard. It cannot identify or contact individual students. Safety flags must be handled separately in a restricted OSAS workflow.')

    # CHAPTER 3 ADDITION: Objective 4
    st.divider()
    st.markdown('### Research Objective 4 · Evidence-linked intervention planning')
    st.caption('Recommendations below are generated from aggregate research findings. They support OSAS program planning and do not replace counselor judgment or individual clinical assessment.')

    if st.button('Generate Objective 4 recommendations',key='chapter3_interventions') or st.session_state.get('chapter3_intervention_ready'):
        st.session_state.chapter3_intervention_ready=True

        r=chapter3_research_scores(df)
        research=r[r.Research_scope & r.Risk_tier_final.notna()].copy()
        counts=research.Risk_tier_final.value_counts().reindex(CHAPTER3_TIERS,fill_value=0)
        total=max(int(counts.sum()),1)
        high_critical=int(counts.get('High Risk',0)+counts.get('Critical Risk',0))
        hc_percent=100*high_critical/total

        c1,c2,c3=st.columns(3)
        c1.metric('Research risk records',f'{total:,}')
        c2.metric('High + Critical tiers',f'{high_critical:,}')
        c3.metric('High + Critical share',f'{hc_percent:.1f}%')

        st.subheader('Risk-tier context')
        risk_table=counts.rename('Responses').to_frame()
        risk_table['Percent']=100*risk_table.Responses/total
        st.dataframe(risk_table.round(1),use_container_width=True)

        recs=chapter3_intervention_recommendations(df)
        st.subheader('Interventions linked to Objective 2 predictors')
        if recs.empty:
            st.info('No predictors passed the current Objective 2 selection rules.')
        else:
            st.dataframe(recs,hide_index=True,use_container_width=True)

        best=st.session_state.get('chapter3_best_model')
        if best:
            st.success(f'Objective 3 validated-model context available in this session: {best}')
        else:
            st.info('Run Objective 3 model validation under Predictive Analytics to display the best validated model here.')

        st.caption('Use these results as Chapter 4 evidence for proposed intervention strategies. Program recommendations should still be reviewed with OSAS and matched to existing referral and counselor-response procedures.')



# CHAPTER 3 ADDITION: Objective 4 evidence-linked interventions
def chapter3_intervention_recommendations(df):
    from .prediction import chapter3_feature_screen

    selected,_=chapter3_feature_screen(df)
    selected=selected[selected.Selected_for_model].copy()
    if selected.empty:
        return pd.DataFrame(columns=['Predictor','Evidence','Program pathway','Recommended action'])

    mapping={
        'academic':('Academic coaching','Study skills support, adviser consultation, tutoring, and workload-management sessions'),
        'housing':('Housing and financial support','Confidential housing, food, transport, scholarship, and financial-assistance referral'),
        'autonomy':('Self-management support','Goal-setting, time-management, boundary-setting, and self-management workshops'),
        'digital_self':('Digital wellness','Sleep hygiene, screen-time boundaries, and healthy digital-use education'),
        'self_care':('Wellness and self-care','Structured self-care, sleep, physical activity, and stress-management programs'),
        'coping':('Coping and counseling support','Evidence-based coping-skills sessions and voluntary counseling referral'),
        'relationship_access':('Relationship and safety support','Relationship-safety education and confidential access to appropriate support services'),
        'help_literacy':('Service awareness and help-seeking','OSAS orientation, mental-health literacy, and confidential help-seeking information'),
        'age':('Targeted student support','Use age group only for aggregate targeting and scheduling of support programs'),
        'college':('College-targeted planning','Coordinate aggregate wellness programming with colleges showing higher need'),
        'year_level':('Year-level support','Target transition, retention, and adjustment programs by year level'),
        'internet_access':('Access support','Improve access to devices, connectivity, and digital student-support channels'),
        'housing_status':('Housing support','Review housing stability and connect students to appropriate institutional support')
    }

    rows=[]
    for _,row in selected.sort_values('Information_Gain',ascending=False).iterrows():
        feature=row['Feature']
        program,action=mapping.get(
            feature,
            ('Student wellness support','Review this predictor with OSAS and design an intervention matched to the observed survey pattern')
        )
        evidence=f"Information gain={row['Information_Gain']:.4f}"
        if pd.notna(row['Chi2_p']):
            evidence+=f"; Chi-square p={row['Chi2_p']:.4f}"
        rows.append({
            'Predictor':feature,
            'Evidence':evidence,
            'Program pathway':program,
            'Recommended action':action
        })
    return pd.DataFrame(rows)
