import streamlit as st
import plotly.express as px
from .theme import polish,TEAL,BLUE
from .core import chapter3_research_scores,CHAPTER3_TIERS

def render(df,a):
    valid=a.dropna(subset=['WHO5_score'])
    summary=valid.groupby('College').agg(Responses=('WHO5_score','size'),Mean_WHO5=('WHO5_score','mean'),Low_WHO5=('low_wellbeing','sum'))
    summary=summary[summary.Responses>=10].copy()
    summary['Below_50_percent']=100*summary.Low_WHO5/summary.Responses
    summary['Low_WHO5']=summary.Low_WHO5.where(summary.Low_WHO5>=10)
    t1,t2=st.tabs(['College comparisons','Statistical summary'])
    with t1:
        chart=summary.reset_index().sort_values('Mean_WHO5')
        c1,c2=st.columns(2)
        with c1:st.plotly_chart(polish(px.bar(chart,x='Mean_WHO5',y='College',orientation='h',title='Average WHO-5 score by college',color_discrete_sequence=[TEAL])),use_container_width=True)
        with c2:st.plotly_chart(polish(px.bar(chart,x='Below_50_percent',y='College',orientation='h',title='Share of respondents below WHO-5 threshold',labels={'Below_50_percent':'Percentage (%)'},color_discrete_sequence=[BLUE])),use_container_width=True)
    with t2:
        st.dataframe(valid[['WHO5_score','PHQ9_score']].describe().round(2),use_container_width=True)
        st.dataframe(summary.round(2),use_container_width=True)
    st.caption('Screening results are not diagnoses. College groups with fewer than 10 responses are excluded; small counts are suppressed in tables.')

    # CHAPTER 3 ADDITION: Objective 1 aggregate wellness-risk profile
    st.divider()
    st.markdown('### Research Objective 1 · Composite wellness profile')
    r=chapter3_research_scores(df)
    research=r[r.Research_scope & r.CRI.notna()].copy()

    c1,c2,c3=st.columns(3)
    c1.metric('Research-scope records',f'{int(r.Research_scope.sum()):,}')
    c2.metric('Valid composite scores',f'{len(research):,}')
    c3.metric('Adapted AUDIT respondents',f'{int((r.Research_scope & r.AUDIT_applicable).sum()):,}')

    o1,o2=st.columns(2)
    o1.metric('PHQ-9 item 9 escalations',f'{int((research.PHQ9_item9_override).sum()):,}')
    o2.metric('Immediate-danger escalations',f'{int((research.Immediate_danger_override).sum()):,}')

    counts=research['Risk_tier_final'].value_counts().reindex(CHAPTER3_TIERS,fill_value=0)
    risk=counts.rename('Responses').to_frame()
    risk['Percent']=100*risk['Responses']/risk['Responses'].sum()
    risk.index.name='Risk tier'

    c1,c2=st.columns([1,1.35])
    with c1:
        st.dataframe(risk.round(1),use_container_width=True)
    with c2:
        chart=risk.reset_index()
        fig=px.bar(
            chart,x='Risk tier',y='Responses',text='Responses',
            title='Final Chapter 3 wellness risk distribution',
            category_orders={'Risk tier':CHAPTER3_TIERS},
            color_discrete_sequence=[TEAL]
        )
        st.plotly_chart(polish(fig),use_container_width=True)

    st.subheader('Research scoring summary')
    cols=['PHQ9_score','WHO5_score','MSPSS_mean','Food_security_score','AUDIT9_score','CRI','CRI_no_AUDIT']
    summary_table=research[cols].describe().T[['count','mean','std','min','25%','50%','75%','max']]
    st.dataframe(summary_table.round(2),use_container_width=True)
    st.caption('This research section uses the OSAS-adapted 12-item MSPSS on a 1–5 scale and the OSAS-adapted 9-item AUDIT. The base tiers are sample-relative CRI quartiles. Chapter 3 escalation rules are then applied: PHQ-9 item 9 > 0 is at least High Risk, and an Immediate Danger = Yes response is Critical Risk. These are research classifications, not clinical diagnoses or automated referral decisions.')

