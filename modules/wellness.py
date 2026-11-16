import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import shapiro,f_oneway,kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
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

    chart=risk.reset_index()
    risk_colors={
        'Low Risk':'#91A66B',
        'Moderate Risk':'#D8B56A',
        'High Risk':'#D9915B',
        'Critical Risk':'#C97167'
    }
    high_critical=int(counts.get('High Risk',0)+counts.get('Critical Risk',0))
    high_critical_pct=100*high_critical/max(int(counts.sum()),1)

    st.metric('High + Critical research tiers',f'{high_critical:,} ({high_critical_pct:.1f}%)')

    c1,c2=st.columns([1.25,1])
    with c1:
        fig=px.bar(
            chart,x='Risk tier',y='Responses',text='Responses',
            color='Risk tier',
            title='Final wellness risk distribution',
            category_orders={'Risk tier':CHAPTER3_TIERS},
            color_discrete_map=risk_colors
        )
        fig.update_traces(textposition='outside',cliponaxis=False)
        st.plotly_chart(polish(fig,430),use_container_width=True)
    with c2:
        fig=px.pie(
            chart,names='Risk tier',values='Responses',
            color='Risk tier',hole=.48,
            title='Share of final research risk tiers',
            category_orders={'Risk tier':CHAPTER3_TIERS},
            color_discrete_map=risk_colors
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(polish(fig,430),use_container_width=True)

    with st.expander('View exact risk-tier counts and percentages'):
        st.dataframe(risk.round(1),use_container_width=True)

    st.subheader('Research scoring summary')
    cols=['PHQ9_score','WHO5_score','MSPSS_mean','Food_security_score','AUDIT9_score','CRI','CRI_no_AUDIT']
    summary_table=research[cols].describe().T[['count','mean','std','min','25%','50%','75%','max']]
    st.dataframe(summary_table.round(2),use_container_width=True)
    st.caption('This research section uses the OSAS-adapted 12-item MSPSS on a 1–5 scale and the OSAS-adapted 9-item AUDIT. The base tiers are sample-relative CRI quartiles. Chapter 3 escalation rules are then applied: PHQ-9 item 9 > 0 is at least High Risk, and an Immediate Danger = Yes response is Critical Risk. These are research classifications, not clinical diagnoses or automated referral decisions.')

    st.divider()
    st.markdown('### Chapter 3 · Statistical analysis')
    st.caption('This section follows the manuscript sequence: Shapiro-Wilk normality checks, then one-way ANOVA or Kruskal-Wallis, followed by Tukey HSD or Dunn-Bonferroni when the omnibus test is significant.')

    if st.button('Run Chapter 3 statistical tests',key='chapter3_stats') or st.session_state.get('chapter3_stats_ready'):
        st.session_state.chapter3_stats_ready=True
        try:
            tests,posthoc=chapter3_statistical_analysis(df)
        except Exception as e:
            st.error(f'Chapter 3 statistical analysis could not run: {e}')
        else:
            st.dataframe(tests.round(4),hide_index=True,use_container_width=True)

            significant=tests[tests['Significant_p<.05']]
            st.metric('Significant omnibus comparisons',f'{len(significant):,}')

            if posthoc:
                st.subheader('Post-hoc comparisons for significant tests')
                for title,table in posthoc.items():
                    with st.expander(title):
                        st.dataframe(table,use_container_width=True)
            else:
                st.info('No post-hoc table was required because no omnibus comparison reached p < 0.05.')




# CHAPTER 3 ADDITION: statistical analysis specified in the methodology
@st.cache_data(show_spinner='Running Chapter 3 statistical analysis...',max_entries=3)
def chapter3_statistical_analysis(df):
    r=chapter3_research_scores(df)
    scope=r.Research_scope.copy()

    data=pd.DataFrame(index=df.index)
    data['College']=r['College']
    data['Year level/status']=r['Year level/status']
    data['Housing status']=df.iloc[:,43].astype('string').fillna('Missing')
    data['PHQ-9']=r['PHQ9_score']
    data['WHO-5']=r['WHO5_score']
    data['MSPSS']=r['MSPSS_mean']
    data['Food Security']=r['Food_security_score']
    data['AUDIT-9']=r['AUDIT9_score']

    rows=[]
    posthoc={}

    for group_col in ['College','Year level/status','Housing status']:
        for measure in ['PHQ-9','WHO-5','MSPSS','Food Security','AUDIT-9']:
            sub=data.loc[scope,[group_col,measure]].dropna()
            groups=[]
            shapiro_ps=[]

            for name,g in sub.groupby(group_col):
                vals=g[measure].astype(float).to_numpy()
                if len(vals)>=3:
                    groups.append((str(name),vals))
                    shapiro_ps.append(float(shapiro(vals).pvalue))

            if len(groups)<2:
                continue

            all_normal=all(p>.05 for p in shapiro_ps)
            arrays=[vals for _,vals in groups]

            if all_normal:
                stat,p=f_oneway(*arrays)
                test_name='One-way ANOVA'
            else:
                stat,p=kruskal(*arrays)
                test_name='Kruskal-Wallis'

            rows.append({
                'Grouping variable':group_col,
                'Measure':measure,
                'Groups tested':len(groups),
                'Normal in all groups':all_normal,
                'Minimum Shapiro p':min(shapiro_ps) if shapiro_ps else np.nan,
                'Test':test_name,
                'Statistic':float(stat),
                'p_value':float(p),
                'Significant_p<.05':bool(p<.05)
            })

            if p<.05:
                if all_normal:
                    tuk=pairwise_tukeyhsd(
                        endog=sub[measure].astype(float),
                        groups=sub[group_col].astype(str),
                        alpha=.05
                    )
                    table=pd.DataFrame(
                        tuk._results_table.data[1:],
                        columns=tuk._results_table.data[0]
                    )
                    posthoc[f'{group_col} · {measure} · Tukey HSD']=table
                else:
                    dunn=sp.posthoc_dunn(
                        sub,val_col=measure,group_col=group_col,
                        p_adjust='bonferroni'
                    )
                    posthoc[f'{group_col} · {measure} · Dunn-Bonferroni']=dunn

    return pd.DataFrame(rows),posthoc
