import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#0F766E", "#2563EB", "#D97706", "#7C3AED", "#0891B2"]
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,adjusted_rand_score,cohen_kappa_score
from .core import domains,protected_table,chapter3_research_scores,CHAPTER3_TIERS
from .theme import polish

@st.cache_data(show_spinner='Computing behavioral clusters...',max_entries=3)
def fit_clusters(d):
    eligible=d.notna().sum(axis=1)>=5
    z=StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(d.loc[eligible]))
    candidates=[]
    for k in range(2,min(7,len(z))):
        km=KMeans(n_clusters=k,n_init=20,random_state=42).fit(z)
        candidates.append({'k':k,'silhouette':silhouette_score(z,km.labels_,sample_size=min(1000,len(z)),random_state=42),'min_cluster':int(np.bincount(km.labels_).min())})
    q=pd.DataFrame(candidates)
    good=q[q.min_cluster>=max(10,int(.03*len(z)))]
    k=int((good if not good.empty else q).sort_values('silhouette',ascending=False).iloc[0].k)
    km=KMeans(n_clusters=k,n_init=20,random_state=42).fit(z)
    pc=PCA(n_components=2).fit_transform(z)
    labels=pd.Series(km.labels_+1,index=d.loc[eligible].index,name='Cluster')
    scatter=pd.DataFrame({'PC1':pc[:,0],'PC2':pc[:,1],'Cluster':labels.astype(str).values},index=labels.index)
    profile=d.loc[eligible].groupby(labels).mean()
    ari=np.mean([adjusted_rand_score(km.labels_,KMeans(n_clusters=k,n_init=20,random_state=s).fit_predict(z)) for s in [7,19,123,2026]])
    return q,labels,scatter,profile,ari

def render(df,a):
    st.markdown('### Behavioral pattern discovery')
    st.caption('Unsupervised clustering uses behavioral domains, excluding wellness outcome, safety responses, college, and direct identifiers.')

    st.divider()
    st.markdown('### K-Means risk validation')
    st.caption('The validation loads automatically using k=4 and the five standardized CRI subscales.')

    try:
        sil,kappa,profile4,agreement,centroids=chapter3_risk_clusters(df)
    except Exception as e:
        st.error(f'K-Means validation could not run: {e}')
    else:
        c1,c2=st.columns(2)
        c1.metric('Silhouette coefficient',f'{sil:.3f}')
        c2.metric("Cohen's kappa",f'{kappa:.3f}')

        if kappa>=.60:
            st.success("Kappa is at least 0.60, meeting the substantial-agreement criterion.")
        else:
            st.warning("Kappa is below 0.60. The clusters show limited agreement with the final research tiers.")

        risk_colors={
            'Low Risk':'#91A66B',
            'Moderate Risk':'#D8B56A',
            'High Risk':'#D9915B',
            'Critical Risk':'#C97167'
        }

        profile_plot=profile4.reset_index().rename(columns={'KMeans_risk_tier':'Risk tier'})
        c1,c2=st.columns([1.05,1])
        with c1:
            fig=px.bar(
                profile_plot,x='Risk tier',y='Mean_CRI',
                color='Risk tier',text='Mean_CRI',
                title='Mean CRI by K-Means risk profile',
                category_orders={'Risk tier':CHAPTER3_TIERS},
                color_discrete_map=risk_colors
            )
            fig.update_traces(texttemplate='%{text:.2f}',textposition='outside',cliponaxis=False)
            st.plotly_chart(polish(fig,430),use_container_width=True)

        with c2:
            fig=px.imshow(
                agreement,
                text_auto=True,
                aspect='auto',
                color_continuous_scale=['#F7F1EA','#E7C89A','#C97167'],
                title='K-Means tier vs final research tier',
                labels={'x':'K-Means tier','y':'Final research tier','color':'Count'}
            )
            st.plotly_chart(polish(fig,430),use_container_width=True)

        with st.expander('View standardized four-cluster profile'):
            st.dataframe(profile4.round(3),use_container_width=True)

        with st.expander('View cluster centroids and risk-order mapping'):
            st.dataframe(centroids.round(3),use_container_width=True)

        st.caption('K-Means cluster numbers are arbitrary. Clusters are ordered from lowest to highest centroid risk score before mapping to Low, Moderate, High and Critical for agreement testing.')

    st.divider()
    st.caption('Supplementary behavioral-clustering routines remain preserved in the source code and are not recomputed automatically on this page.')



# CHAPTER 3 ADDITION: fixed k=4 risk-cluster validation
@st.cache_data(show_spinner='Running K-Means validation...',max_entries=3)
def chapter3_risk_clusters(df):
    r=chapter3_research_scores(df)
    cols=['z_PHQ9','z_AUDIT9','z_Food','z_WHO5','z_MSPSS']
    valid=(
        r.Research_scope
        & r.Risk_tier_final.notna()
        & r[cols].notna().all(axis=1)
    )
    x=r.loc[valid,cols].astype(float)

    if len(x)<20:
        raise ValueError('Not enough complete research-scope records for k=4 clustering.')

    km=KMeans(n_clusters=4,n_init=50,random_state=42).fit(x)
    sil=float(silhouette_score(x,km.labels_))

    centroids=pd.DataFrame(km.cluster_centers_,columns=cols,index=range(4))
    centroids['Risk_score']=(
        centroids['z_PHQ9']
        +centroids['z_AUDIT9']
        +centroids['z_Food']
        -centroids['z_WHO5']
        -centroids['z_MSPSS']
    )
    ordered=centroids['Risk_score'].sort_values().index.tolist()
    mapping={cluster:CHAPTER3_TIERS[i] for i,cluster in enumerate(ordered)}

    cluster_tier=pd.Series(
        [mapping[int(label)] for label in km.labels_],
        index=x.index,
        name='KMeans_risk_tier'
    )
    observed=r.loc[x.index,'Risk_tier_final'].astype(str)
    kappa=float(cohen_kappa_score(
        observed,cluster_tier,
        labels=CHAPTER3_TIERS
    ))

    profile=x.copy()
    profile['CRI']=r.loc[x.index,'CRI']
    profile['KMeans_risk_tier']=cluster_tier
    summary=profile.groupby('KMeans_risk_tier').agg(
        Responses=('CRI','size'),
        Mean_CRI=('CRI','mean'),
        z_PHQ9=('z_PHQ9','mean'),
        z_AUDIT9=('z_AUDIT9','mean'),
        z_Food=('z_Food','mean'),
        z_WHO5=('z_WHO5','mean'),
        z_MSPSS=('z_MSPSS','mean')
    ).reindex(CHAPTER3_TIERS)

    agreement=pd.crosstab(
        observed.rename('Quartile_plus_override'),
        cluster_tier,
        dropna=False
    ).reindex(index=CHAPTER3_TIERS,columns=CHAPTER3_TIERS,fill_value=0)

    centroids['Mapped_tier']=pd.Series(mapping)
    centroids=centroids.sort_values('Risk_score')
    return sil,kappa,summary,agreement,centroids
