import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#0F766E", "#2563EB", "#D97706", "#7C3AED", "#0891B2"]
from scipy.stats import mannwhitneyu,chi2_contingency
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,f1_score,recall_score,confusion_matrix,brier_score_loss,roc_curve,accuracy_score,precision_score,classification_report
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from .core import domains,chapter3_research_scores,CHAPTER3_TIERS
from .theme import polish

def features(df):
    x=domains(df)
    for key,col in {'age':'Age','college':'College','year_level':'Year level/status','internet_access':df.columns[14],'housing_status':df.columns[43],'food_last':df.columns[44],'food_balanced':df.columns[45],'food_cut_meal':df.columns[46],'food_hungry':df.columns[49]}.items():
        x[key]=pd.to_numeric(df[col],errors='coerce') if key=='age' else df[col].astype('object')
    return x

@st.cache_data(show_spinner='Running significance tests...',max_entries=3)
def significance(x,y):
    rows=[]
    for col in x:
        s=x[col]
        if pd.api.types.is_numeric_dtype(s):
            aa=s[y==1].dropna();bb=s[y==0].dropna()
            if min(len(aa),len(bb))<5:continue
            stat,p=mannwhitneyu(aa,bb); effect=float(aa.median()-bb.median());method='Mann–Whitney U'
        else:
            tab=pd.crosstab(s.fillna('Missing'),y)
            if min(tab.shape)<2:continue
            stat,p,_,_=chi2_contingency(tab)
            effect=float(np.sqrt(stat/(tab.values.sum()*max(1,min(tab.shape)-1))));method='Chi-square'
        rows.append({'Feature':col,'Test':method,'Statistic':stat,'p_value':p,'Effect':effect})
    r=pd.DataFrame(rows)
    if len(r):r['FDR_q']=multipletests(r.p_value,method='fdr_bh')[1]
    return r.sort_values('FDR_q')

@st.cache_data(show_spinner='Training models and running cross-validation...',max_entries=2)
def train(x,y):
    numeric=x.select_dtypes(include='number').columns.tolist();categorical=[c for c in x if c not in numeric]
    prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),categorical)])
    models={'Logistic regression':LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42),'Random forest':RandomForestClassifier(n_estimators=200,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=-1)}
    xt,xv,yt,yv=train_test_split(x,y,test_size=.2,stratify=y,random_state=42)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    rows=[]
    for name,model in models.items():
        pipe=Pipeline([('preprocess',prep),('model',model)])
        scores=cross_validate(pipe,xt,yt,cv=cv,scoring={'roc_auc':'roc_auc','f1':'f1','recall':'recall','balanced_accuracy':'balanced_accuracy','average_precision':'average_precision'},n_jobs=1)
        row={'Model':name}
        for metric in ['roc_auc','f1','recall','balanced_accuracy','average_precision']:
            row[metric+'_mean']=scores['test_'+metric].mean();row[metric+'_sd']=scores['test_'+metric].std(ddof=1)
        rows.append(row)
    comparison=pd.DataFrame(rows).sort_values('roc_auc_mean',ascending=False)
    selected=comparison.iloc[0].Model
    best=Pipeline([('preprocess',prep),('model',models[selected])]).fit(xt,yt)
    prob=best.predict_proba(xv)[:,1];pred=(prob>=.5).astype(int)
    metrics={'ROC-AUC':roc_auc_score(yv,prob),'Average precision':average_precision_score(yv,prob),'Balanced accuracy':balanced_accuracy_score(yv,pred),'Recall':recall_score(yv,pred),'F1':f1_score(yv,pred),'Brier score':brier_score_loss(yv,prob)}
    imp=permutation_importance(best,xv,yv,scoring='roc_auc',n_repeats=5,random_state=42,n_jobs=1)
    importance=pd.DataFrame({'Feature':x.columns,'Mean AUC drop':imp.importances_mean}).sort_values('Mean AUC drop',ascending=False)
    fpr,tpr,_=roc_curve(yv,prob)
    return comparison,selected,metrics,importance,pd.DataFrame({'FPR':fpr,'TPR':tpr}),confusion_matrix(yv,pred)

def render(df,a):
    st.markdown('### Predictive modeling and evidence')
    valid=a.low_wellbeing.notna()
    x=features(df).loc[valid];y=a.loc[valid,'low_wellbeing'].astype(int)
    st.subheader('Exploratory statistical significance')
    if st.button('Run significance tests',key='sig') or st.session_state.get('sig_ready'):
        st.session_state.sig_ready=True
        result=significance(x,y)
        st.dataframe(result.round(4),hide_index=True,use_container_width=True)
    st.subheader('Classification and 5-fold cross-validation')
    if st.button('Train and evaluate models',type='primary',key='train') or st.session_state.get('train_ready'):
        st.session_state.train_ready=True
        comp,name,metrics,imp,roc,cm=train(x,y)
        st.success(f'Selected by training CV ROC-AUC: {name}')
        st.dataframe(comp.round(3),hide_index=True,use_container_width=True)
        st.dataframe(pd.DataFrame([metrics]).round(3),hide_index=True,use_container_width=True)
        st.plotly_chart(polish(px.line(roc,x='FPR',y='TPR',title='Held-out ROC curve')),use_container_width=True)
        st.plotly_chart(polish(px.imshow(cm,text_auto=True,color_continuous_scale='Teal',labels={'x':'Predicted','y':'Observed'},title='Held-out confusion matrix')),use_container_width=True)
        st.plotly_chart(polish(px.bar(imp.head(12).sort_values('Mean AUC drop'),x='Mean AUC drop',y='Feature',orientation='h',title='Predictor importance on held-out data'),520),use_container_width=True)
    st.warning('Prediction is of WHO-5 screening status, not a diagnosis. Threshold 0.5 is not a validated clinical action cutoff; never automate individual referrals.')

    # CHAPTER 3 ADDITION: Objective 2
    st.divider()
    st.markdown('### Research Objective 2 · Significant survey factors')
    st.caption('This section screens non-CRI survey predictors against the four research risk tiers. CRI component scores are shown separately because using them as predictors would leak the target definition into the model.')

    if st.button('Run Objective 2 feature selection',key='chapter3_feature_selection') or st.session_state.get('chapter3_feature_ready'):
        st.session_state.chapter3_feature_ready=True
        selected,components=chapter3_feature_screen(df)

        c1,c2,c3=st.columns(3)
        c1.metric('Candidate factors',f'{len(selected):,}')
        c2.metric('Selected factors',f'{int(selected.Selected_for_model.sum()):,}')
        c3.metric('Significant categorical factors',f'{int(selected.Significant_Chi2.sum()):,}')

        st.subheader('Feature-selection results')
        show=selected.copy()
        show['Information_Gain']=show['Information_Gain'].round(4)
        show['Chi2_p']=show['Chi2_p'].round(4)
        st.dataframe(show,hide_index=True,use_container_width=True)

        ranked=selected.sort_values('Information_Gain',ascending=False).head(12).sort_values('Information_Gain')
        st.plotly_chart(
            polish(px.bar(
                ranked,x='Information_Gain',y='Feature',orientation='h',
                title='Top survey factors by information gain'
            ),520),
            use_container_width=True
        )

        st.subheader('CRI component associations · descriptive only')
        st.dataframe(components.round(3),use_container_width=True)
        st.caption('For Objective 3, feature selection will be repeated inside each training fold to prevent leakage into validation folds, as required by Chapter 3.')



# CHAPTER 3 ADDITION: Objective 2 feature screening
def chapter3_candidate_features(df):
    """
    Candidate predictors that do not directly build the CRI target.
    The five CRI component scores are reported separately to avoid target leakage.
    """
    d=domains(df,reverse=True).copy()
    keep_domains=['academic','housing','autonomy','digital_self','self_care','coping','relationship_access','help_literacy']
    x=d[keep_domains].copy()

    x['age']=pd.to_numeric(df['Age'],errors='coerce')
    x['college']=df['College'].astype('object')
    x['year_level']=df['Year level/status'].astype('object')
    x['internet_access']=df.iloc[:,14].astype('object')
    x['housing_status']=df.iloc[:,43].astype('object')
    return x

@st.cache_data(show_spinner='Running Objective 2 feature screening...',max_entries=3)
def chapter3_feature_screen(df):
    r=chapter3_research_scores(df)
    valid=r.Research_scope & r.Risk_tier_quartile.notna()
    x=chapter3_candidate_features(df).loc[valid].copy()
    y=r.loc[valid,'Risk_tier_quartile'].astype(str)
    y_code=y.map({name:i for i,name in enumerate(CHAPTER3_TIERS)}).astype(int)

    rows=[]
    numeric=x.select_dtypes(include='number').columns.tolist()
    categorical=[col for col in x.columns if col not in numeric]

    # Information gain / mutual information for numeric predictors.
    for col in numeric:
        s=pd.to_numeric(x[col],errors='coerce')
        filled=s.fillna(s.median())
        if filled.nunique()<2:
            mi=0.0
        else:
            mi=float(mutual_info_classif(
                filled.to_frame(),y_code,
                discrete_features=False,random_state=42
            )[0])
        rows.append({
            'Feature':col,'Type':'Numeric',
            'Information_Gain':mi,'Chi2_p':np.nan,
            'Keep_after_corr':True
        })

    # Information gain + chi-square for categorical predictors.
    for col in categorical:
        s=x[col].fillna('Missing').astype(str)
        codes,_=pd.factorize(s,sort=True)
        if pd.Series(codes).nunique()<2:
            mi=0.0
        else:
            mi=float(mutual_info_classif(
                pd.DataFrame({col:codes}),y_code,
                discrete_features=True,random_state=42
            )[0])
        tab=pd.crosstab(s,y)
        p=np.nan
        if min(tab.shape)>=2:
            _,p,_,_=chi2_contingency(tab)
        rows.append({
            'Feature':col,'Type':'Categorical',
            'Information_Gain':mi,'Chi2_p':p,
            'Keep_after_corr':True
        })

    result=pd.DataFrame(rows)

    # Correlation filter for numeric predictors. If |r| > .85, keep the one
    # with the higher information gain.
    if len(numeric)>1:
        nx=x[numeric].apply(pd.to_numeric,errors='coerce')
        nx=nx.fillna(nx.median())
        corr=nx.corr(method='pearson').abs()
        mi_map=result.set_index('Feature')['Information_Gain'].to_dict()
        dropped=set()
        for i,left in enumerate(numeric):
            if left in dropped:
                continue
            for right in numeric[i+1:]:
                if right in dropped:
                    continue
                if corr.loc[left,right]>.85:
                    if mi_map.get(left,0)>=mi_map.get(right,0):
                        dropped.add(right)
                    else:
                        dropped.add(left)
                        break
        result.loc[result.Feature.isin(dropped),'Keep_after_corr']=False

    result['Significant_Chi2']=(
        result['Type'].eq('Categorical')
        & result['Chi2_p'].notna()
        & result['Chi2_p'].lt(.05)
    )
    result['Selected_for_model']=np.where(
        result['Type'].eq('Numeric'),
        result['Keep_after_corr'] & result['Information_Gain'].gt(0),
        result['Significant_Chi2']
    )
    result=result.sort_values(
        ['Selected_for_model','Information_Gain'],
        ascending=[False,False]
    ).reset_index(drop=True)

    # Descriptive association of CRI component scores. These are not used
    # as predictive inputs because they mathematically define the target.
    components=r.loc[valid,[
        'PHQ9_score','WHO5_score','MSPSS_mean',
        'Food_security_score','AUDIT9_score','CRI'
    ]].copy()
    component_corr=components.corr(method='spearman')['CRI'].drop('CRI')
    component_table=component_corr.rename('Spearman_with_CRI').to_frame()
    component_table['Absolute_association']=component_table.Spearman_with_CRI.abs()
    component_table=component_table.sort_values('Absolute_association',ascending=False)

    return result,component_table
