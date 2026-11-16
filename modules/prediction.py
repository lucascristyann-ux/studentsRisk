import gc
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#0F766E", "#2563EB", "#D97706", "#7C3AED", "#0891B2"]
from scipy.stats import mannwhitneyu,chi2_contingency,chi2
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB,ComplementNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
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
    st.markdown('### Research prediction results')
    st.caption('The research results load automatically from the uploaded dataset. Detailed statistical tables remain available below each graph.')

    # Objective 2: feature selection
    st.markdown('### Significant survey factors')
    st.caption('Higher information gain means a survey factor carries more information about the final four research risk tiers.')

    try:
        selected,components=chapter3_feature_screen(df)
    except Exception as e:
        st.error(f'Feature selection could not run: {e}')
    else:
        c1,c2,c3=st.columns(3)
        c1.metric('Candidate factors',f'{len(selected):,}')
        c2.metric('Selected factors',f'{int(selected.Selected_for_model.sum()):,}')
        c3.metric('Significant categorical factors',f'{int(selected.Significant_Chi2.sum()):,}')

        ranked=selected[selected.Selected_for_model].copy()
        if ranked.empty:
            ranked=selected.head(10).copy()
        ranked=ranked.sort_values('Information_Gain',ascending=True).tail(12)

        fig=px.bar(
            ranked,
            x='Information_Gain',
            y='Feature',
            orientation='h',
            color='Information_Gain',
            color_continuous_scale=['#EDE4D9','#D7AE78','#91A66B'],
            title='Selected survey predictors by information gain',
            text='Information_Gain'
        )
        fig.update_traces(texttemplate='%{text:.3f}',textposition='outside',cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(polish(fig,500),use_container_width=True)

        with st.expander('View detailed feature-selection results'):
            show=selected.copy()
            show['Information_Gain']=show['Information_Gain'].round(4)
            show['Chi2_p']=show['Chi2_p'].round(4)
            st.dataframe(show,hide_index=True,use_container_width=True)
            st.subheader('CRI component associations · descriptive only')
            st.dataframe(components.round(3),use_container_width=True)

    st.divider()

    # Preprocessing diagnostics
    st.markdown('### Preprocessing diagnostics')
    try:
        vif,outliers,md_threshold,pca_table=chapter3_preprocessing_diagnostics(df)
    except Exception as e:
        st.error(f'Preprocessing diagnostics could not run: {e}')
    else:
        c1,c2=st.columns(2)
        c1.metric('Predictors with VIF > 10',f'{int(vif.VIF_above_10.sum()):,}')
        c2.metric('Mahalanobis outlier flags',f'{int(outliers.Flag_above_99pct_chi2.sum()):,}')

        vif_plot=vif.sort_values('VIF',ascending=True)
        fig=px.bar(
            vif_plot,x='VIF',y='Feature',orientation='h',
            color='VIF',
            color_continuous_scale=['#DCE6C8','#D9A36D','#C97167'],
            title='Multicollinearity diagnostic · Variance Inflation Factor',
            text='VIF'
        )
        fig.add_vline(x=10,line_dash='dash',line_color='#C97167',annotation_text='VIF = 10')
        fig.update_traces(texttemplate='%{text:.2f}',textposition='outside',cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(polish(fig,440),use_container_width=True)

        with st.expander('View preprocessing diagnostic tables'):
            st.dataframe(vif.round(3),hide_index=True,use_container_width=True)
            st.caption(f'Mahalanobis review threshold: {md_threshold:.3f}')
            st.dataframe(outliers.head(50).round(3),hide_index=True,use_container_width=True)
            if not pca_table.empty:
                st.dataframe(pca_table.round(2),hide_index=True,use_container_width=True)

    st.divider()

    # Objective 3: model validation
    st.markdown('### Four-model validation')
    st.caption('Decision Tree, Naive Bayes, Random Forest and SVM are compared using stratified 10-fold cross-validation with training-fold-only feature selection and SMOTE.')

    try:
        comp,best,report,cm3,mcnemar_table,stability,tuning=chapter3_validate_models(df)
    except Exception as e:
        st.error(f'Model validation could not run: {e}')
    else:
        st.session_state.chapter3_best_model=best
        st.session_state.chapter3_model_comparison=comp.to_dict('records')

        best_row=comp.loc[comp.Model.eq(best)].iloc[0]
        c1,c2,c3=st.columns(3)
        c1.metric('Best model',best)
        c2.metric('Best macro F1',f"{best_row['F1_macro']:.3f}")
        c3.metric('Best macro AUC',f"{best_row['AUC_OVR_macro']:.3f}")

        long=comp.melt(
            id_vars='Model',
            value_vars=['Accuracy','F1_macro','AUC_OVR_macro'],
            var_name='Metric',value_name='Score'
        )
        labels={'F1_macro':'Macro F1','AUC_OVR_macro':'Macro AUC','Accuracy':'Accuracy'}
        long['Metric']=long['Metric'].map(labels)

        fig=px.bar(
            long,x='Model',y='Score',color='Metric',barmode='group',
            title='10-fold classification model comparison',
            category_orders={'Model':comp.Model.tolist()},
            color_discrete_map={
                'Accuracy':'#91A66B',
                'Macro F1':'#D9915B',
                'Macro AUC':'#8F83B8'
            }
        )
        fig.update_yaxes(range=[0,1])
        st.plotly_chart(polish(fig,470),use_container_width=True)

        c1,c2=st.columns([1.05,1])
        with c1:
            tier_report=report.loc[
                [x for x in CHAPTER3_TIERS if x in report.index],
                ['precision','recall','f1-score','support']
            ].reset_index().rename(columns={'index':'Risk tier'})
            fig=px.bar(
                tier_report,x='Risk tier',y='recall',
                color='Risk tier',text='recall',
                title=f'{best} recall by research risk tier',
                category_orders={'Risk tier':CHAPTER3_TIERS},
                color_discrete_map={
                    'Low Risk':'#91A66B',
                    'Moderate Risk':'#D8B56A',
                    'High Risk':'#D9915B',
                    'Critical Risk':'#C97167'
                }
            )
            fig.update_traces(texttemplate='%{text:.2f}',textposition='outside',cliponaxis=False)
            fig.update_yaxes(range=[0,1])
            st.plotly_chart(polish(fig,430),use_container_width=True)

        with c2:
            fig=px.imshow(
                cm3,
                text_auto=True,
                color_continuous_scale=['#F7F1EA','#DFA06F','#8F6C56'],
                labels={'x':'Predicted tier','y':'Observed tier','color':'Count'},
                title=f'{best} out-of-fold confusion matrix'
            )
            st.plotly_chart(polish(fig,430),use_container_width=True)

        with st.expander('View model tables and statistical comparisons'):
            st.dataframe(comp.round(3),hide_index=True,use_container_width=True)
            st.subheader('Per-tier performance of best model')
            st.dataframe(tier_report.round(3),hide_index=True,use_container_width=True)
            st.subheader('Pairwise McNemar tests')
            st.dataframe(mcnemar_table.round(4),hide_index=True,use_container_width=True)
            st.subheader('Feature-selection stability across 10 folds')
            st.dataframe(stability.round(1),hide_index=True,use_container_width=True)
            st.subheader('Fold-by-fold tuning details')
            st.dataframe(tuning,hide_index=True,use_container_width=True)

        st.caption('These are aggregate research-validation results, not automated clinical referral decisions.')

    st.divider()

    # The original exploratory WHO-5 functions remain preserved in this module,
    # but are not executed automatically because the research validation above
    # is the primary result and duplicate model fitting can exceed cloud limits.
    st.caption('Supplementary exploratory WHO-5 routines remain preserved in the source code and are not recomputed on this results page to keep the dashboard responsive.')

    st.warning('Research classifications and model outputs are not diagnoses and must not automate individual referrals.')


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

@st.cache_data(show_spinner='Running feature screening...',max_entries=3)
def chapter3_feature_screen(df):
    r=chapter3_research_scores(df)
    valid=r.Research_scope & r.Risk_tier_final.notna()
    x=chapter3_candidate_features(df).loc[valid].copy()
    y=r.loc[valid,'Risk_tier_final'].astype(str)
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


# CHAPTER 3 ADDITION: Objective 3 four-model validation
def _chapter3_fold_transform(x_train,x_test,y_train):
    """
    Fit correlation, chi-square, information-gain selection, imputation,
    encoding and Min-Max scaling using the training fold only.
    """
    numeric=x_train.select_dtypes(include='number').columns.tolist()
    categorical=[c for c in x_train.columns if c not in numeric]
    y_code=y_train.map({name:i for i,name in enumerate(CHAPTER3_TIERS)}).astype(int)

    # Numeric redundancy filter, using training-fold information gain to decide
    # which member of a highly correlated pair is retained.
    num_mi={}
    if numeric:
        nx=x_train[numeric].apply(pd.to_numeric,errors='coerce')
        nx=nx.fillna(nx.median())
        for col in numeric:
            if nx[col].nunique()<2:
                num_mi[col]=0.0
            else:
                num_mi[col]=float(mutual_info_classif(
                    nx[[col]],y_code,discrete_features=False,random_state=42
                )[0])
        corr=nx.corr(method='pearson').abs()
        dropped=set()
        for i,left in enumerate(numeric):
            if left in dropped:
                continue
            for right in numeric[i+1:]:
                if right in dropped:
                    continue
                if corr.loc[left,right]>.85:
                    if num_mi.get(left,0)>=num_mi.get(right,0):
                        dropped.add(right)
                    else:
                        dropped.add(left)
                        break
        numeric=[c for c in numeric if c not in dropped]

    # Chi-square selection for categorical predictors using training data only.
    selected_cat=[]
    for col in categorical:
        s=x_train[col].fillna('Missing').astype(str)
        tab=pd.crosstab(s,y_train)
        if min(tab.shape)<2:
            continue
        _,p,_,_=chi2_contingency(tab)
        if p<.05:
            selected_cat.append(col)

    # Keep numeric predictors after correlation filtering. If no categorical
    # variable passes chi-square, the model still has the numeric candidates.
    selected_raw=numeric+selected_cat
    if not selected_raw:
        selected_raw=x_train.columns.tolist()[:1]

    transformers=[]
    selected_num=[c for c in selected_raw if c in numeric]
    selected_cat=[c for c in selected_raw if c not in selected_num]
    if selected_num:
        transformers.append((
            'num',
            Pipeline([
                ('impute',SimpleImputer(strategy='median')),
                ('scale',MinMaxScaler())
            ]),
            selected_num
        ))
    if selected_cat:
        transformers.append((
            'cat',
            Pipeline([
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
            ]),
            selected_cat
        ))

    prep=ColumnTransformer(transformers,remainder='drop')
    xtr=prep.fit_transform(x_train[selected_raw])
    xte=prep.transform(x_test[selected_raw])
    names=np.asarray(prep.get_feature_names_out(),dtype=object)

    # Information-gain filtering is also fitted only on the training fold.
    if xtr.shape[1]>1:
        discrete=np.array([str(n).startswith('cat__') for n in names])
        mi=mutual_info_classif(
            xtr,y_code,discrete_features=discrete,random_state=42
        )
        keep=mi>0
        if keep.sum()<min(5,len(mi)):
            order=np.argsort(mi)[::-1]
            keep=np.zeros(len(mi),dtype=bool)
            keep[order[:min(5,len(mi))]]=True
        xtr=xtr[:,keep]
        xte=xte[:,keep]
        names=names[keep]

    return xtr,xte,selected_raw,names


@st.cache_data(show_spinner='Running stratified 10-fold model validation...',max_entries=2)
def chapter3_validate_models(df):
    r=chapter3_research_scores(df)
    valid=r.Research_scope & r.Risk_tier_final.notna()
    x=chapter3_candidate_features(df).loc[valid].reset_index(drop=True)
    y=r.loc[valid,'Risk_tier_final'].astype(str).reset_index(drop=True)

    if len(y)<40 or y.value_counts().min()<10:
        raise ValueError('Not enough valid cases in every risk tier for stratified 10-fold validation.')

    outer=StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
    model_specs={
        'Decision Tree (entropy)':(
            DecisionTreeClassifier(criterion='entropy',random_state=42),
            {'model__max_depth':[4,8,None]}
        ),
        'Naive Bayes':(
            GaussianNB(),
            [
                {
                    'model':[GaussianNB()],
                    'model__var_smoothing':[1e-9,1e-8]
                },
                {
                    'model':[ComplementNB()],
                    'model__alpha':[0.5,1.0]
                }
            ]
        ),
        'Random Forest':(
            RandomForestClassifier(random_state=42,n_jobs=1),
            {'model__n_estimators':[100,500],'model__max_features':['sqrt','log2']}
        ),
        'SVM':(
            SVC(probability=True,random_state=42,cache_size=50),
            [
                {'model__kernel':['linear'],'model__C':[1,2]},
                {'model__kernel':['rbf'],'model__C':[1,2],'model__gamma':['scale']},
                {'model__kernel':['poly'],'model__C':[1],'model__degree':[3]}
            ]
        )
    }

    predictions={name:np.empty(len(y),dtype=object) for name in model_specs}
    probabilities={name:np.zeros((len(y),len(CHAPTER3_TIERS)),dtype=float) for name in model_specs}
    feature_counts={}
    tuning_rows=[]

    for fold,(train_idx,test_idx) in enumerate(outer.split(x,y),start=1):
        xtr_raw=x.iloc[train_idx].copy()
        xte_raw=x.iloc[test_idx].copy()
        ytr=y.iloc[train_idx].copy()
        yte=y.iloc[test_idx].copy()

        xtr,xte,selected_raw,selected_names=_chapter3_fold_transform(
            xtr_raw,xte_raw,ytr
        )
        for feature in selected_raw:
            feature_counts[feature]=feature_counts.get(feature,0)+1

        min_class=int(ytr.value_counts().min())
        smote_k=max(1,min(5,min_class-1))
        inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=fold)

        for name,(model,param_grid) in model_specs.items():
            pipe=ImbPipeline([
                ('smote',SMOTE(random_state=42+fold,k_neighbors=smote_k)),
                ('model',model)
            ])
            search=GridSearchCV(
                pipe,param_grid=param_grid,scoring='f1_macro',
                cv=inner,n_jobs=1,refit=True
            )
            search.fit(xtr,ytr)
            pred=search.predict(xte)
            raw_prob=search.predict_proba(xte)

            predictions[name][test_idx]=pred
            aligned=np.zeros((len(test_idx),len(CHAPTER3_TIERS)),dtype=float)
            model_classes=list(search.best_estimator_.named_steps['model'].classes_)
            for source_i,cls in enumerate(model_classes):
                aligned[:,CHAPTER3_TIERS.index(cls)]=raw_prob[:,source_i]
            probabilities[name][test_idx,:]=aligned

            tuning_rows.append({
                'Fold':fold,
                'Model':name,
                'Best parameters':str(search.best_params_),
                'Selected raw factors':len(selected_raw),
                'Selected encoded features':len(selected_names)
            })
            del search,pipe,raw_prob,aligned
            gc.collect()

    y_int=y.map({name:i for i,name in enumerate(CHAPTER3_TIERS)}).astype(int)
    metric_rows=[]
    reports={}
    cms={}
    for name in model_specs:
        pred=pd.Series(predictions[name])
        prob=probabilities[name]
        auc=roc_auc_score(
            y_int,prob,multi_class='ovr',average='macro',
            labels=list(range(len(CHAPTER3_TIERS)))
        )
        metric_rows.append({
            'Model':name,
            'Accuracy':accuracy_score(y,pred),
            'Precision_macro':precision_score(y,pred,average='macro',zero_division=0),
            'Recall_macro':recall_score(y,pred,average='macro',zero_division=0),
            'F1_macro':f1_score(y,pred,average='macro',zero_division=0),
            'AUC_OVR_macro':auc
        })
        rep=classification_report(
            y,pred,labels=CHAPTER3_TIERS,
            output_dict=True,zero_division=0
        )
        reports[name]=pd.DataFrame(rep).T
        cms[name]=pd.DataFrame(
            confusion_matrix(y,pred,labels=CHAPTER3_TIERS),
            index=CHAPTER3_TIERS,columns=CHAPTER3_TIERS
        )

    comparison=pd.DataFrame(metric_rows).sort_values(
        ['F1_macro','AUC_OVR_macro'],ascending=False
    ).reset_index(drop=True)
    best_name=comparison.iloc[0].Model

    # Pairwise McNemar tests compare whether model correctness differs.
    pair_rows=[]
    model_names=list(model_specs)
    for i,left in enumerate(model_names):
        for right in model_names[i+1:]:
            left_ok=predictions[left]==y.to_numpy()
            right_ok=predictions[right]==y.to_numpy()
            table=[
                [int((left_ok & right_ok).sum()),int((left_ok & ~right_ok).sum())],
                [int((~left_ok & right_ok).sum()),int((~left_ok & ~right_ok).sum())]
            ]
            test=mcnemar(table,exact=False,correction=True)
            pair_rows.append({
                'Model A':left,'Model B':right,
                'McNemar statistic':float(test.statistic),
                'p_value':float(test.pvalue)
            })

    stability=pd.DataFrame([
        {'Feature':feature,'Selected_folds':count,'Selection_percent':100*count/10}
        for feature,count in feature_counts.items()
    ]).sort_values(['Selected_folds','Feature'],ascending=[False,True])

    return (
        comparison,
        best_name,
        reports[best_name],
        cms[best_name],
        pd.DataFrame(pair_rows),
        stability,
        pd.DataFrame(tuning_rows)
    )


# CHAPTER 3 ADDITION: preprocessing diagnostics
@st.cache_data(show_spinner='Running preprocessing diagnostics...',max_entries=3)
def chapter3_preprocessing_diagnostics(df):
    r=chapter3_research_scores(df)
    valid=r.Research_scope & r.Risk_tier_final.notna()
    x=chapter3_candidate_features(df).loc[valid].copy()
    numeric=x.select_dtypes(include='number').columns.tolist()

    if not numeric:
        raise ValueError('No numeric candidate predictors are available for diagnostics.')

    z=x[numeric].apply(pd.to_numeric,errors='coerce')
    z=z.fillna(z.median())
    z=z.loc[:,z.nunique()>1]

    if z.shape[1]<2:
        raise ValueError('At least two non-constant numeric predictors are required.')

    scaled=StandardScaler().fit_transform(z)

    vif=pd.DataFrame({
        'Feature':z.columns,
        'VIF':[variance_inflation_factor(scaled,i) for i in range(scaled.shape[1])]
    }).sort_values('VIF',ascending=False)
    vif['VIF_above_10']=vif.VIF.gt(10)

    center=scaled.mean(axis=0)
    cov=np.cov(scaled,rowvar=False)
    inv=np.linalg.pinv(cov)
    diff=scaled-center
    md2=np.einsum('ij,jk,ik->i',diff,inv,diff)
    threshold=float(chi2.ppf(.99,df=scaled.shape[1]))
    outliers=pd.DataFrame({
        'Dataset_row':z.index,
        'Mahalanobis_D2':md2,
        'Flag_above_99pct_chi2':md2>threshold
    }).sort_values('Mahalanobis_D2',ascending=False)

    pca_table=pd.DataFrame()
    if vif['VIF_above_10'].any():
        pca=PCA().fit(scaled)
        pca_table=pd.DataFrame({
            'Component':np.arange(1,len(pca.explained_variance_ratio_)+1),
            'Explained_variance_percent':100*pca.explained_variance_ratio_,
            'Cumulative_percent':100*np.cumsum(pca.explained_variance_ratio_)
        })

    return vif,outliers,threshold,pca_table
