import io
import numpy as np
import pandas as pd
import streamlit as st

BLOCKS={'academic':range(18,34),'housing':range(37,43),'social_support':range(52,70),'autonomy':range(70,78),'digital_self':range(78,85),'self_care':range(85,89),'coping':range(89,97),'relationship_access':range(111,117),'help_literacy':range(132,140)}
REVERSE={28,29,32,33,69,78,79,80,81,83,84,94,95,96}
MIN_CELL=10

def first_num(s):
    return pd.to_numeric(s.astype('string').str.extract(r'^\s*(\d+)',expand=False),errors='coerce')

@st.cache_data(show_spinner=False,max_entries=3)
def load_data(raw):
    df=pd.read_csv(io.BytesIO(raw),low_memory=False)
    if df.shape[1]<142 or 'College' not in df or not all('World Health Organization-Five' in c for c in df.columns[118:123]):
        raise ValueError('The CSV does not match the expected ADSSU-CWI survey schema.')
    return df

@st.cache_data(show_spinner=False,max_entries=3)
def enrich(df):
    who=df.iloc[:,118:123].apply(first_num).where(lambda z:z.ge(0) & z.le(5))
    phq=df.iloc[:,123:132].apply(first_num).where(lambda z:z.ge(0) & z.le(3))
    a=pd.DataFrame(index=df.index)
    a['College']=df['College'].fillna('Unspecified').astype(str)
    a['Year level/status']=df['Year level/status'].fillna('Unspecified').astype(str)
    a['WHO5_score']=who.sum(axis=1,min_count=5)*4
    a['PHQ9_score']=phq.sum(axis=1,min_count=9)
    a['low_wellbeing']=a.WHO5_score.lt(50).where(a.WHO5_score.notna())
    a['PHQ9_band']=pd.cut(a.PHQ9_score,[-1,4,9,14,19,27],labels=['Minimal','Mild','Moderate','Moderately severe','Severe'])
    return a

@st.cache_data(show_spinner=False,max_entries=3)
def domains(df,reverse=False):
    x=pd.DataFrame(index=df.index)
    for name,indices in BLOCKS.items():
        z=df.iloc[:,list(indices)].apply(first_num)
        if reverse:
            for idx in indices:
                if idx in REVERSE:
                    z.iloc[:,list(indices).index(idx)]=6-z.iloc[:,list(indices).index(idx)]
        x[name]=z.mean(axis=1)
    return x

@st.cache_data(show_spinner=False,max_entries=3)
def services(df):
    col=next((c for c in df.columns if c.startswith('SERVICES_NEEDED')),df.columns[141])
    tokens=df[col].fillna('').astype(str).apply(lambda s:set(t.strip() for t in s.split(',') if t.strip()))
    names=sorted({t for row in tokens for t in row})
    return pd.DataFrame({n:tokens.apply(lambda row:n in row).astype(int) for n in names},index=df.index)

def protected_table(counts,denominators=None):
    """Conservative suppression: hide cells below 10, including zeros, and hide small groups."""
    x=counts.copy().astype(float)
    if denominators is not None:
        x=x.loc[denominators.reindex(x.index).fillna(0)>=MIN_CELL]
    return x.mask(x<MIN_CELL)

def download_csv(df,name,label):
    st.download_button(label,df.to_csv(index=True).encode('utf-8'),file_name=name,mime='text/csv')


# CHAPTER 3 ADDITION: OSAS-adapted MSPSS scoring
def chapter3_mspss_score(df):
    """Calculate the 12-item MSPSS mean using the OSAS 1-5 response scale."""
    mspss=df.iloc[:,52:64].apply(first_num)
    mspss=mspss.where(mspss.ge(1) & mspss.le(5))
    missing=mspss.isna().sum(axis=1)
    row_median=mspss.median(axis=1)
    filled=mspss.T.fillna(row_median).T
    score=filled.mean(axis=1)
    return score.where(missing<=1)
