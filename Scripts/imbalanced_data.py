




# ---- Over-sampling
def makeOverSamplesADASYN(X,y, ratio=1, random_state=42, m_neighbors=5):
    #input DataFrame
    #X →Independent Variable in DataFrame\
    #y →dependent Variable in Pandas DataFrame format
    from imblearn.over_sampling import ADASYN 
    sm = ADASYN(ratio=ratio, n_neighbors=m_neighbors, random_state=random_state)
    X, y = sm.fit_sample(X, y)
    return(X,y)

def makeOverSamplesSMOTE(X,y, ratio=1, random_state=42, m_neighbors=5):
 #input DataFrame
 #X →Independent Variable in DataFrame\
 #y →dependent Variable in Pandas DataFrame format
    from imblearn.over_sampling import SMOTE
    sm = SMOTE( ratio=ratio, k_neighbors=m_neighbors, random_state=random_state)
    X, y = sm.fit_sample(X, y)
    return X,y

def makeOverSamplesKMeans(X,y, ratio=1, random_state=42, m_neighbors=5):
 #input DataFrame
 #X →Independent Variable in DataFrame\
 #y →dependent Variable in Pandas DataFrame format
    from imblearn.over_sampling import KMeansSMOTE
    sm = SMOTE(ratio=ratio, k_neighbors=m_neighbors, random_state=random_state)
    X, y = sm.fit_sample(X, y)
    return X,y

def makeOverSamplesBorderline(X,y, ratio=1, random_state=42, m_neighbors=5):
 #input DataFrame
 #X →Independent Variable in DataFrame\
 #y →dependent Variable in Pandas DataFrame format
    from imblearn.over_sampling import BorderlineSMOTE
    sm = SMOTE(ratio=ratio, k_neighbors=m_neighbors, random_state=random_state)
    X, y = sm.fit_sample(X, y)
    return X,y

def makeOverSamplesRandom(X,y, ratio=1, random_state=42, m_neighbors=5):
 #input DataFrame
 #X →Independent Variable in DataFrame\
 #y →dependent Variable in Pandas DataFrame format
    from imblearn.over_sampling import RandomOverSampler
    sm = SMOTE(ratio=ratio, k_neighbors=m_neighbors, random_state=random_state)
    X, y = sm.fit_sample(X, y)
    return X,y

def makeOverSamplesSMOTENC(X,y, ratio=1, random_state=42, m_neighbors=5):
 #input DataFrame
 #X →Independent Variable in DataFrame\
 #y →dependent Variable in Pandas DataFrame format
    from imblearn.over_sampling import SMOTENC
    sm = SMOTE(ratio=ratio, k_neighbors=m_neighbors, random_state=random_state)
    X, y = sm.fit_sample(X, y)
    return X,y


def makeOverSamplesSVMSMOTE(X,y, ratio=1, random_state=42, m_neighbors=5):
 #input DataFrame
 #X →Independent Variable in DataFrame\
 #y →dependent Variable in Pandas DataFrame format
    from imblearn.over_sampling import SVMSMOTE
    sm = SMOTE(ratio=ratio, k_neighbors=m_neighbors, random_state=random_state)
    X, y = sm.fit_sample(X, y)
    return X,y

