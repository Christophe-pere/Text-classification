
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SVMSMOTE


class ImbalancedClass(object):
    
    def __init__(self, ratio=1, random_state=42, neighbors=5):
        #self.X = x_train
        #self.y = y_train
        self.random_state = random_state
        self.m_neighbors = neighbors
        self.ratio = ratio

    # ---- Over-sampling
    @classmethod
    def makeOverSamplesADASYN(self, X,y, ratio=1, random_state=42, m_neighbors=5):
        #input DataFrame
        #X →Independent Variable in DataFrame\
        #y →dependent Variable in Pandas DataFrame format
         
        sm = ADASYN(ratio=ratio, n_neighbors=m_neighbors, random_state=random_state)
        X, y = sm.fit_sample(X, y)
        return(X,y)

    @classmethod
    def makeOverSamplesSMOTE(self, X,y, ratio=1, random_state=42, m_neighbors=5):
     #input DataFrame
     #X →Independent Variable in DataFrame\
     #y →dependent Variable in Pandas DataFrame format
        
        sm = SMOTE( ratio=ratio, k_neighbors=m_neighbors, random_state=random_state)
        X, y = sm.fit_sample(X, y)
        return X,y

    @classmethod
    def makeOverSamplesKMeans(self, X,y, ratio=1, random_state=42, m_neighbors=5):
     #input DataFrame
     #X →Independent Variable in DataFrame\
     #y →dependent Variable in Pandas DataFrame format
        
        sm = SMOTE(ratio=ratio, k_neighbors=m_neighbors, random_state=random_state)
        X, y = sm.fit_sample(X, y)
        return X,y

    @classmethod
    def makeOverSamplesBorderline(self, X,y, ratio=1, random_state=42, m_neighbors=5):
     #input DataFrame
     #X →Independent Variable in DataFrame\
     #y →dependent Variable in Pandas DataFrame format
        
        sm = SMOTE(ratio=ratio, k_neighbors=m_neighbors, random_state=random_state)
        X, y = sm.fit_sample(X, y)
        return X,y

    @classmethod
    def makeOverSamplesRandom(self, X,y, ratio=1, random_state=42, m_neighbors=5):
     #input DataFrame
     #X →Independent Variable in DataFrame\
     #y →dependent Variable in Pandas DataFrame format
        
        sm = SMOTE(ratio=ratio, k_neighbors=m_neighbors, random_state=random_state)
        X, y = sm.fit_sample(X, y)
        return X,y

    def makeOverSamplesSMOTENC(self, X,y, ratio=1, random_state=42, m_neighbors=5):
     #input DataFrame
     #X →Independent Variable in DataFrame\
     #y →dependent Variable in Pandas DataFrame format
        
        sm = SMOTE(ratio=ratio, k_neighbors=m_neighbors, random_state=random_state)
        X, y = sm.fit_sample(X, y)
        return X,y

    @classmethod
    def makeOverSamplesSVMSMOTE(self, X,y, ratio=1, random_state=42, m_neighbors=5):
     #input DataFrame
     #X →Independent Variable in DataFrame\
     #y →dependent Variable in Pandas DataFrame format
        
        sm = SMOTE(ratio=ratio, k_neighbors=m_neighbors, random_state=random_state)
        X, y = sm.fit_sample(X, y)
        return X,y

