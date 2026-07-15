import sys,numpy as np
sys.path.insert(0,'.');sys.path.insert(0,'src')
from scipy.special import log_softmax
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from scripts._optimize_thresholds import coordinate_ascent
a=np.load('eval/probe/feats_25class_224.npz',allow_pickle=True);b=np.load('eval/probe/feats_25class_518.npz',allow_pickle=True)
X=np.concatenate([a['features'],b['features']],1).astype(np.float32);y=b['y'];split=b['split'].astype(str);fold=b['fold'];C=len(b['classes'])
tv=split=='train_val';ex=split=='external_test';Xtv,ytv,ftv=X[tv],y[tv],fold[tv];Xex,yex=X[ex],y[ex]
X5=b['features'].astype(np.float32);tvn=X5[tv]/(np.linalg.norm(X5[tv],axis=1,keepdims=True)+1e-8);exn=X5[ex]/(np.linalg.norm(X5[ex],axis=1,keepdims=True)+1e-8)
mx=np.zeros(len(exn))
for i in range(0,len(exn),256):mx[i:i+256]=(exn[i:i+256]@tvn.T).max(1)
clean=mx<=0.99
oof=np.zeros((len(ytv),C))
for fk in sorted(set(ftv[ftv>=0])):
 tr=ftv!=fk;te=ftv==fk;sc=StandardScaler().fit(Xtv[tr]);m=MLPClassifier((512,),max_iter=300,early_stopping=True,alpha=1e-3).fit(sc.transform(Xtv[tr]),ytv[tr]);oof[te]=m.predict_proba(sc.transform(Xtv[te]))
bias,_,_=coordinate_ascent(log_softmax(np.log(np.clip(oof,1e-9,1)),1),ytv,C,rounds=3)
sc=StandardScaler().fit(Xtv);m=MLPClassifier((512,),max_iter=300,early_stopping=True,alpha=1e-3).fit(sc.transform(Xtv),ytv)
p=(log_softmax(np.log(np.clip(m.predict_proba(sc.transform(Xex)),1e-9,1)),1)+bias).argmax(1)
print(f"224+518 concat MLP+bias CLEAN macroF1 {f1_score(yex[clean],p[clean],average='macro'):.4f} top1 {accuracy_score(yex[clean],p[clean]):.3f} (518-only 0.573)")
