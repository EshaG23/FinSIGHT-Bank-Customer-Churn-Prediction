"""Regenerate actual predictions while retaining each input customer's ID.
Saved DT/RF/LR/SVM parameters are used for inference. KNN is rebuilt from the
existing validation-selected settings; hyperparameters are NOT reselected.
Only load the trusted model artifacts supplied with this project.
"""
from pathlib import Path
import argparse, ast, hashlib, json, pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from compare_models import canonical_split, require, SPECS, compute_metrics

SCALE=['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']

def encoded_inputs(root):
    # Identity is retained BEFORE split/encoding and before any prediction.
    raw=pd.read_csv(root/'data/Churn_Modelling.csv').drop_duplicates()
    require(raw.CustomerId.is_unique,'Duplicate source customer IDs')
    raw=raw.set_index('CustomerId',drop=True)
    X=raw.drop(columns=['RowNumber','Surname','Exited']);y=raw.Exited
    xa,xb,ya,yb=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    train=pd.get_dummies(xa,columns=['Geography','Gender']).astype(float)
    test=pd.get_dummies(xb,columns=['Geography','Gender']).astype(float).reindex(columns=train.columns,fill_value=0)
    require(test.index.equals(yb.index),'Test input IDs do not match labels')
    # Verify rebuilt features against shared CSVs, without using those CSVs as
    # inference inputs or assigning IDs to an old prediction array.
    manifest,dataset_hash=canonical_split(root)
    require(np.array_equal(test.index.to_numpy(),manifest.CustomerId.to_numpy()),'Split mismatch')
    return train,test,ya.to_numpy(),yb.to_numpy(),dataset_hash

def tree_scores(root,X):
    result=np.zeros(len(X))
    for i,row in enumerate(X):
        node=root
        while 'feature' in node:
            node=node['left'] if row[node['feature']]<=node['cut'] else node['right']
        result[i]=node['prob']
    return result

def knn_scores(root,train,test,y_train):
    settings=pd.read_csv(root/'outputs/knn_improved/selected_settings.csv').iloc[0]
    # Reuse the exact from-scratch class/functions; skip top-level tuning code.
    source=ast.parse((root/'src/KNN.py').read_text())
    nodes=[n for n in source.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['KNN','prepare_features','balance_training']]
    namespace={'np':np,'pd':pd,'scale_columns':SCALE,'StandardScaler':StandardScaler,'MinMaxScaler':MinMaxScaler}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_KNN_definitions','exec'),namespace)
    columns=train.columns.tolist() if settings['Features']=='all' else ['CreditScore','Age','Balance','NumOfProducts','IsActiveMember','Geography_France','Geography_Germany','Geography_Spain','Gender_Female','Gender_Male']
    a,b=namespace['prepare_features'](train,test,str(settings['Scaling']),columns)
    a,labels=namespace['balance_training'](a,y_train,str(settings['Balance']))
    model=namespace['KNN'](k=int(settings['K']),voting=str(settings['Voting']))
    model.fit(a,labels)
    return model.predict_proba(b),float(settings['Threshold']),{'inference':'Refitted KNN using existing validation-selected settings','training_rows_after_balancing':len(labels)}

def regenerate(root):
    root=Path(root).resolve();train,test,yt,y,dataset_hash=encoded_inputs(root)
    ids=test.index.to_numpy(copy=True);pending=[];audit=[]
    for name,spec in SPECS.items():
        print('Regenerating:',name,flush=True)
        details={}
        if name=='KNN':scores,threshold,details=knn_scores(root,train,test,yt)
        elif name in ['Decision Tree','Random Forest']:
            folder=Path(spec[0]).parent
            with open(root/'outputs'/folder/'model.pkl','rb') as f:model=pickle.load(f)
            require(set(model['feature_columns'])==set(test.columns),name+': feature mismatch')
            values=test[model['feature_columns']].to_numpy(dtype=float)
            scores=np.mean([tree_scores(tree,values) for tree in model['tree_roots']],axis=0)
            threshold=float(model['threshold']);details={'inference':'Saved tree roots','trees':len(model['tree_roots'])}
        elif name=='Logistic Regression':
            model=joblib.load(root/'outputs/logistic_results/logistic_model.pkl')
            values=test[model['feature_columns']].copy()
            # Only use stored StandardScaler numeric parameters, avoiding version-
            # dependent transformer methods from the uploaded pickle.
            scaler=model['scaler'];cols=model['scale_columns']
            values[cols]=(values[cols].to_numpy()-scaler.mean_)/scaler.scale_
            z=np.clip(values.to_numpy()@model['weights']+model['bias'],-500,500)
            scores=1/(1+np.exp(-z));threshold=float(model['threshold'])
            details={'inference':'Saved weights, bias and training scaler statistics'}
        else:
            model=json.loads((root/'outputs/svm_scratch/model_parameters.json').read_text())
            values=test[model['feature_names']].to_numpy(dtype=float)
            values=(values-np.array(model['scaler_mean']))/np.array(model['scaler_scale'])
            scores=values@np.array(model['weights'])+model['bias'];threshold=float(model['threshold'])
            details={'inference':'Saved SVM weights, bias and training scaler statistics'}
        require(len(scores)==len(test)==len(ids)==2000,name+': row count mismatch')
        predicted=(scores>=threshold).astype(int)
        export=pd.DataFrame({'CustomerId':ids,spec[1][0]:y,spec[1][1]:predicted,spec[1][2]:scores})
        old=pd.read_csv(root/'outputs'/spec[0])
        unchanged_labels=np.array_equal(old[spec[1][0]].to_numpy(),y)
        unchanged_preds=np.array_equal(old[spec[1][1]].to_numpy(),predicted)
        difference=float(np.max(np.abs(old[spec[1][2]].to_numpy()-scores)))
        require(unchanged_labels,name+': prior labels differ')
        require(unchanged_preds and np.allclose(old[spec[1][2]].to_numpy(),scores,atol=1e-10,rtol=1e-9),name+': regenerated results differ; refusing to overwrite without review')
        pending.append((root/'outputs'/spec[0],export))
        audit.append({'Model':name,'Test_rows':2000,'Identity':'Retained from raw input index before split/encoding/inference',
            'Predictions_recomputed':True,'Original_predictions_match':unchanged_preds,
            'Max_score_difference':difference,'Threshold':threshold,**details})
    # Do not publish partial regenerated files if any model failed verification.
    for path,export in pending:export.to_csv(path,index=False)
    dest=root/'outputs/identity_verification';dest.mkdir(exist_ok=True)
    pd.DataFrame(audit).to_csv(dest/'regeneration_audit.csv',index=False)
    pd.DataFrame({'CustomerId':ids,'Exited':y}).to_csv(dest/'test_manifest.csv',index=False)
    (dest/'provenance.json').write_text(json.dumps({'dataset_sha256':dataset_hash,
        'seed':42,'test_size':.2,'stratify':'Exited','test_customers':2000,
        'identity_retention':'CustomerId index retained from raw source before split and used with freshly generated predictions',
        'hyperparameter_selection':'Existing validation-selected settings preserved; no test-set tuning',
        'models':audit},indent=2))
    print('All 5 prediction exports regenerated with retained customer IDs.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
    regenerate(p.parse_args().project)
