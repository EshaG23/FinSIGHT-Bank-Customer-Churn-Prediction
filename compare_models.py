"""Compare all five supplied FinSIGHT algorithms, validating alignment first.
Run from the project root: python compare_models.py --allow-legacy-order
Without the flag, CustomerId must be present in EVERY prediction export.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import html
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay)

METRICS=['Accuracy','Precision','Recall','F1','ROC_AUC']
SPECS={
 'KNN':('knn_improved/test_predictions.csv', ['Actual','Predicted','Churn_Score'],'probability',
         'knn_improved/selected_settings.csv', 'knn_improved/final_test_metrics.csv'),
 'Logistic Regression':('logistic_results/logistic_predictions.csv',
         ['Actual','Predicted','Churn_Probability'],'probability',
         'logistic_results/logistic_validation_results.csv','logistic_results/logistic_test_metrics.csv'),
 'Decision Tree':('decision_tree_scratch_entropy/predictions.csv',
         ['y_true','y_pred','y_score'],'probability','decision_tree_scratch_entropy/test_metrics.csv',
         'decision_tree_scratch_entropy/test_metrics.csv'),
 'Random Forest':('random_forest_scratch_entropy/predictions.csv',
         ['y_true','y_pred','y_score'],'probability','random_forest_scratch_entropy/test_metrics.csv',
         'random_forest_scratch_entropy/test_metrics.csv'),
 'SVM':('svm_scratch/predictions.csv',['Actual','Predicted','DecisionScore'],
         'decision margin','svm_scratch/test_metrics.csv','svm_scratch/test_metrics.csv')}

def require(condition,message):
    if not condition: raise ValueError(message)

def same_frame(actual,expected,label):
    require(list(actual.columns)==list(expected.columns),label+': column order differs')
    require(actual.shape==expected.shape,label+': shape differs')
    require(np.allclose(actual.to_numpy(dtype=float),expected.to_numpy(dtype=float),rtol=1e-9,atol=1e-9),label+': values/order differ')

def canonical_split(root):
    raw_path=root/'data/Churn_Modelling.csv'
    raw=pd.read_csv(raw_path).drop_duplicates()
    require(raw.CustomerId.is_unique,'Source CustomerIds must be unique')
    X=raw.drop(columns=['RowNumber','CustomerId','Surname','Exited']); y=raw.Exited
    a,b=train_test_split(np.arange(len(raw)),test_size=.2,random_state=42,stratify=y)
    require(len(b)==2000,'Expected exactly 2,000 test customers')
    require(not set(a)&set(b),'Train/test overlap')
    train=pd.get_dummies(X.iloc[a],columns=['Geography','Gender']).astype(float)
    test=pd.get_dummies(X.iloc[b],columns=['Geography','Gender']).astype(float).reindex(columns=train.columns,fill_value=0)
    same_frame(pd.read_csv(root/'outputs/X_train_encoded.csv'),train,'X_train_encoded')
    same_frame(pd.read_csv(root/'outputs/X_test_encoded.csv'),test,'X_test_encoded')
    for part,idx in [('train',a),('test',b)]:
        labels=pd.read_csv(root/f'outputs/y_{part}.csv').iloc[:,0]
        require(np.array_equal(labels.to_numpy(),y.iloc[idx].to_numpy()),f'y_{part}: labels/order differ')
    manifest=pd.DataFrame({'CustomerId':raw.CustomerId.iloc[b].to_numpy(),'y_true':y.iloc[b].to_numpy(),
                           'source_row_position':b})
    return manifest,hashlib.sha256(raw_path.read_bytes()).hexdigest()

def read_threshold(root,name,spec):
    table=pd.read_csv(root/'outputs'/spec[3])
    if name=='Logistic Regression':
        table=table.sort_values(['F1','Recall'],ascending=False)
    threshold=float(table.iloc[0]['Threshold'])
    require(np.isfinite(threshold),name+': invalid threshold')
    return threshold

def load_predictions(root,manifest,allow_legacy):
    loaded={};audit=[]
    for name,spec in SPECS.items():
        path=root/'outputs'/spec[0]
        p=pd.read_csv(path)
        require(len(p)==2000,name+': expected exactly 2000 predictions')
        require(set(spec[1]).issubset(p.columns),name+': required columns missing')
        if 'CustomerId' in p.columns:
            require(p.CustomerId.is_unique,name+': duplicate CustomerId')
            require(set(p.CustomerId)==set(manifest.CustomerId),name+': customer IDs differ')
            p=p.set_index('CustomerId').loc[manifest.CustomerId].reset_index()
            status='VERIFIED_BY_CUSTOMER_ID'
        else:
            require(allow_legacy,name+': CustomerId missing. Re-export IDs from the actual prediction rows, or explicitly use --allow-legacy-order for a provisional comparison.')
            status='ASSUMED_SHARED_ORDER_NO_IDS'
        standard=p[spec[1]].copy();standard.columns=['y_true','y_pred','y_score']
        require(np.isfinite(standard.to_numpy(dtype=float)).all(),name+': NaN/infinite values')
        require(np.array_equal(standard.y_true.to_numpy(),manifest.y_true.to_numpy()),name+': labels/order differ')
        require(standard.y_pred.isin([0,1]).all(),name+': predictions must be binary')
        if spec[2]=='probability':require(standard.y_score.between(0,1).all(),name+': probability outside [0,1]')
        threshold=read_threshold(root,name,spec)
        require(np.array_equal((standard.y_score.to_numpy()>=threshold).astype(int),standard.y_pred.to_numpy()),name+': predictions inconsistent with saved threshold')
        standard.insert(0,'CustomerId',manifest.CustomerId.to_numpy())
        # For legacy files these IDs are reconstructed ASSUMPTIONS, not observed proof.
        loaded[name]=(standard,threshold,spec[2])
        audit.append({'Model':name,'Rows':len(p),'Alignment':status,'Labels_match':True,
                      'Threshold_consistent':True,'Prediction_file':spec[0],
                      'Prediction_SHA256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return loaded,pd.DataFrame(audit)

def compute_metrics(p):
    return {'Accuracy':accuracy_score(p.y_true,p.y_pred),
        'Precision':precision_score(p.y_true,p.y_pred,zero_division=0),
        'Recall':recall_score(p.y_true,p.y_pred,zero_division=0),
        'F1':f1_score(p.y_true,p.y_pred,zero_division=0),
        'ROC_AUC':roc_auc_score(p.y_true,p.y_score)}

def compare(root,output,allow_legacy=False):
    root=Path(root).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    # Overwrite audit on every attempt so a failed run cannot appear verified.
    report={'status':'VALIDATING','allow_legacy_order':allow_legacy}
    audit_path=out/'verification.json'
    audit_path.write_text(json.dumps(report,indent=2))
    try:
        manifest,dataset_hash=canonical_split(root)
        loaded,audit=load_predictions(root,manifest,allow_legacy)
    except Exception as exc:
        report.update(status='FAILED',error=str(exc));audit_path.write_text(json.dumps(report,indent=2));raise
    strict=bool(audit.Alignment.eq('VERIFIED_BY_CUSTOMER_ID').all())
    status='VERIFIED_BY_CUSTOMER_ID' if strict else 'PROVISIONAL_SHARED_ORDER_NO_IDS'
    note=('Every prediction file has the exact same 2,000 customer IDs and matching labels.' if strict else
          'All five files have 2,000 matching labels in canonical order. Shared encoded train/test data exactly match the reconstructed split. Prediction files lack customer IDs; individual prediction-to-customer identity remains unproven. Metrics assume the scripts exported predictions in shared input order. Re-export CustomerId with actual prediction rows, then rerun WITHOUT --allow-legacy-order for strict verification.')
    report.update(status=status,n_test=2000,dataset_sha256=dataset_hash,note=note,
                  dataset_checks='Encoded train/test feature values, columns and row order, and train/test labels match reconstructed seed-42 stratified 80:20 split.',
                  per_model=audit.to_dict(orient='records'))
    audit_path.write_text(json.dumps(report,indent=2));audit.to_csv(out/'alignment_audit.csv',index=False)
    manifest.to_csv(out/'canonical_test_manifest.csv',index=False)
    rows=[];published_checks=[]
    for name,(p,threshold,kind) in loaded.items():
        scores=compute_metrics(p);tn,fp,fn,tp=confusion_matrix(p.y_true,p.y_pred,labels=[0,1]).ravel()
        rows.append({'Model':name,**scores,'Threshold':threshold,'Score_Type':kind,
                     'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),'Alignment':status})
        # Recompute all metrics; validate exported summaries as a consistency check.
        published=pd.read_csv(root/'outputs'/SPECS[name][4]).iloc[0]
        for metric,value in scores.items():
            key={'F1':'F1-score','ROC_AUC':'ROC-AUC'}.get(metric,metric)
            key=metric if metric in published.index else key
            if key in published.index:
                require(np.isclose(value,float(published[key]),atol=1e-10,rtol=1e-8),name+': published '+metric+' does not match predictions')
                published_checks.append({'Model':name,'Metric':metric,'Matches':True})
    table=pd.DataFrame(rows).sort_values('ROC_AUC',ascending=False).reset_index(drop=True)
    table.to_csv(out/'comparison_metrics.csv',index=False)
    pd.DataFrame(published_checks).to_csv(out/'published_metric_checks.csv',index=False)
    winners=[]
    for metric in METRICS:
        best=table[metric].max();names=table.loc[np.isclose(table[metric],best,atol=1e-12,rtol=0),'Model'].tolist()
        winners.append({'Metric':metric,'Best_Model':' / '.join(names),'Best_Value':best,'Alignment':status})
    winners=pd.DataFrame(winners);winners.to_csv(out/'best_by_metric.csv',index=False)
    plt.rcParams.update({'font.size':10})
    title_suffix=' (provisional)' if not strict else ''
    fig,ax=plt.subplots(figsize=(11,6))
    table.set_index('Model')[METRICS].plot.bar(ax=ax,width=.8,ylim=(0,1))
    ax.set_title('Five-model comparison'+title_suffix);ax.set_ylabel('Score');ax.set_xlabel('')
    ax.tick_params(axis='x',rotation=15);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=5)
    fig.tight_layout();fig.savefig(out/'metric_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    im=ax.imshow(table[METRICS],vmin=0,vmax=1,cmap='YlGnBu')
    ax.set_xticks(range(5),METRICS);ax.set_yticks(range(5),table.Model)
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{table.iloc[i][METRICS[j]]:.4f}',ha='center',va='center',color='white' if table.iloc[i][METRICS[j]]>.65 else 'black')
    ax.set_title('Metric heatmap'+title_suffix);fig.colorbar(im,ax=ax);fig.tight_layout()
    fig.savefig(out/'metric_heatmap.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8));roc_fig,roc_ax=plt.subplots(figsize=(8,6))
    for ax,(name,(p,threshold,kind)) in zip(axes.flat,loaded.items()):
        ConfusionMatrixDisplay.from_predictions(p.y_true,p.y_pred,labels=[0,1],display_labels=['Stayed','Churned'],ax=ax,colorbar=False,cmap='Blues')
        ax.set_title(f'{name} (threshold {threshold:g})')
        small,sax=plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay.from_predictions(p.y_true,p.y_pred,labels=[0,1],display_labels=['Stayed','Churned'],ax=sax,colorbar=False,cmap='Blues')
        sax.set_title(name+title_suffix);small.tight_layout();small.savefig(out/(name.lower().replace(' ','_')+'_confusion.png'),dpi=150);plt.close(small)
        fpr,tpr,_=roc_curve(p.y_true,p.y_score)
        roc_ax.plot(fpr,tpr,label=f'{name} (AUC={roc_auc_score(p.y_true,p.y_score):.4f})')
        small,sax=plt.subplots(figsize=(6,5));sax.plot(fpr,tpr);sax.plot([0,1],[0,1],'k--')
        sax.set(xlabel='False positive rate',ylabel='True positive rate',title=name+' ROC'+title_suffix)
        small.tight_layout();small.savefig(out/(name.lower().replace(' ','_')+'_roc.png'),dpi=150);plt.close(small)
    axes.flat[-1].axis('off');fig.suptitle('Confusion matrices'+title_suffix);fig.tight_layout()
    fig.savefig(out/'confusion_matrices.png',dpi=160);plt.close(fig)
    roc_ax.plot([0,1],[0,1],'k--',label='Chance');roc_ax.legend(loc='lower right')
    roc_ax.set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curves'+title_suffix)
    roc_fig.tight_layout();roc_fig.savefig(out/'roc_curves.png',dpi=160);plt.close(roc_fig)
    baseline=(manifest.y_true==0).mean()
    caveat=('Models use their own training-validation-selected thresholds, not a common 0.5 threshold. '
            'SVM scores are decision margins, not probabilities; ROC-AUC uses ranking scores directly. '
            'Winners are descriptive maxima on this held-out set, not proof of significance or a reason to retune on test data. '
            f'Always-stayed baseline accuracy: {baseline:.2%}; churn recall: 0%.')
    def embedded_image(filename):
        encoded=base64.b64encode((out/filename).read_bytes()).decode()
        return f'<img alt="{html.escape(filename)}" src="data:image/png;base64,{encoded}">'
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FinSIGHT model comparison</title><style>body{margin:0;background:#f1f5f9;color:#14233b;font:16px system-ui}main{max-width:1150px;margin:auto;padding:30px}h1{font-size:32px}section{background:white;border-radius:14px;padding:22px;margin:20px 0;overflow:auto}.notice{background:#fff3cd;border-left:5px solid #b58105;padding:18px;line-height:1.6}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px;border-bottom:1px solid #dce4ef;text-align:left}th{background:#eaf0f7}img{width:100%;height:auto}p{line-height:1.6}.tag{color:#8b5c00;font-weight:700}</style><main><h1>FinSIGHT · Five-model comparison</h1>'''
    doc+=f'<p class="tag">{html.escape(status)}</p><div class="notice">{html.escape(note)}</div>'
    doc+='<section><h2>Held-out performance</h2>'+table[['Model']+METRICS+['Threshold','Score_Type']].to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)+'<p>'+html.escape(caveat)+'</p></section>'
    doc+='<section><h2>Best model by metric</h2>'+winners[['Metric','Best_Model','Best_Value']].to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)+'</section>'
    for heading,filename in [('Metric comparison','metric_comparison.png'),('Metric heatmap','metric_heatmap.png'),('Confusion matrices','confusion_matrices.png'),('ROC curves','roc_curves.png')]:
        doc+=f'<section><h2>{heading}</h2>'+embedded_image(filename)+'</section>'
    doc+='<section><h2>Verification</h2>'+audit[['Model','Rows','Alignment','Labels_match','Threshold_consistent']].to_html(index=False,border=0)+'</section></main></html>'
    (out/'dashboard.html').write_text(doc,encoding='utf-8')
    (out/'SUMMARY.txt').write_text(status+'\n\n'+note+'\n\n'+table.to_string(index=False)+'\n\n'+winners.to_string(index=False)+'\n\n'+caveat+'\n')
    print(status+'\n'+note+'\n');print(table[['Model']+METRICS].to_string(index=False))
    print('\nBest by metric:\n'+winners[['Metric','Best_Model','Best_Value']].to_string(index=False))
    print('\nDashboard:',out/'dashboard.html');return table,report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    default_root=Path(__file__).resolve().parent
    if default_root.name=='src':default_root=default_root.parent
    parser.add_argument('--project',type=Path,default=default_root)
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--allow-legacy-order',action='store_true',help='Explicitly accept unproven prediction row identity; outputs are marked provisional')
    args=parser.parse_args()
    try:compare(args.project,args.output or args.project/'outputs/comparison',args.allow_legacy_order)
    except (ValueError,FileNotFoundError,KeyError) as exc:
        failure_out=args.output or args.project/'outputs/comparison'
        failure_out.mkdir(parents=True,exist_ok=True)
        (failure_out/'verification.json').write_text(json.dumps({'status':'FAILED','error':str(exc)},indent=2))
        print('Comparison stopped: '+str(exc),file=sys.stderr);sys.exit(1)
