import argparse, random, zipfile, io, json
from pathlib import Path
from collections import defaultdict, Counter
from types import SimpleNamespace
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parent

def load(name):
    base=ROOT/'data'/name
    labels={}
    if name=='Weibo':
        for line in (base/'weibo_id_label.txt').read_text().splitlines():
            eid,y=line.split(); labels[eid]=int(y)
        archive=base/'weibotree.txt.zip'
        source=archive if archive.exists() else io.BytesIO(b''.join(part.read_bytes() for part in sorted(base.glob('weibotree.txt.zip.part*'))))
        z=zipfile.ZipFile(source); stream=io.TextIOWrapper(z.open('weibotree.txt'),encoding='utf-8')
    else:
        mapping={'news':0,'non-rumor':0,'false':1,'true':2,'unverified':3}
        for line in (base/f'{name}_label_All.txt').read_text().splitlines():
            a=line.split('\t'); labels[a[2]]=mapping[a[0].lower()]
        stream=(base/'data.TD_RvNN.vol_5000.txt').open()
    trees=defaultdict(dict)
    with stream:
        for line in stream:
            a=line.rstrip('\n').split('\t'); eid=a[0]
            if eid in labels: trees[eid][int(a[2])]=(a[1],a[-1])
    if name=='Weibo': z.close()
    valid={eid:rows for eid,rows in trees.items() if len(rows)>1 and sum(p=='None' for p,v in rows.values())==1 and all(p=='None' or int(p) in rows for p,v in rows.values())}
    return valid,labels

def graph(rows,y):
    ids=sorted(rows); idx={v:i for i,v in enumerate(ids)}; x=torch.zeros(len(ids),5000); edges=[]
    for k,(parent,vec) in rows.items():
        i=idx[k]
        if parent=='None': root=i
        else: edges.append((idx[int(parent)],i))
        for pair in vec.split():
            a,b=pair.split(':'); j=int(a)
            if 0<=j<5000: x[i,j]=float(b)
    return x,torch.tensor(edges,dtype=torch.long).reshape(-1,2).T,root,y

def batch(items,device,training=False):
    xs=[]; td=[]; bu=[]; roots=[]; ys=[]; bs=[]; offset=0
    for i,(x,e,r,y) in enumerate(items):
        xs.append(x); roots.append(r+offset); ys.append(y); bs.extend([i]*len(x))
        for dest,edge in [(td,e),(bu,e.flip(0))]:
            if training: edge=edge[:,torch.randperm(edge.size(1))[:int(edge.size(1)*.8)]]
            dest.append(edge+offset)
        offset+=len(x)
    return SimpleNamespace(**{k:v.to(device) for k,v in dict(x=torch.cat(xs),edge_index=torch.cat(td,1),BU_edge_index=torch.cat(bu,1),rootindex=torch.tensor(roots),y=torch.tensor(ys),batch=torch.tensor(bs)).items()})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['Twitter15','Twitter16','Weibo'],default='Twitter15'); p.add_argument('--epochs',type=int,default=3); p.add_argument('--limit-per-class',type=int,default=20,help='0 uses all valid graphs'); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--hidden',type=int,default=32); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('BiGCN | DEBRAJ DE | 25BTRAO013',flush=True); print('PyTorch:',torch.__version__,'| Device:',device,flush=True)
    trees,labels=load(a.dataset); groups=defaultdict(list)
    for eid in sorted(trees): groups[labels[eid]].append(eid)
    print('Dataset:',a.dataset,'| Valid labeled graphs:',len(trees),'| Class counts:',dict(sorted((k,len(v)) for k,v in groups.items())),flush=True)
    selected=[]
    for k,ids in sorted(groups.items()):
        random.shuffle(ids); selected.extend(ids[:a.limit_per_class] if a.limit_per_class else ids)
    tr,te=train_test_split(selected,test_size=.25,random_state=a.seed,stratify=[labels[i] for i in selected]); print('Train:',len(tr),'| Held-out test:',len(te),flush=True)
    if a.dataset=='Weibo': from model.Weibo.network import Net
    else: from model.Twitter.network import Net
    model=Net(5000,a.hidden,a.hidden).to(device); opt=torch.optim.Adam([{'params':model.TDrumorGCN.parameters()},{'params':model.BUrumorGCN.parameters(),'lr':.0001},{'params':model.fc.parameters()}],lr=.0005,weight_decay=.0001)
    def batches(ids,training=False):
        for i in range(0,len(ids),a.batch_size):
            yield batch([graph(trees[k],labels[k]) for k in ids[i:i+a.batch_size]],device,training)
    history=[]
    for epoch in range(a.epochs):
        model.train(); random.shuffle(tr); loss_sum=0; correct=0
        for b in batches(tr,True):
            opt.zero_grad(); pred=model(b); loss=torch.nn.functional.nll_loss(pred,b.y); loss.backward(); opt.step(); loss_sum+=loss.item()*len(b.y); correct+=int((pred.argmax(1)==b.y).sum())
        row={'epoch':epoch+1,'loss':loss_sum/len(tr),'train_accuracy':correct/len(tr)}; history.append(row); print(f"Epoch {epoch+1:02d}/{a.epochs:02d} | Loss {row['loss']:.4f} | Train accuracy {row['train_accuracy']:.2%}",flush=True)
    model.eval(); truth=[]; predicted=[]
    with torch.no_grad():
        for b in batches(te): truth.extend(b.y.cpu().tolist()); predicted.extend(model(b).argmax(1).cpu().tolist())
    names=['non-rumor','false','true','unverified'] if a.dataset!='Weibo' else ['non-rumor','rumor']; report=classification_report(truth,predicted,labels=list(range(len(names))),target_names=names,zero_division=0,output_dict=True)
    print(classification_report(truth,predicted,labels=list(range(len(names))),target_names=names,zero_division=0)); print('Confusion matrix:\n',confusion_matrix(truth,predicted,labels=list(range(len(names))))); print('Held-out accuracy:',sum(x==y for x,y in zip(truth,predicted))/len(truth)); print('SUCCESS: forward pass, backpropagation and held-out evaluation completed.')
    out=ROOT/'results'; out.mkdir(exist_ok=True); (out/f'{a.dataset}_metrics.json').write_text(json.dumps(dict(config=vars(a),train_ids=tr,test_ids=te,history=history,truth=truth,predicted=predicted,report=report),indent=2)); torch.save(model.state_dict(),out/f'{a.dataset}_model.pt')
if __name__=='__main__': main()
