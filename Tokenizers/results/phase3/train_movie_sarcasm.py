"""
Standalone (non-project) script: fine-tune a BERT sarcasm classifier on
movie-subreddit Reddit comments (SARC), then compare it on a held-out movie
test set against the ready models dnzblgn (customer reviews) and helinivan
(news headlines). Writes a trained checkpoint + a fixed test split for
reproducibility. Run from the Tokenizers/ directory.
"""
import os, random, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

SEED=42; MAXLEN=128; BATCH=32; EPOCHS=3; LR=2e-5
# base model + output dir are configurable so the same script trains BERT or RoBERTa
BASE=os.environ.get("BASE_MODEL","bert-base-uncased")
OUT=os.environ.get("OUT_DIR","phase3_sarcasm/checkpoints/movie_reddit_model")
TESTCSV="phase3_sarcasm/data/reddit_movie_test.csv"
DEVICE=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
MOVIE={"movies","truefilm","flicks","moviedetails","marvelstudios","boxoffice","horror","film","filmmakers","moviesuggestions","dc_cinematic","criterion","netflixbestof","scifi","marvelmemes","starwars","batman","dceuleaks","letterboxd","movieclub"}

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def gather():
    ds=load_dataset("marcbishara/sarcasm-on-reddit")
    seen=set(); rows=[]
    for split in ds:
        s=ds[split]
        for sub,com,lab in zip(s["subreddit"],s["comment"],s["label"]):
            if sub and sub.lower() in MOVIE and com and len(com)>=15:
                k=com.strip().lower()
                if k not in seen:
                    seen.add(k); rows.append({"text":com,"label":int(lab)})
    return pd.DataFrame(rows)

class DS(Dataset):
    def __init__(self,df,tok): self.t=df["text"].astype(str).tolist(); self.y=df["label"].tolist(); self.tok=tok
    def __len__(self): return len(self.t)
    def __getitem__(self,i):
        e=self.tok(self.t[i],truncation=True,max_length=MAXLEN,padding="max_length",return_tensors="pt")
        return {"input_ids":e["input_ids"].squeeze(0),"attention_mask":e["attention_mask"].squeeze(0),"labels":torch.tensor(self.y[i])}

def run_epoch(model,loader,opt=None,sch=None):
    train=opt is not None; model.train() if train else model.eval()
    tot=0;correct=0;seen=0
    for step,b in enumerate(loader):
        b={k:v.to(DEVICE) for k,v in b.items()}
        with torch.set_grad_enabled(train): out=model(**b)
        if train:
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
        tot+=out.loss.item()*b["labels"].size(0)
        correct+=(out.logits.argmax(-1)==b["labels"]).sum().item(); seen+=b["labels"].size(0)
        if train and step%20==0: print(f"    step {step}/{len(loader)} loss {out.loss.item():.4f}",flush=True)
    return tot/seen, correct/seen

def predict(model,tok,texts):
    model.eval(); preds=[]
    for i in range(0,len(texts),64):
        e=tok([str(x) for x in texts[i:i+64]],truncation=True,padding=True,max_length=MAXLEN,return_tensors="pt").to(DEVICE)
        with torch.no_grad(): preds.extend(model(**e).logits.argmax(-1).cpu().tolist())
    return np.array(preds)

def metrics(name,y,yp):
    acc=accuracy_score(y,yp);p=precision_score(y,yp,zero_division=0);r=recall_score(y,yp,zero_division=0);f=f1_score(y,yp,zero_division=0)
    cm=confusion_matrix(y,yp)
    print(f"{name:<28} acc {acc:.4f} | P {p:.4f} | R {r:.4f} | F1 {f:.4f} | sarc-share {yp.mean():.0%}")
    print(f"    CM: TN{cm[0,0]} FP{cm[0,1]} FN{cm[1,0]} TP{cm[1,1]}")
    return acc,f

def main():
    set_seed()
    print(f"Device: {DEVICE}")
    df=gather()
    print(f"Movie-subreddit sarcasm comments: {len(df)} | balance {df.label.value_counts().to_dict()}")
    tr,te=train_test_split(df,test_size=0.2,stratify=df["label"],random_state=SEED)
    tr2,va=train_test_split(tr,test_size=0.1,stratify=tr["label"],random_state=SEED)
    os.makedirs(os.path.dirname(TESTCSV),exist_ok=True); te.to_csv(TESTCSV,index=False)
    print(f"Split: train {len(tr2)} / val {len(va)} / test {len(te)}  (test saved -> {TESTCSV})")

    tok=AutoTokenizer.from_pretrained(BASE)
    model=AutoModelForSequenceClassification.from_pretrained(BASE,num_labels=2).to(DEVICE)
    trl=DataLoader(DS(tr2,tok),batch_size=BATCH,shuffle=True); val=DataLoader(DS(va,tok),batch_size=BATCH)
    steps=len(trl)*EPOCHS
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
    sch=get_linear_schedule_with_warmup(opt,int(steps*0.1),steps)
    best=1e9
    for ep in range(1,EPOCHS+1):
        t0=time.time(); trloss,tracc=run_epoch(model,trl,opt,sch); valoss,vacc=run_epoch(model,val)
        print(f"Epoch {ep}/{EPOCHS} train loss {trloss:.4f} acc {tracc:.4f} | val loss {valoss:.4f} acc {vacc:.4f} | {(time.time()-t0)/60:.1f}min",flush=True)
        if valoss<best:
            best=valoss; os.makedirs(OUT,exist_ok=True); model.save_pretrained(OUT); tok.save_pretrained(OUT)
            print(f"  ↳ saved best -> {OUT}",flush=True)

    # ---- Comparison on the held-out movie test set ----
    print("\n"+"="*70); print("COMPARISON on held-out MOVIE test set ("+str(len(te))+" comments)"); print("="*70)
    y=te["label"].to_numpy(); texts=te["text"].tolist()
    rows={}
    # new model (reload best)
    nm=AutoModelForSequenceClassification.from_pretrained(OUT).to(DEVICE); nt=AutoTokenizer.from_pretrained(OUT)
    newlabel=f"NEW {BASE} (movie Reddit)"
    rows[newlabel]=metrics(newlabel,y,predict(nm,nt,texts))
    for label,mid in [("dnzblgn (cust.reviews)","dnzblgn/Sarcasm-Detection-Customer-Reviews"),
                      ("helinivan (headlines)","helinivan/english-sarcasm-detector")]:
        m=AutoModelForSequenceClassification.from_pretrained(mid).to(DEVICE); t=AutoTokenizer.from_pretrained(mid)
        rows[label]=metrics(label,y,predict(m,t,texts))
    maj=np.full_like(y,int(np.bincount(y).argmax()))
    print(f"{'majority baseline':<28} acc {accuracy_score(y,maj):.4f} | F1 {f1_score(y,maj,zero_division=0):.4f}")
    print("\n=== RANKING by F1 on movie test set ===")
    for k,(a,f) in sorted(rows.items(),key=lambda x:-x[1][1]):
        print(f"  {k:<28} F1 {f:.4f}  acc {a:.4f}")
    print("\nDONE")

if __name__=="__main__":
    main()
