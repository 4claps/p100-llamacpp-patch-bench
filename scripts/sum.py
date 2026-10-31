import json,glob,sys,os,statistics as st,collections
d=sys.argv[1]; R=collections.defaultdict(list)
for f in glob.glob(d+"/*.jsonl"):
    a,m,kv,r=os.path.basename(f)[:-6].rsplit(".",3) if False else (None,None,None,None)
    parts=os.path.basename(f)[:-6].split("."); a=parts[0]; r=parts[-1]; kv=parts[-2]; m=".".join(parts[1:-2])
    for line in open(f):
        try: j=json.loads(line)
        except: continue
        t="pp%d"%j["n_prompt"] if j["n_prompt"] else "tg%d"%j["n_gen"]
        R[(m,kv,t,j.get("n_depth",0),a)].append(j["avg_ts"])
keys=sorted(set(k[:4] for k in R)); arms=sorted(set(k[4] for k in R))
print("model kv test depth | "+" | ".join(arms))
for k in keys:
    vals={a:R.get(k+(a,)) for a in arms}
    base=vals.get(arms[0]) 
    row=[]
    for a in arms:
        v=vals[a]
        row.append("%.2f(n%d,%.2f-%.2f)"%(st.median(v),len(v),min(v),max(v)) if v else "-")
    print(" ".join(map(str,k))," | "," | ".join(row))
