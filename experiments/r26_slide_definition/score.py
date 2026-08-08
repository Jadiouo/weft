import json,pathlib,sys
tot_ok=tot_n=0
for f in sorted(pathlib.Path("tests/golden").glob("*.golden.json")):
    g=json.loads(f.read_text()); vid=g["video_id"]
    d=pathlib.Path(f"work/{vid}/04_slide_understanding")
    reps=sorted(set(g["slide_groups"].values()))
    ok=n=0; errs=[]
    for sid in reps:
        p=d/f"{sid}.json"
        if not p.exists(): continue
        got=bool(json.loads(p.read_text()).get("is_slide")); exp=g["slide_classes"][sid]
        n+=1; ok+= got==exp
        if got!=exp: errs.append(f"{sid.split('_')[1]}{'FP' if got else 'FN'}")
    if not n: print(f"{vid}: 無輸出"); continue
    tot_ok+=ok; tot_n+=n
    mark="" if n>=len(reps) else f" **只跑 {n}/{len(reps)}**"
    print(f"{vid}: {ok}/{n} = {ok/n:.3f}{mark}   錯: {' '.join(errs) or '—'}")
print(f"\n合計 {tot_ok}/{tot_n} = {tot_ok/tot_n:.3f}")
