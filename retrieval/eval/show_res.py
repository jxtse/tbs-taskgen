import json, sys
tid = sys.argv[1]
rows = [json.loads(l) for l in open("eval/to_label_res.jsonl") if json.loads(l)["target"] == tid]
print("##", tid, len(rows))
for i, r in enumerate(rows):
    print(f"{i}|{r['by']}|{r['kind']}|{r['key'][:70]}|{r['title'][:60]}|{r['snippet'][:70]}")
