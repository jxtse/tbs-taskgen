import json, sys
tid = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 160
rows = [json.loads(l) for l in open("eval/to_label.jsonl") if json.loads(l)["target"] == tid]
print(tid, len(rows))
for i, r in enumerate(rows):
    print(f"{i}|{r['title'][:110]}|{(r['abstract'] or '')[:n].replace(chr(10),' ')}")
