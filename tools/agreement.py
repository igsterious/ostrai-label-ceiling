import json, glob, itertools, collections, os

def load(p):
    d = {}
    for line in open(p, encoding='utf-8'):
        r = json.loads(line)
        d[str(r.get('id'))] = r.get('intent')
    return d

A = 'labels/benchmark_a'
# independent passes only -- derived sets (consensus, handfixed, reranked) excluded
PASSES = {
    'sonnet-5 (v1)':      'pseudo_labels.anthropic_claude-sonnet-5.jsonl',
    'sonnet-5 (rules-v2)':'pseudo_labels.anthropic_claude-sonnet-5.rules-v2.jsonl',
    'sonnet-5 (10-shot)': 'pseudo_labels.anthropic_claude-sonnet-5.shot10.jsonl',
    'gpt-5.1 (v1)':       'pseudo_labels.openai_gpt-5.1.jsonl',
    'gpt-5.1 (rules-v2)': 'pseudo_labels.openai_gpt-5.1.rules-v2.jsonl',
    'gpt-5.1 (batch 10)': 'pseudo_labels.openai_gpt-5.1.b10.jsonl',
    'deepseek-chat':      'pseudo_labels.deepseek_deepseek-chat.jsonl',
    'gemini-3.1-flash':   'pseudo_labels.google_gemini-3.1-flash-lite.jsonl',
}
runs = {k: load(os.path.join(A, v)) for k, v in PASSES.items() if os.path.exists(os.path.join(A, v))}
ids = set.intersection(*[set(v) for v in runs.values()])
print(f'common ids: {len(ids):,}\n')

names = list(runs)
fam = lambda n: n.split(' ')[0]

print('PAIRWISE AGREEMENT (%)')
print(f'{"":<22}' + ''.join(f'{n.split(" ")[0][:7]:>9}' for n in names))
M = {}
for a in names:
    row = f'{a:<22}'
    for b in names:
        if a == b:
            row += f'{"--":>9}'; continue
        ag = sum(1 for i in ids if runs[a][i] == runs[b][i]) / len(ids) * 100
        M[(a, b)] = ag
        row += f'{ag:>9.1f}'
    print(row)

within = [v for (a, b), v in M.items() if fam(a) == fam(b)]
between = [v for (a, b), v in M.items() if fam(a) != fam(b)]
print(f'\nwithin-family  mean agreement: {sum(within)/len(within):.2f}%  (n={len(within)//2} pairs)')
print(f'between-family mean agreement: {sum(between)/len(between):.2f}%  (n={len(between)//2} pairs)')

# gpt-5.1 family specifically -- the three runs that formed the consensus
g = [n for n in names if n.startswith('gpt-5.1')]
gw = [M[(a,b)] for a,b in itertools.combinations(g,2)]
print(f'gpt-5.1 three-run mean pairwise agreement: {sum(gw)/len(gw):.2f}%')
allthree = sum(1 for i in ids if len({runs[n][i] for n in g}) == 1) / len(ids) * 100
print(f'gpt-5.1 all three identical: {allthree:.2f}%  ({round(allthree/100*len(ids)):,} rows)')

# where do they disagree -- top confusable pairs across the gpt runs
pairs = collections.Counter()
for i in ids:
    v = sorted({runs[n][i] for n in g})
    for a, b in itertools.combinations(v, 2):
        pairs[(a, b)] += 1
print('\nTOP 12 CONFUSED CATEGORY PAIRS (gpt-5.1 three runs)')
for (a, b), c in pairs.most_common(12):
    print(f'  {c:>4}  {a}  vs  {b}')

json.dump({'M': {f'{a}||{b}': v for (a,b), v in M.items()},
           'within': sum(within)/len(within), 'between': sum(between)/len(between),
           'gpt_pairwise': sum(gw)/len(gw), 'gpt_unanimous': allthree,
           'n': len(ids), 'names': names,
           'pairs': [[a, b, c] for (a, b), c in pairs.most_common(12)]},
          open('/c/Users/Igor/AppData/Local/Temp/claude/C--Users-Igor/fa603f87-4ffc-453a-a5f5-ee8a08d39aa0/scratchpad/agree.json','w'), indent=1)
