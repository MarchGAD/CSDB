import os

def cntline(file):
    tcnt = 0
    with open(file, 'r') as f:
        for line in f:
            tcnt += 1
    return tcnt

ign = ['exp', 'json_configs']
sta = ['.']

cnt = 0
while len(sta) > 0:
    pat = sta.pop()
    ts = os.listdir(pat)
    for t in ts:
        tmppat = os.path.join(pat, t)
        if os.path.isdir(t):
            sta.append(tmppat)
        elif '.py' in t:
            cnt += cntline(tmppat)
print(cnt)