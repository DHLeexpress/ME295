import json,sys
R=json.load(open('results.json'))
def sel(**kw):
    out=[]
    for r in R:
        c=r['config']; ok=True
        for k,v in kw.items():
            cv=c.get(k)
            if k=='cells': cv=tuple(cv)
            if cv!=v: ok=False;break
        if ok: out.append(r)
    return out
def f(x,fmt='%.3g'):
    if x is None: return '-'
    try: return fmt%x
    except: return str(x)
def row(r):
    c=r['config']
    return ' '.join([r['id'][:6],c['variant'][:5],'x'.join(map(str,c['cells'])),c['case'],c['contact'][:5],'it%d'%c['iterations'],c['solver'][:2],'s%d'%c['split'],'nu%.2f'%c['nu'],c['driver'][:2],
      r['status'][:8],'K=',f(r.get('K_app_N_per_um'),'%.3f'),'Kl=',f(r.get('K_layer_N_per_um'),'%.3f'),'pf=',f(r.get('penetration_frac_foot'),'%.3f'),
      'nc=',f(r.get('ncon'),'%d'),'it=',f(r.get('niter_median'),'%.0f'),'cap=',f(r.get('frac_steps_at_iter_cap'),'%.2f'),'ch=',f(r.get('chatter_rel'),'%.1e'),
      'st1=',f(r.get('settle_1pct_s'),'%.3f'),'us=',f(r.get('us_per_step_median'),'%.0f')])
if __name__=='__main__':
    kw=eval('dict(%s)'%sys.argv[1])
    for r in sorted(sel(**kw),key=lambda r:(r['config']['case'],r['config']['cells'][0],r['config']['cells'][2])): print(row(r))
