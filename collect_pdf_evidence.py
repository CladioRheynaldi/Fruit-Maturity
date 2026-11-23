import concurrent.futures
import hashlib
import io
import json
import platform
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image, ImageEnhance, ImageFilter
from fruits import ROOT, paths
from models import build, transform, head_and_features

OUT = ROOT / 'output/pdf/evidence'
OUT.mkdir(parents=True, exist_ok=True)
BASE = 'http://127.0.0.1:8000'
def get(route):
    with urllib.request.urlopen(BASE + route, timeout=40) as response:
        return json.load(response)

def main():
    fruits = get('/api/fruits')
    snapshot = dict(captured_at=datetime.now(ZoneInfo('Asia/Jakarta')).isoformat(),
                    base_url=BASE, fruits=fruits, summaries={}, predictions={}, validation={})
    for fruit_info in fruits:
        fruit = fruit_info['id']
        data, artifacts = paths(fruit)
        summary = get('/api/summary?fruit=' + fruit)
        snapshot['summaries'][fruit] = summary
        local = json.loads((artifacts / 'summary.json').read_text(encoding='utf-8'))
        assert summary['experiments'] == local['experiments'], 'Live/local experiment mismatch'
        assert summary['audit'] == local['audit'], 'Live/local audit mismatch'
        snapshot['predictions'][fruit] = {}
        for run in summary['experiments']:
            ident = run['config']['id']
            rows = get(f'/api/errors?fruit={fruit}&model={ident}')
            local_rows = pd.read_csv(artifacts / ident / 'test_predictions.csv').to_dict('records')
            assert len(rows) == len(local_rows)
            assert all(a['path']==b['path'] and a['prediction']==b['prediction'] and abs(a['confidence']-b['confidence'])<1e-10 for a,b in zip(rows, local_rows))
            assert abs(sum(r['correct'] for r in rows)/len(rows) - run['test']['accuracy']) < 1e-10
            snapshot['predictions'][fruit][ident] = rows
        snapshot['validation'][fruit] = 'Live API matches local summary, audit and every prediction; accuracy recomputed.'
        print('Captured', fruit, flush=True)
    (OUT / 'live_snapshot.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    torch.set_num_threads(4)
    hardware = dict(platform=platform.platform(), processor=platform.processor(),
                    logical_cores=psutil.cpu_count(), physical_cores=psutil.cpu_count(logical=False),
                    system_ram_gib=psutil.virtual_memory().total/2**30, torch=torch.__version__,
                    cuda_available=torch.cuda.is_available(), threads=4)
    (OUT/'hardware.json').write_text(json.dumps(hardware,indent=2),encoding='utf-8')
    stats=[]
    for fruit_info in fruits:
        fruit=fruit_info['id']; data, artifacts=paths(fruit)
        summary=snapshot['summaries'][fruit]
        manifest=pd.read_csv(data/'manifest.csv'); test=manifest[manifest.split=='test'].reset_index(drop=True)
        for run in summary['experiments']:
            ident=run['config']['id']; checkpoint=artifacts/ident/'best.pt'
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=build(run['config']['architecture'],num_classes=len(summary['classes']))
            model.load_state_dict(state['state_dict']); model.eval()
            with Image.open(ROOT/test.iloc[0].path) as im:
                payload=io.BytesIO(); im.convert('RGB').save(payload,format='PNG'); payload=payload.getvalue()
                x=transform()(im.convert('RGB')).unsqueeze(0)
            with torch.inference_mode():
                for _ in range(3): model(x)
                latencies=[]
                for _ in range(20):
                    start=time.perf_counter(); model(x); latencies.append((time.perf_counter()-start)*1000)
            params=sum(p.numel() for p in model.parameters())
            weights_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
            buffer_bytes=sum(p.numel()*p.element_size() for p in model.buffers())
            trainable=params
            if run['config']['transfer']:
                head,_=head_and_features(model,run['config']['architecture']); trainable=sum(p.numel() for p in head.parameters())
            row=dict(fruit=fruit,model=ident,parameters=params,trainable_parameters=trainable,
                     weights_mib=weights_bytes/2**20,buffers_mib=buffer_bytes/2**20,
                     checkpoint_mib=checkpoint.stat().st_size/2**20,
                     checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     warm_forward_ms_median=float(np.median(latencies)), warm_forward_ms_p95=float(np.percentile(latencies,95)),
                     process_rss_mib=psutil.Process().memory_info().rss/2**20)
            
            if ident == summary['selected_model']:
                subset=[]
                for label in summary['classes']:
                    subset.extend(test[test.label==label].sample(n=min(10,len(test[test.label==label])),random_state=42).index.tolist())
                correct={k:0 for k in ['original','dark_0.55','bright_1.45','blur_radius_2']}
                changed={k:0 for k in correct}
                for index in subset:
                    with Image.open(ROOT/test.iloc[index].path) as im:
                        im=im.convert('RGB')
                        variants=dict(original=im,dark_0_55=ImageEnhance.Brightness(im).enhance(.55),
                                      bright_1_45=ImageEnhance.Brightness(im).enhance(1.45),blur_radius_2=im.filter(ImageFilter.GaussianBlur(2)))
                        variants={'original':variants['original'],'dark_0.55':variants['dark_0_55'],
                                  'bright_1.45':variants['bright_1_45'],'blur_radius_2':variants['blur_radius_2']}
                        with torch.inference_mode():
                            predicted=model(torch.stack([transform()(v) for v in variants.values()])).argmax(1).tolist()
                    truth=summary['classes'].index(test.iloc[index].label)
                    for k,p in zip(variants,predicted):
                        correct[k]+=int(p==truth); changed[k]+=int(p!=predicted[0])
                row['robustness_probe']=dict(n=len(subset),indices=subset,correct=correct,
                    accuracy={k:v/len(subset) for k,v in correct.items()},predictions_changed=changed)
            stats.append(row)
            print('Measured',fruit,ident,flush=True)
            del model,state
    (OUT/'measurements.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')

if __name__=='__main__': main()
