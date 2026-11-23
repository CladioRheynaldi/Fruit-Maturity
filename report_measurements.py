import argparse
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torch
from sklearn.metrics import accuracy_score, f1_score
from fruits import ROOT, FRUITS, paths
from models import build, transform, head_and_features

DEST = ROOT.parent / 'Laporan_Ripeness_Lab' / 'bukti'

def worker(fruit, ident):
    data, artifacts = paths(fruit)
    process = psutil.Process()
    base = process.memory_info().rss
    start = time.perf_counter()
    state = torch.load(artifacts / ident / 'best.pt', map_location='cpu', weights_only=False)
    model = build(state['config']['architecture'], num_classes=len(state['classes']))
    model.load_state_dict(state['state_dict'])
    model.eval()
    load_ms = (time.perf_counter()-start)*1000
    frame = pd.read_csv(data / 'manifest.csv')
    p = ROOT / frame[frame.split=='test'].iloc[0].path
    with Image.open(p) as im:
        x = transform()(ImageOps.exif_transpose(im).convert('RGB')).unsqueeze(0)
    latencies = []
    with torch.inference_mode():
        for _ in range(5): model(x)
        for _ in range(30):
            start = time.perf_counter()
            model(x)
            latencies.append((time.perf_counter()-start)*1000)
    memory = process.memory_info()
    return dict(fruit=fruit, model=ident, load_ms=load_ms, p50_ms=float(np.median(latencies)),
        p95_ms=float(np.percentile(latencies,95)), rss_base_mib=base/2**20,
        rss_after_mib=memory.rss/2**20, peak_process_mib=getattr(memory,'peak_wset',memory.rss)/2**20,
        checkpoint_mib=(artifacts / ident / 'best.pt').stat().st_size/2**20,
        param_mib=sum(p.numel()*p.element_size() for p in model.parameters())/2**20,
        notes='Fresh process; imports included in RSS baseline; checkpoint tensors retained. 5 warmups + 30 forward-only batch-1 passes; 4 CPU threads, excludes image decode/resize/network. Peak is whole process, not GPU VRAM or training peak.')

def robustness(fruit):
    data, artifacts = paths(fruit)
    summary = json.loads((artifacts/'summary.json').read_text())
    ident = summary['selected_model']
    state = torch.load(artifacts/ident/'best.pt',map_location='cpu',weights_only=False)
    model=build(state['config']['architecture'],num_classes=len(state['classes']))
    model.load_state_dict(state['state_dict']); model.eval()
    frame=pd.read_csv(data/'manifest.csv'); frame=frame[frame.split=='test']
    images=[]
    for p in frame.path:
        with Image.open(ROOT/p) as im:
            images.append(ImageOps.exif_transpose(im).convert('RGB').resize((128,128),Image.Resampling.BICUBIC))
    y=[state['classes'].index(s) for s in frame.label]
    conditions={
        'original':lambda im:im,
        'brightness_0.6':lambda im:ImageEnhance.Brightness(im).enhance(.6),
        'brightness_1.4':lambda im:ImageEnhance.Brightness(im).enhance(1.4),
        'blur_radius_2':lambda im:im.filter(ImageFilter.GaussianBlur(2)),
    }
    results=[]
    for name,change in conditions.items():
        preds=[]
        with torch.inference_mode():
            for i in range(0,len(images),32):
                x=torch.stack([transform()(change(im)) for im in images[i:i+32]])
                preds.extend(model(x).argmax(1).tolist())
        results.append(dict(condition=name,accuracy=accuracy_score(y,preds),
                            f1=f1_score(y,preds,labels=list(range(len(state['classes']))),average='macro',zero_division=0)))
    return dict(fruit=fruit,selected_model=ident,n=len(y),conditions=results,
        note='Exploratory sensitivity evaluation on existing test images, not independent field validation. Perturbations applied after deterministic 128x128 bicubic resize. Test metrics not used for retraining/model selection.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',nargs=2);args=parser.parse_args()
    torch.set_num_threads(4)
    if args.worker:
        print(json.dumps(worker(*args.worker)))
    else:
        DEST.mkdir(parents=True,exist_ok=True)
        out=dict(date='2026-11-23',hardware=dict(platform=platform.platform(),cpu=platform.processor(),
            logical_cpus=psutil.cpu_count(),physical_cpus=psutil.cpu_count(logical=False),
            system_ram_gib=psutil.virtual_memory().total/2**30,torch=torch.__version__,threads=4,
            historical_training_hardware='Historical training logs record CPU device only; exact historical CPU/load is not fully logged.'),benchmarks=[],robustness=[])
        for fruit in FRUITS:
            for ident in ['baseline','cnn_low_lr','resnet18','mobilenet']:
                process=subprocess.run([sys.executable,__file__,'--worker',fruit,ident],capture_output=True,text=True,check=True)
                line=process.stdout.strip().splitlines()[-1]
                out['benchmarks'].append(json.loads(line))
                print('BENCH',fruit,ident,flush=True)
            out['robustness'].append(robustness(fruit));print('ROBUSTNESS',fruit,flush=True)
            (DEST/'measurements.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
