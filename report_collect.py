import json
import shutil
import urllib.request
from pathlib import Path
import pandas as pd
from PIL import Image
from fruits import ROOT, FRUITS, paths

DEST=ROOT.parent/'Laporan_Ripeness_Lab'/'bukti'
def main():
    out={}
    for fruit in FRUITS:
        data,artifacts=paths(fruit)
        with urllib.request.urlopen('http://127.0.0.1:8000/api/summary?fruit='+fruit) as res:
            summary=json.load(res)
        local=json.loads((artifacts/'summary.json').read_text())
        assert summary['experiments']==local['experiments']
        folder=DEST/fruit;folder.mkdir(exist_ok=True)
        shutil.copy2(data/'audit.json',folder/'audit.json')
        shutil.copy2(data/'manifest.csv',folder/'manifest.csv')
        (folder/'api_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        shutil.copy2(artifacts/'comparison.csv',folder/'comparison.csv')
        for run in summary['experiments']:
            ident=run['config']['id'];target=folder/ident;target.mkdir(exist_ok=True)
            for name in ['history.csv','results.json','curves.png','confusion_matrix.png','test_predictions.csv']:
                shutil.copy2(artifacts/ident/name,target/name)
            with urllib.request.urlopen(f'http://127.0.0.1:8000/api/errors?fruit={fruit}&model={ident}') as res:
                errors=json.load(res)
            frame=pd.read_csv(artifacts/ident/'test_predictions.csv')
            assert len(errors)==len(frame)
            assert [x['prediction'] for x in errors]==frame.prediction.tolist()
            (target/'api_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
        out[fruit]=summary
    (DEST/'all_summaries.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    
    for p in DEST.glob('*.png'):
        if p.stem.endswith('_panel'):
            continue
        with Image.open(p) as im:
            
            top=425 if im.height>1500 else 0
            height=1000 if 'comparison' in p.name else 1250
            panel=im.crop((0,top,im.width,min(im.height,top+height)))
            panel.save(p.with_stem(p.stem+'_panel'))
    print('Verified live API summaries and prediction rows for 5 fruits x 4 experiments.')

if __name__=='__main__':main()
