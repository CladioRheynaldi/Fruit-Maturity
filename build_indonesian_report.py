import json, math, shutil, zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import pandas as pd
import fitz
from PIL import Image as PILImage
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from fruits import ROOT, FRUITS, paths

OUT=ROOT.parent/'Laporan_Ripeness_Lab'; E=OUT/'bukti'
S=json.loads((E/'all_summaries.json').read_text(encoding='utf-8'))
M=json.loads((E/'measurements.json').read_text(encoding='utf-8'))
PDF=OUT/'Laporan_Analisis_Ripeness_Lab.pdf'
pdfmetrics.registerFont(TTFont('Calibri','C:/Windows/Fonts/calibri.ttf'))
pdfmetrics.registerFont(TTFont('CalibriB','C:/Windows/Fonts/calibrib.ttf'))
pdfmetrics.registerFontFamily('Calibri',normal='Calibri',bold='CalibriB')
st=getSampleStyleSheet()
for name in ['Normal','BodyText']:
 st[name].fontName='Calibri';st[name].fontSize=10;st[name].leading=13.5;st[name].spaceAfter=7
for name in ['Heading1','Heading2','Heading3']:
 st[name].fontName='CalibriB';st[name].textColor=colors.HexColor('#244b3b')
st.add(ParagraphStyle(name='SmallReport',fontName='Calibri',fontSize=8,leading=10,spaceAfter=5))
st.add(ParagraphStyle(name='CellReport',fontName='Calibri',fontSize=8,leading=10))
story=[]; W=503
def p(t,style='BodyText'):return Paragraph(t,st[style])
def para(t):story.append(p(t))
def heading(t):story.append(p(t,'Heading1'))
def sub(t):story.append(p(t,'Heading2'))
def page(t):story.append(PageBreak());heading(t)
def table(rows,widths=None,size=8):
 cells=[[p(escape(str(x)).replace('\n','<br/>'),'CellReport') for x in row] for row in rows]
 t=Table(cells,colWidths=widths or [W/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3ecdf')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#dce3d9')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9f4')])]))
 story.append(t);story.append(Spacer(1,8))
def pic(path,maxh=260,width=W):
 path=Path(path)
 if not path.exists():raise FileNotFoundError(path)
 with PILImage.open(path) as im:w,h=im.size
 scale=min(width/w,maxh/h)
 story.append(Image(str(path),width=w*scale,height=h*scale,hAlign='LEFT'));story.append(Spacer(1,6))
def pct(x):return f'{100*x:.2f}%'
def n(x):return f'{x:,.0f}'
def modelname(r):return r['config']['name'].replace(' · ',' / ')
def runlist(f):return S[f]['experiments']
def frame(f,r):return pd.read_csv(E/f/r['config']['id']/'test_predictions.csv')
def bench(f,id):return next(b for b in M['benchmarks'] if b['fruit']==f and b['model']==id)
def foot(c,d):
 c.setFont('Calibri',8);c.setFillColor(colors.HexColor('#758075'));c.drawString(46,817,'RIPENESS LAB | LAPORAN TEKNIS DAN EVALUASI');c.drawString(46,25,'Audit data: 23 November 2026 | Nilai berasal dari artefak proyek dan pengukuran tambahan');c.drawRightString(549,25,str(d.page))

heading('01 / Dataset proyek')
para('<b>Ripeness Lab - Analisis model, eksperimen, error, dan kelayakan penggunaan nyata</b><br/>Laporan disusun dari kode, checkpoint, manifest, CSV/JSON hasil pelatihan, dan tangkapan layar Flask localhost:8000. Dataset ditempatkan pertama sesuai permintaan. Bahasa laporan: Indonesia. Tanggal pengambilan bukti: 23 November 2026.')
table([['Buah','Sebelum cleaning','Setelah cleaning','Dipakai','Train / Val / Test']]+[[f,n(s['audit']['downloaded_images']),n(s['audit']['retained_after_cleaning']),n(s['audit']['used']), ' / '.join(str(sum(s['audit']['counts'][k].values())) for k in ['train','val','test'])] for f,s in S.items()],[65,91,92,67,188])
for f,s in S.items():
 a=s['audit'];para(f'<b>{f.title()}</b>: '+escape(', '.join(s['classes']))+f'. Sumber: <link href="{a["source"]}" color="#245e4b">{escape(a["source"])}</link>',)
para('Apel dan jeruk berasal dari sumber yang sama tetapi menggunakan subset buah yang berbeda. Nanas menggunakan crop buah terbesar dari anotasi YOLO. Stroberi pada checkpoint aktif tetap menggunakan label tingkat gambar; dataset alternatif beranotasi yang terunduh tidak otomatis menjadi sumber model aktif.')
para('<b>Panduan isi:</b> dataset dan persiapan; arsitektur serta parameter; pretrained model; perbandingan lima buah; detail 20 eksperimen; screenshot dan error analysis; real-world suitability; lampiran seluruh prediksi test. CSV/JSON lengkap juga disematkan sebagai attachment PDF dan disertakan dalam paket bukti.')

for f,s in S.items():
 page('Dataset / '+f.title());a=s['audit']
 table([['Kelas','Awal train*','Train dipakai','Validation','Test']]+[[c,a['original_counts'].get('train',{}).get(c,'N/A'),a['counts']['train'][c],a['counts']['val'][c],a['counts']['test'][c]] for c in s['classes']])
 para('*N/A berarti sumber tidak menyediakan pemisahan train awal; untuk stroberi jumlah awal tercatat sebagai unsplit. Jumlah pada tabel adalah jumlah sampel, bukan jumlah physical fruit independen.')
 para('<b>Lisensi yang dinyatakan sumber:</b> '+escape(a['license'])+'. Status ini berasal dari metadata dataset; distribusi ulang dan penggunaan komersial perlu mengikuti ketentuan sumber, khususnya GPL 2 yang tercantum pada dataset nanas.')
 para('<b>Pembagian data:</b> '+escape(a['split_policy']))
 para('<b>Cleaning:</b> validasi citra, pengecualian citra terlalu kecil/tidak terbaca, pengelompokan source ID, pixel SHA-256 dan dHash; kelompok dengan label bertentangan dibuang. Satu perwakilan setiap grup dipertahankan. Daftar pengecualian tersimpan di rejected.csv.')
 para('<b>Missing data:</b> '+escape(a['missing_data']))
 para('<b>Batasan:</b> '+escape(a['limitation']))
 counts=list(a['counts']['train'].values());para(f'Rasio kelas train terbesar/terkecil: <b>{max(counts)/min(counts):.2f}</b>. CrossEntropyLoss saat training tidak diberi class weights dan tidak memakai weighted sampler. Macro F1 digunakan agar tiap kelas mendapat bobot evaluasi yang sama.')
 if f=='pineapple':para('Label sumber UNMATURED_PINEAPPLE dipetakan ke unripe, MATURED_PINEAPPLE ke ripe. Ini label kematangan sumber, bukan pengukuran gula, rasa, atau keamanan konsumsi. Foto inferensi harus dekat dengan satu buah karena classifier tidak melakukan deteksi/crop otomatis.')
 if f=='strawberry':para('Unripe, Semiripe, Fullripe dipetakan menjadi unripe, semi-ripe, ripe. Satu foto dapat memuat beberapa buah dengan kematangan berbeda: target tingkat gambar dapat menjadi ambigu. Validasi ulang anotasi dan crop per buah adalah prioritas perbaikan.')

page('02 / Persiapan data dan protokol bersama')
table([['Proses','Implementasi'],['Cleaning','Decode/load gambar; minimum sisi 32 px; duplikat source/pixel/dHash; konflik label ditolak.'],['Missing data','Gambar hilang/tidak terbaca dikeluarkan dan dicatat. Label tidak ditebak.'],['Resize','128 x 128, RGB, orientasi EXIF; bicubic.'],['Scaling dan normalisasi','ToTensor: 0..255 menjadi 0..1. Mean RGB [0.485,0.456,0.406], std [0.229,0.224,0.225].'],['Augmentation','Horizontal flip p=0.5 dan RandomRotation +/-12 derajat. Satu seeded view per gambar train digunakan identik pada semua kondisi.'],['Tokenization','Tidak relevan untuk CNN gambar yang dipakai. Tidak ada tokenizer teks/patch Transformer.'],['Split','Train untuk optimasi; validation untuk memilih checkpoint/model; test hanya evaluasi akhir tiap kondisi.']],[112,391])
para('Prosedur pelatihan: seed Python/NumPy/PyTorch 42; batch 32; 12 epoch; Adam (beta default 0.9/0.999, epsilon 1e-8), weight decay 0; cross-entropy rata-rata. Gradien di-clip pada norm 5 dan nilai non-finite menghentikan pelatihan. Tidak ada scheduler, early stopping, atau pengulangan banyak seed. Checkpoint dengan validation macro F1 tertinggi disimpan; tie mempertahankan checkpoint lebih awal.')
para('Train metrics dihitung saat model/head berada dalam train mode pada input augmentasi. Validation berada dalam eval mode tanpa augmentasi tambahan. Pada MobileNet, dropout aktif hanya untuk training head. Perbedaan mode dan input ini berarti train-validation gap tidak boleh ditafsirkan sebagai bukti tunggal overfitting.')
para('Ukuran asli pretrained biasa dievaluasi dengan resep 224 px; proyek memilih 128 px demi biaya CPU. Adaptive pooling membuat ukuran tersebut valid, tetapi detail tekstur dapat hilang. Angka benchmark ImageNet dari dokumentasi bukan hasil proyek ini.')

page('03 / Parameter dan perbedaan model')
table([['Parameter','CNN baseline','CNN LR rendah','ResNet18','MobileNetV3 Small'],['Arsitektur','3 blok Conv','Sama baseline','Residual CNN','Inverted residual CNN'],['Input','B x 3 x 128 x 128','Sama','Sama','Sama'],['Output','B x C logits','Sama','Sama','Sama'],['Loss','CrossEntropy','Sama','Sama','Sama'],['Optimizer','Adam','Adam','Adam head','Adam classifier'],['Learning rate','0.001','0.0001','0.001','0.001'],['Batch / epoch','32 / 12','32 / 12','32 / 12','32 / 12'],['Inisialisasi','Random','Random identik seed','ImageNet backbone','ImageNet backbone + fc awal'],['Yang dilatih','Seluruh parameter','Seluruh parameter','Linear 512 -> C','Linear 576 ->1024 ->C'],['Aktivasi','ReLU','ReLU','ReLU','ReLU, Hardswish, Hardsigmoid'],['Regularisasi','Augmentasi','Augmentasi','Augmentasi; frozen backbone','Augmentasi; dropout 0.2; frozen backbone']],[88,103,103,104,105])
para('C adalah jumlah kelas buah: pisang 4, apel/jeruk/stroberi 3, nanas 2. Training menerima logits langsung; CrossEntropyLoss menggabungkan log-softmax dan negative log likelihood. Softmax digunakan saat inferensi untuk menampilkan skor kelas, bukan probabilitas yang sudah dikalibrasi.')
table([['Jumlah kelas','CNN total/trainable','ResNet total / trainable','MobileNet total / trainable']]+[[c,n(next(r for r in runlist(f) if r['config']['id']=='baseline')['parameters']),n(runlist(f)[2]['parameters'])+' / '+n(runlist(f)[2]['trainable_parameters']),n(runlist(f)[3]['parameters'])+' / '+n(runlist(f)[3]['trainable_parameters'])] for f,c in [('banana',4),('apple',3),('pineapple',2)]],[70,125,154,154])

page('Arsitektur / CNN sederhana')
table([['Tahap','Komponen','Output per gambar','Parameter'],['Input','RGB','3 x 128 x 128','0'],['Blok 1','Conv 3->16, 3x3 pad1; ReLU; MaxPool2','16 x 64 x 64','448'],['Blok 2','Conv 16->32, 3x3 pad1; ReLU; MaxPool2','32 x 32 x 32','4,640'],['Blok 3','Conv 32->64, 3x3 pad1; ReLU; MaxPool2','64 x 16 x 16','18,496'],['Pooling','AdaptiveAvgPool2d(1), Flatten','64','0'],['Head','Linear 64 -> C','C logits','65 x C']],[65,225,123,90])
para('Seluruh bobot dipelajari dari dataset buah. Konvolusi mempelajari pola lokal seperti tepi, warna dan bercak; pooling mereduksi resolusi; global averaging merangkum fitur sebelum klasifikasi. Model ini ringan, tetapi kapasitas representasi dan keragaman fitur awal terbatas. Tidak ada BatchNorm, residual connection atau dropout pada CNN sederhana.')
para('Rumus parameter: Conv = (kernel_h x kernel_w x channel_in + bias) x channel_out. Contoh blok 1: (3 x 3 x 3 + 1) x 16 = 448. Total CNN = 23,584 + 65C. Kondisi learning rate rendah memiliki arsitektur dan jumlah parameter yang sama; yang berubah hanya besar langkah optimizer.')

page('Pretrained / ResNet18')
para('<b>Arsitektur:</b> stem Conv7x7 stride2 (3->64), BatchNorm, ReLU, MaxPool3 stride2; empat stage residual berisi [2,2,2,2] BasicBlock pada 64/128/256/512 channel. Setiap BasicBlock memuat dua Conv3x3 dan skip connection; downsample memakai Conv1x1 saat dimensi berubah. Global average pooling menghasilkan 512 fitur; final linear diganti menjadi C output. [R1,R3]')
para('<b>Pretraining objective dan domain:</b> supervised image classification pada ImageNet-1K, memprediksi 1.000 kategori objek. Bobot torchvision yang dipakai: ResNet18_Weights.IMAGENET1K_V1 (DEFAULT pada versi proyek 0.25.0). Fitur awal mengenali pola visual umum, bukan label kematangan buah secara langsung. [R1]')
para('<b>Transfer strategy:</b> seluruh backbone, termasuk affine BatchNorm, dibekukan dengan requires_grad=False. features.eval() menjaga statistik BatchNorm. Input train/validation dilewatkan sekali melalui backbone dan fitur 512 dimensi di-cache. Hanya fc 512->C dilatih. Ini feature extraction dengan head training, bukan fine-tuning seluruh jaringan.')
para('<b>Komponen yang dipakai:</b> semua convolution, BatchNorm, residual blocks, dan pooling pretrained. Classifier ImageNet asli dibuang dan head baru diinisialisasi random. Jumlah parameter head = 513C; trunk = 11,176,512. Learning rate 0.001; Adam; 12 epoch; batch 32; loss cross-entropy.')
para('<b>Alasan sesuai:</b> representasi lebih kaya dapat membantu dataset buah berukuran kecil dan mengurangi biaya backward karena hanya head yang dilatih. <b>Batasan:</b> domain ImageNet berbeda, backbone tetap tidak dapat beradaptasi dengan ciri kematangan khusus; ukuran 128 px mengurangi detail; 11 juta parameter tetap dibutuhkan saat inferensi. Performa perlu diuji pada varietas dan kondisi nyata yang baru.')

page('Pretrained / MobileNetV3 Small')
para('<b>Arsitektur:</b> stem Conv3x3 3->16 stride2; 11 inverted residual blocks; final pointwise Conv96->576; adaptive average pooling; classifier Linear576->1024, Hardswish, Dropout0.2, Linear1024->C. Blocks memakai expansion pointwise, depthwise convolution 3x3/5x5, projection pointwise, skip connection saat bentuk cocok, serta squeeze-and-excitation pada sebagian blocks. [R2,R4]')
para('<b>Pretraining objective/domain:</b> supervised ImageNet-1K 1.000 kategori. Bobot MobileNet_V3_Small_Weights.IMAGENET1K_V1 (DEFAULT lokal). Arsitektur aslinya dirancang dengan pertimbangan efisiensi perangkat mobile; proyek ini menjalankannya di CPU Windows, belum mengukur perangkat ponsel. [R2,R4]')
para('<b>Transfer strategy:</b> features dibekukan dan eval mode; fitur 576 dimensi di-cache. Seluruh classifier dilatih, termasuk Linear576->1024 yang mempertahankan inisialisasi pretrained dan final linear baru. Karena classifier mencakup dua linear, parameter trainable lebih besar daripada head ResNet18 meskipun total model lebih kecil.')
para('<b>Aktivasi:</b> ReLU pada beberapa blocks, Hardswish untuk nonlinearitas ringan, Hardsigmoid sebagai gating SE. Dropout p=0.2 bekerja saat classifier.train(), dinonaktifkan saat evaluasi. Loss, batch, epoch dan learning rate sama dengan kondisi transfer ResNet18.')
para('<b>Alasan sesuai:</b> model lebih kecil dan operasi depthwise mengurangi biaya dibanding residual convolution padat. <b>Batasan:</b> fitur tetap dapat gagal menangkap kematangan ambigu, skor tidak terkalibrasi, penurunan detail akibat resize, dan efisiensi aktual tergantung perangkat/runtime. Total = 1,517,856 + 1,025C; trainable = 590,848 + 1,025C.')

page('04 / Makna perubahan eksperimen dan metrik')
para('Baseline vs CNN LR rendah mengisolasi learning rate: 0.001 menjadi 0.0001 dengan epoch dan seed sama. Pada budget 12 epoch, langkah lebih kecil dapat membutuhkan waktu lebih lama untuk mencapai solusi. Penurunan skor pada budget ini tidak membuktikan bahwa learning rate kecil selalu buruk.')
para('Baseline vs model transfer mengubah arsitektur, inisialisasi, kedalaman dan bagian yang dilatih sekaligus. Kenaikan skor dapat berkaitan dengan fitur pretrained, tetapi eksperimen ini tidak memisahkan kontribusi setiap faktor. ResNet18 vs MobileNetV3 juga memiliki head berbeda, dropout berbeda, dan jumlah parameter trainable berbeda. Penjelasan sebab adalah hipotesis yang didukung pola hasil, bukan kesimpulan kausal dari ablation lengkap.')
para('Accuracy = jumlah benar / jumlah test. Precision kelas = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2PR/(P+R). Macro merata-ratakan tiap kelas; weighted average menggunakan support. Confusion matrix berbaris label benar dan berkolom prediksi. Selisih F1 sebesar 0.10 berarti 10 percentage points, bukan kenaikan relatif 10%.')
para('Pemilihan model menggunakan validation macro F1, tanpa mengganti pemenang setelah melihat test. Jeruk: MobileNet dipilih walau ResNet memiliki test F1 lebih tinggi. Nanas: ResNet dipilih walau MobileNet lebih tinggi pada test. Praktik ini menjaga test dari pemilihan ulang yang bias. Seluruh hasil memakai satu seed; tidak tersedia confidence interval antarrun atau pengujian signifikansi.')
table([['Buah','Pilihan validation','Val F1','Test accuracy','Test F1','Salah / N']]+[[f,s['selected_model'],f"{next(r for r in runlist(f) if r['config']['id']==s['selected_model'])['validation_f1']:.4f}",pct(next(r for r in runlist(f) if r['config']['id']==s['selected_model'])['test']['accuracy']),f"{next(r for r in runlist(f) if r['config']['id']==s['selected_model'])['test']['f1']:.4f}",f"{sum(~frame(f,next(r for r in runlist(f) if r['config']['id']==s['selected_model'])).correct)} / {sum(s['audit']['counts']['test'].values())}"] for f,s in S.items()])

for f,s in S.items():
 page('05 / Model comparison - '+f.title())
 runs=runlist(f);base=runs[0]['test']['f1']
 table([['Model','LR','Accuracy','P macro','R macro','F1 macro','Delta F1 pp']]+[[r['config']['id'],r['config']['lr'],pct(r['test']['accuracy']),pct(r['test']['precision']),pct(r['test']['recall']),f"{r['test']['f1']:.4f}",f"{100*(r['test']['f1']-base):+.2f}"] for r in runs],[87,49,70,70,70,75,82])
 para('Untuk setiap kondisi: baseline menjadi referensi; LR rendah menguji sensitivitas langkah optimizer; ResNet menguji fitur residual pretrained; MobileNet menguji alternatif kompak. Seluruh perubahan dan dampak di tabel berlaku dalam dataset buah ini saja.')
 for r in runs[1:]:
  para(f"<b>{modelname(r)}:</b> perubahan test macro F1 {100*(r['test']['f1']-base):+.2f} pp. Validation F1 terbaik {r['validation_f1']:.4f} pada epoch {r['best_epoch']}.")
 if f=='strawberry':para('<b>Temuan kritis:</b> transfer tidak memperbaiki test F1 dibanding baseline. MobileNet train accuracy mendekati 100% tetapi test accuracy 44.71%. Ini mendukung masalah generalisasi dan kemungkinan target tingkat gambar ambigu. Majority-class accuracy test adalah 37/85 = 43.53%; model terpilih hanya sedikit lebih tinggi. Prioritasnya audit label/crop dan dataset independen, bukan klaim siap pakai.')
 else:para('Fitur pretrained memberi perbaikan pada kondisi ini, konsisten dengan manfaat representasi awal pada dataset kecil. Namun test apel hanya 35 gambar dan jeruk 26; satu kesalahan mengubah accuracy masing-masing sekitar 2.86 dan 3.85 pp. Perubahan kecil belum cukup membuktikan keunggulan yang stabil.')
 pic(E/f'{f}_comparison_panel.png',330)
 for r in runs:
  id=r['config']['id'];d=frame(f,r);h=r['history'];b=bench(f,id)
  page(f'{f.title()} / {modelname(r)} / Training & diagnostics')
  para(f"Best checkpoint epoch <b>{r['best_epoch']}</b>; validation macro F1 <b>{r['validation_f1']:.4f}</b>. Total parameter {n(r['parameters'])}; trainable {n(r['trainable_parameters'])}. Waktu training tercatat {r['training_seconds']:.3f} detik. Waktu ini dimulai setelah konstruksi/download pretrained dan tidak mencakup persiapan data; transfer mencakup caching fitur serta optimasi head.")
  pic(E/f/id/'curves.png',155)
  table([['Ep','Tr loss','Val loss','Tr acc','Val acc','Tr F1','Val F1','Grad mean','Grad max']]+[[x['epoch'],*[f'{x[k]:.4f}' for k in ['train_loss','val_loss','train_accuracy','val_accuracy','train_f1','val_f1','gradient_mean','gradient_max']]] for x in h])
  for diag in r['diagnostics']:
   para(f"<b>{escape(diag['topic'])}: {'indikator muncul' if diag['flag'] else 'indikator tidak muncul'}</b>. {escape(diag['detail'])}")
  para('Flags bersifat heuristik. Unstable memakai nilai absolut perubahan loss sehingga perbaikan cepat pun dapat ter-flag. Plateau tidak membuktikan konvergensi optimal. Gradien hanya parameter trainable, bukan backbone yang sengaja dibekukan.')
  page(f'{f.title()} / {modelname(r)} / Test analysis')
  para(f"Test: accuracy {pct(r['test']['accuracy'])}; macro precision {pct(r['test']['precision'])}; macro recall {pct(r['test']['recall'])}; macro F1 {r['test']['f1']:.4f}. Benar {int(d.correct.sum())}/{len(d)}, salah {int((~d.correct).sum())}. Test loss tidak dicatat oleh pipeline dan tidak direka dalam laporan.")
  pic(E/f/id/'confusion_matrix.png',215)
  rows=[['Kelas / agregat','Precision','Recall','F1','Support']]
  for c in s['classes']+['macro avg','weighted avg']:
   v=r['test']['per_class'][c];rows.append([c,f"{v['precision']:.4f}",f"{v['recall']:.4f}",f"{v['f1-score']:.4f}",int(v['support'])])
  table(rows)
  mistakes=d[~d.correct];confusions=mistakes.groupby(['true_label','prediction']).size().sort_values(ascending=False)
  para('<b>Pola kebingungan:</b> '+('; '.join(f'{a} menjadi {z}: {v}' for (a,z),v in confusions.items()) if len(confusions) else 'Tidak ada salah klasifikasi pada subset ini.'))
  para(f"Batched inference historis: {r['test']['inference_ms_per_image_batched']:.3f} ms/gambar (bukan latensi request pengguna). Benchmark tambahan forward batch-1: median {b['p50_ms']:.3f} ms, p95 {b['p95_ms']:.3f} ms. Ukuran checkpoint {b['checkpoint_mib']:.3f} MiB.")
  if len(mistakes):
   table([['True','Prediksi','Skor','Brightness','Edge']]+[[x.true_label,x.prediction,f'{x.confidence:.4f}',f'{x.brightness:.1f}',f'{x.edge_detail:.1f}'] for x in mistakes.sort_values('confidence',ascending=False).head(3).itertuples()])
  para('Skor tinggi tidak menjamin benar. Hipotesis overlap visual perlu inspeksi manusia; brightness/edge adalah pengukuran sederhana, bukan bukti kausal. Setiap baris prediksi test, termasuk yang benar, dicatat pada lampiran akhir dan CSV attachment.')
 page(f'{f.title()} / Screenshot dashboard dan error visual')
 selected=s['selected_model'];d=frame(f,next(r for r in runs if r['config']['id']==selected))
 pic(E/f'{f}_{selected}_test_panel.png',440)
 para('Screenshot asli dari tab Test analysis untuk model pilihan validation. Seluruh 45 screenshot penuh (5 comparison, 20 training, 20 test) tersedia dalam paket bukti. PNG panel merupakan crop untuk keterbacaan; tidak mengubah nilai evaluasi.')
 chosen=pd.concat([d[~d.correct].sort_values('confidence',ascending=False).head(2),d[d.correct].head(1)])
 for idx,row in chosen.iterrows():
  with PILImage.open(ROOT/row.path) as im:
   im=im.convert('RGB');im.thumbnail((450,300));tmp=E/f'{f}_example_{idx}.jpg';im.save(tmp)
  story.append(KeepTogether([p(f"Sampel #{idx} | True: {row.true_label} | Prediksi: {row.prediction} | Skor: {row.confidence:.4f}",'Heading3'),Image(str(tmp),width=145,height=110,kind='proportional'),p(escape(row.analysis),'SmallReport')]))
 para('Contoh dipilih secara eksplisit: dua kesalahan berskor tertinggi dan satu prediksi benar. Pemilihan ini mengilustrasikan kegagalan, bukan estimasi prevalensi masalah. Untuk stroberi, adanya buah/latar yang beragam merupakan hipotesis ambiguitas; penyebab spesifik tidak dipastikan otomatis.')

page('06 / Real-world suitability: performance dan robustness')
para('<b>Kesimpulan praktis:</b> sistem layak sebagai prototipe edukasi dan uji coba terkontrol untuk buah yang dipilih manual. Belum cukup bukti untuk sortir industri, keputusan keamanan pangan, atau penggunaan otonom lintas varietas. Stroberi memerlukan perbaikan substansial sebelum penggunaan operasional.')
para('<b>Performance:</b> pisang menunjukkan hasil internal paling kuat pada test 518 gambar. Apel/jeruk berukuran test kecil; nanas memakai crop anotasi dan label dua kelas. Skor antarbuah tidak dapat dibandingkan langsung sebagai ranking kesulitan karena domain, kelas, dan ukuran sampel berbeda. Gunakan recall kelas yang paling merugikan bila salah, bukan accuracy saja.')
para('<b>Robustness:</b> berikut uji sensitivitas tambahan pada seluruh gambar test untuk model pilihan validation. Brightness 0.6/1.4 dan Gaussian blur radius2 diterapkan setelah resize128. Ini simulasi perubahan citra, bukan pengujian kebun/toko independen. Tidak ada retraining atau pemilihan ulang model dari hasil ini.')
rows=[['Buah / model','Kondisi','Accuracy','Macro F1','Delta F1 pp']]
for v in M['robustness']:
 base=v['conditions'][0]['f1']
 for z in v['conditions']:rows.append([v['fruit']+' / '+v['selected_model'],z['condition'],pct(z['accuracy']),f"{z['f1']:.4f}",f"{100*(z['f1']-base):+.2f}"])
table(rows,[135,125,80,80,83])
para('Penurunan saat transformasi menunjukkan sensitivitas terhadap pencahayaan/detail. Kenaikan pada satu transformasi tidak membuktikan ketahanan umum. Occlusion, variasi kamera, latar baru, varietas, musim, objek nonbuah dan mixed-stage bunch belum diuji terpisah. Softmax dan threshold 0.65 belum dikalibrasi untuk menolak input di luar domain.')

page('Computational cost / seluruh eksperimen')
para('Training time adalah catatan historis; batch inference berasal dari loop test historis. Median/p95 baru diukur 23 November 2026 pada proses terpisah per model: 5 warmup, 30 forward batch-1, 4 thread CPU, input tensor sudah siap. Decode/resize, jaringan, cold load dan antrian tidak termasuk. Karena protokol berbeda, jangan membandingkan batch ms/gambar dengan request latency secara langsung.')
table([['Buah / model','Train detik','Batch ms/img','P50 ms','P95 ms','File MiB']]+[[b['fruit']+' / '+b['model'],f"{next(r for r in runlist(b['fruit']) if r['config']['id']==b['model'])['training_seconds']:.3f}",f"{next(r for r in runlist(b['fruit']) if r['config']['id']==b['model'])['test']['inference_ms_per_image_batched']:.3f}",f"{b['p50_ms']:.3f}",f"{b['p95_ms']:.3f}",f"{b['checkpoint_mib']:.3f}"] for b in M['benchmarks']],[153,70,75,65,65,75])
page('Computational cost / memori dan hardware')
para('Hardware pengukuran saat ini: '+escape(str(M['hardware']))+'. Identitas CPU/load saat pelatihan historis tidak lengkap, sehingga kecepatan lintas run tidak boleh dianggap benchmark hardware terkontrol.')
table([['Buah / model','Bobot FP32 MiB','RSS awal MiB','RSS setelah MiB','Peak proses MiB']]+[[b['fruit']+' / '+b['model'],f"{b['param_mib']:.3f}",f"{b['rss_base_mib']:.1f}",f"{b['rss_after_mib']:.1f}",f"{b['peak_process_mib']:.1f}"] for b in M['benchmarks']],[145,85,90,90,93])
para('RSS/peak adalah memori seluruh proses Python termasuk import PyTorch, checkpoint state, allocator dan aktivasi. Bukan memori model murni, GPU VRAM, atau peak training. Bobot FP32 dihitung numel x 4 byte, tidak termasuk buffer. Peak training historis tidak tercatat; tidak ada angka biaya energi/listrik yang terukur.')
para('GPU tidak wajib untuk inferensi atau head training; versi yang digunakan CPU. Kebutuhan minimum RAM belum diprofilkan sebagai batas. Training menyimpan PIL images, view augmentasi, tensor stack dan feature cache sehingga konsumsi lebih tinggi daripada inferensi satu gambar. Dataset pisang saja memerlukan sekitar 300 MiB untuk satu tensor float32 seluruh 1.600 input, sebelum duplikasi/cache/optimizer.')

page('Scalability, limitations dan practical suitability')
para('<b>Scalability:</b> Flask lokal memakai development server dan satu threading.Lock untuk load/cache/inferensi. Request inferensi terserialisasi melalui lock; beberapa request bersamaan dapat mengantre. Cache menyimpan model berdasarkan (fruit, model) tanpa eviction. Memuat semua 20 checkpoint meningkatkan RAM; model serupa untuk buah berbeda masih diduplikasi. Belum ada load test, QPS, concurrency p95, SLA atau autoscaling yang diukur.')
para('Arah deployment: server WSGI produksi yang sesuai platform, pembatasan ukuran request, logging, model registry/versioning, cache terbatas, antrian/batching dan evaluasi concurrency. Multiworker menggandakan memori model; GPU atau quantization/ONNX perlu benchmark akurasi dan latensi baru. Model multibuah bersama dapat mengurangi duplikasi, tetapi memerlukan dataset dan evaluasi baru.')
para('<b>Limitations:</b> pemilihan jenis buah manual; tidak ada detektor identitas atau OOD rejection; resolusi kecil; label visual tidak mengukur kadar gula/internal ripeness; kelas rotten adalah spoilage visual. Dataset publik dapat mengandung frame terkait; hash identik tidak menjamin bebas near-duplicate/physical-fruit leakage. Dataset nanas train berupa crop terarah, sementara upload tidak dicrop otomatis. Stroberi mixed-stage scenes dan generalization gap sangat besar. Satu seed tidak cukup untuk signifikansi statistik.')
para('<b>Practical suitability:</b> model akademik terbaik belum tentu pilihan deployment. MobileNet dipilih untuk pisang/apel/jeruk/stroberi berdasarkan validation, ResNet untuk nanas. Model lebih kecil bisa cocok jika RAM/latensi dibatasi, namun stroberi tidak layak hanya karena model cepat. Pemenang test berbeda pada jeruk/nanas tidak boleh dijadikan alasan mengganti pilihan setelah test. Pilih dengan validasi lapangan, biaya kesalahan, latensi end-to-end dan resource target.')
para('<b>Sustainability operasional:</b> frozen features menghemat backward training, CPU memungkinkan demo tanpa GPU, tetapi energi/CO2 tidak diukur sehingga klaim hemat energi numerik tidak tersedia. Sustained real-life use membutuhkan pemantauan drift, audit error, label dari ahli, kalibrasi, retraining terjadwal berbasis kebutuhan dan human fallback.')
table([['Aspek','Status sekarang','Bukti lanjutan yang diperlukan'],['Performance','Internal; stroberi lemah','Test independen per fruit/varietas + repeated seed'],['Robustness','Uji brightness/blur terbatas','Kamera/latar/cuaca/occlusion/OOD'],['Cost','Training log + benchmark CPU','Request end-to-end dan peak training'],['Scalability','Local prototype','Load test, cache, worker/memory planning'],['Deployment','Pilot terkontrol','Target SLA, monitoring dan human review']],[86,174,243])

page('07 / Referensi, jejak bukti dan cara membaca lampiran')
refs=[('R1','Torchvision ResNet18','https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html'),('R2','Torchvision MobileNetV3 Small','https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html'),('R3','He et al., Deep Residual Learning for Image Recognition','https://arxiv.org/abs/1512.03385'),('R4','Howard et al., Searching for MobileNetV3','https://arxiv.org/abs/1905.02244')]
for k,t,u in refs:para(f'[{k}] {t}. <link href="{u}" color="#245e4b">{u}</link>')
para('Informasi arsitektur terperinci dan parameter proyek diverifikasi dari models.py dan torchvision lokal 0.25.0, bukan menganggap dokumentasi stable terbaru identik dengan environment training. Data hasil bersumber langsung dari API summary/errors dan dibandingkan dengan artefak lokal. Tidak ada skor yang diambil dari hasil paper ImageNet sebagai skor kematangan buah.')
para('Sumber kode: models.py, train.py, prepare.py, prepare_multifruit.py, fruits.py, app.py dan web/app.js. Bukti: audit.json, manifest.csv, history.csv, results.json, summary.json, comparison.csv, test_predictions.csv, curves.png dan confusion_matrix.png. Pengukuran tambahan: report_measurements.py dan measurements.json. Snapshot API, screenshot asli dan salinan CSV berada di folder bukti.')
para('Lampiran berikut mencatat semua baris test untuk seluruh 20 eksperimen. ID adalah indeks nol pada test_predictions.csv masing-masing model; urutan sama dengan manifest test. B=brightness rata-rata 0..255; E=edge detail. Kode analisis: OK=Correct; V=possible visual overlap/human review; D=low brightness; L=low edge detail. Kode adalah ringkasan teks aplikasi, bukan diagnosis sebab. Path gambar lengkap dan teks analisis utuh tersedia dalam attachment CSV/JSON serta paket ZIP. Thumbnail contoh tercantum pada bagian error visual.')

for f in S:
 for r in runlist(f):
  id=r['config']['id'];d=frame(f,r)
  page(f'Lampiran / Semua prediksi test / {f} / {id}')
  para(f'{len(d)} sampel; semua prediksi benar dan salah. Skor berupa softmax uncalibrated. Angka dibulatkan di PDF; file CSV mempertahankan presisi tersimpan.')
  rows=[['ID','True label','Prediction','Score','B','E','Analisis']]
  for i,x in d.iterrows():
   code='OK' if x.correct else 'D' if 'brightness' in x.analysis else 'L' if 'edge' in x.analysis else 'V'
   rows.append([i,x.true_label,x.prediction,f'{x.confidence:.4f}',f'{x.brightness:.2f}',f'{x.edge_detail:.2f}',code])
  table(rows,[40,105,105,70,65,65,53])

doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=46,leftMargin=46,topMargin=45,bottomMargin=43,title='Ripeness Lab - Laporan Analisis dan Evaluasi',author='Ripeness Lab Project',pageCompression=1)
doc.build(story,onFirstPage=foot,onLaterPages=foot)

d=fitz.open(PDF)
for file in [E/'all_summaries.json',E/'measurements.json']+list(E.glob('*/*/test_predictions.csv'))+list(E.glob('*/*/history.csv')):
 name=file.relative_to(E).as_posix().replace('/','__');d.embfile_add(name,file.read_bytes(),filename=name)
temp=PDF.with_suffix('.tmp.pdf');d.save(temp,garbage=4,deflate=True);pages=len(d);d.close();temp.replace(PDF)
destination=ROOT.parent.parent/'DeepLearning';destination.mkdir(exist_ok=True)
shutil.copy2(PDF,destination/PDF.name)
with zipfile.ZipFile(destination/'Bukti_Evaluasi_Ripeness_Lab.zip','w',zipfile.ZIP_DEFLATED) as z:
 for file in E.rglob('*'):
  if file.is_file():z.write(file,'bukti/'+file.relative_to(E).as_posix())
print('PDF',destination/PDF.name,'pages',pages,'bytes',PDF.stat().st_size)
