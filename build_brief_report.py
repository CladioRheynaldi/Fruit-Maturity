import json
from pathlib import Path
from xml.sax.saxutils import escape
import pandas as pd
from PIL import Image as PIL
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
ROOT=Path(__file__).resolve().parent
E=ROOT.parent/'Laporan_Ripeness_Lab/bukti'
S=json.loads((E/'all_summaries.json').read_text())
M=json.loads((E/'measurements.json').read_text())
pdfmetrics.registerFont(TTFont('Cal','C:/Windows/Fonts/calibri.ttf'))
pdfmetrics.registerFont(TTFont('CalB','C:/Windows/Fonts/calibrib.ttf'))
pdfmetrics.registerFontFamily('Cal',normal='Cal',bold='CalB')
ss=getSampleStyleSheet()
ss['BodyText'].fontName='Cal';ss['BodyText'].fontSize=10;ss['BodyText'].leading=13;ss['BodyText'].spaceAfter=7
ss['Heading1'].fontName='CalB';ss['Heading1'].fontSize=18;ss['Heading1'].textColor=colors.HexColor('#245440')
ss.add(ParagraphStyle(name='Cell',fontName='Cal',fontSize=8,leading=10))
story=[]
def p(t):return Paragraph(t,ss['BodyText'])
def add(t):story.append(p(t))
def head(t,new=True):
 if new:story.append(PageBreak())
 story.append(Paragraph(t,ss['Heading1']))
def tab(rows,widths=None):
 t=Table([[Paragraph(escape(str(x)),ss['Cell']) for x in row] for row in rows],colWidths=widths or [503/len(rows[0])]*len(rows[0]),repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfebdc')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#d8dfd5'))]));story.extend([t,Spacer(1,9)])
def img(path,w,h):
 with PIL.open(path) as im:a,b=im.size
 k=min(w/a,h/b);return Image(str(path),width=a*k,height=b*k)
def selected(f):return next(r for r in S[f]['experiments'] if r['config']['id']==S[f]['selected_model'])
def pct(v):return f'{v*100:.2f}%'
head('1. Dataset yang digunakan',False)
add('<b>RIPENESS LAB | Ringkasan enam aspek proyek</b><br/>Lima buah; empat eksperimen per buah; Flask + PyTorch. Data dan screenshot diambil 23 November 2026. Urutan dimulai dari dataset sesuai permintaan awal.')
tab([['Buah','Label kematangan','Train','Val','Test']]+[[f,', '.join(s['classes']),*[sum(s['audit']['counts'][k].values()) for k in ['train','val','test']]] for f,s in S.items()],[65,255,61,61,61])
for f,s in S.items():
 a=s['audit'];add(f'<b>{f.title()}</b> - <link href="{a["source"]}" color="#245440">{escape(a["source"])}</link><br/>Lisensi menurut sumber: {escape(a["license"])}.')
add('<b>Persiapan:</b> validasi gambar, penghapusan duplikat/konflik label, pengecualian file hilang atau rusak, RGB dan resize 128 x 128, scaling 0-1 serta normalisasi ImageNet. Augmentasi training: horizontal flip dan rotasi +/-12 derajat. Tokenization tidak relevan untuk CNN ini.')
add('<b>Split:</b> pisang mempertahankan split sumber dengan cap per kelas; apel/jeruk/nanas mempertahankan test sumber lalu mengambil validation dari train. Stroberi dibagi sekitar 70/15/15 setelah cleaning. Nanas menggunakan crop buah dari anotasi YOLO; stroberi menggunakan label per gambar. Foto berurutan/shared fruit masih dapat menyebabkan kebocoran yang tidak terdeteksi hash.')

head('2. Arsitektur dan parameter model')
tab([['Parameter','Small CNN','ResNet18','MobileNetV3 Small'],['Input','RGB 3 x 128 x 128','Sama','Sama'],['Output','C logits; softmax saat inference','Sama','Sama'],['Komponen utama','Conv3x3 16/32/64; ReLU; MaxPool; global avg pool; Linear64->C','Stem Conv7; 8 residual BasicBlocks; channel64/128/256/512; global avg pool; Linear512->C','11 inverted residual blocks; depthwise conv; SE; global avg pool; Linear576->1024->C'],['Aktivasi','ReLU','ReLU','ReLU, Hardswish, Hardsigmoid'],['Loss','CrossEntropyLoss','Sama','Sama'],['Optimizer','Adam','Adam pada head','Adam pada classifier'],['Learning rate','Baseline0.001; eksperimen0.0001','0.001','0.001'],['Batch size / epoch','32 / 12','32 / 12','32 / 12'],['Parameter dilatih','Seluruh model','Linear terakhir','Dua linear classifier'],['Regularisasi','Augmentasi','Augmentasi; backbone frozen','Augmentasi; backbone frozen; dropout0.2']],[95,136,136,136])
add('C = 4 untuk pisang, 3 untuk apel/jeruk/stroberi, dan 2 untuk nanas. Kondisi CNN learning rate rendah memakai arsitektur identik dengan baseline.')
tab([['C','CNN total/trainable','ResNet total / trainable','MobileNet total / trainable']]+[[len(S[f]['classes']),f"{S[f]['experiments'][0]['parameters']:,}",f"{S[f]['experiments'][2]['parameters']:,} / {S[f]['experiments'][2]['trainable_parameters']:,}",f"{S[f]['experiments'][3]['parameters']:,} / {S[f]['experiments'][3]['trainable_parameters']:,}"] for f in ['banana','apple','pineapple']])
add('<b>Training procedure:</b> seed42, Adam tanpa weight decay/scheduler, clip gradient norm5. Satu view augmentasi per gambar digunakan tetap untuk perbandingan. CNN dilatih end-to-end; backbone transfer di-eval dan fiturnya di-cache. Checkpoint terbaik dipilih dari validation macro F1; test digunakan untuk evaluasi. Tidak ada early stopping: setiap kondisi tetap12 epoch.')

head('3. Penjelasan pretrained model')
for title,text in [
('ResNet18','Arsitektur residual dengan skip connection: stem Conv7x7, empat stage berisi [2,2,2,2] BasicBlocks, pooling dan classifier. Skip connection mengalirkan fitur melalui jalur identitas dan memudahkan optimasi jaringan dalam. Proyek membekukan seluruh backbone dan mengganti classifier 1.000 kelas dengan Linear512->C. Hanya head baru yang dilatih.'),
('MobileNetV3 Small','Arsitektur efisien memakai inverted residual, depthwise/pointwise convolution, squeeze-and-excitation serta Hardswish. Backbone dibekukan. Classifier Linear576->1024 pretrained dipertahankan dan dilatih bersama final Linear1024->C baru, dengan dropout0.2. Karena head lebih besar, parameter trainable melebihi head ResNet meskipun total model lebih kecil.')]:add(f'<b>{title}</b><br/>{text}')
tab([['Aspek pretrained','Penjelasan'],['Pretraining objective','Supervised classification: mengenali 1.000 kategori ImageNet-1K, bukan menilai kematangan buah.'],['Dataset/domain','ImageNet-1K, citra objek umum. Bobot lokal DEFAULT=IMAGENET1K_V1 untuk kedua model.'],['Transfer / fine-tuning','Frozen feature extraction + training head. Tidak melakukan fine-tuning seluruh backbone; BatchNorm backbone tetap eval mode.'],['Komponen yang dipakai','Convolution, blocks dan pooling pretrained; head output ImageNet diganti sesuai jumlah kelas buah.'],['Mengapa sesuai','Fitur visual awal dapat membantu dataset kecil; caching fitur mempercepat training CPU. MobileNet memberikan alternatif lebih kecil; ResNet memberi residual features yang lebih besar.'],['Batasan','Domain berbeda; fitur frozen tidak beradaptasi penuh; resize128 menghilangkan detail. Confidence belum dikalibrasi dan tidak mengenali input di luar domain.']],[125,378])
add('Model Small CNN memakai bobot acak. Pada pretrained, head ResNet baru bersifat acak; MobileNet mempertahankan layer classifier awal pretrained. Angka ImageNet pada publikasi tidak digunakan sebagai skor proyek ini.')
add('Referensi: <link href="https://arxiv.org/abs/1512.03385">He et al., Deep Residual Learning</link>; <link href="https://arxiv.org/abs/1905.02244">Howard et al., Searching for MobileNetV3</link>. Implementasi diverifikasi dari models.py dan torchvision0.25.0 lokal.')

head('4. Perbedaan model dan perubahan hasil')
tab([['Buah','Eksperimen','Accuracy','P macro','R macro','F1','Delta F1 pp']]+[[f,r['config']['id'],pct(r['test']['accuracy']),f"{r['test']['precision']:.3f}",f"{r['test']['recall']:.3f}",f"{r['test']['f1']:.4f}",f"{100*(r['test']['f1']-s['experiments'][0]['test']['f1']):+.2f}"] for f,s in S.items() for r in s['experiments']],[65,95,72,65,65,70,71])
add('<b>Learning rate:</b> menurunkan0.001 menjadi0.0001 menurunkan F1 pada semua buah dalam budget12 epoch. Langkah lebih kecil mungkin belum mencapai solusi yang baik; ini tidak membuktikan learning rate kecil selalu buruk.')
add('<b>Arsitektur/transfer:</b> fitur pretrained meningkatkan hasil pisang/apel/jeruk/nanas dibanding baseline, tetapi tidak untuk stroberi. Architecture, initialization dan trainable layers berubah bersama, sehingga penyebab tiap faktor tidak terisolasi. Stroberi menunjukkan generalization gap besar dan label gambar berpotensi ambigu; ini hipotesis yang perlu audit.')
add('Pilihan validation: MobileNet untuk pisang/apel/jeruk/stroberi; ResNet untuk nanas. Test tertinggi jeruk adalah ResNet dan nanas MobileNet; pilihan tidak diganti dari test. Perbedaan skor berasal dari prediksi pada test yang sama untuk setiap buah. Satu seed dan test apel35/jeruk26 membuat hasil rentan variasi sampel.')

head('5. Hasil dashboard, diagnostik dan error')
add('Ringkasan tab Model comparison ada pada bagian4. Berikut ringkasan Training & diagnostics serta Test analysis untuk model pilihan validation. Indikator dihitung dari riwayat12 epoch, bukan hanya checkpoint terpilih.')
tab([['Buah / model','Best epoch','Val F1','Train acc akhir','Val acc akhir','Test salah/N','Indikator muncul']]+[[f+' / '+s['selected_model'],r['best_epoch'],f"{r['validation_f1']:.4f}",pct(r['history'][-1]['train_accuracy']),pct(r['history'][-1]['val_accuracy']),f"{round((1-r['test']['accuracy'])*sum(s['audit']['counts']['test'].values()))}/{sum(s['audit']['counts']['test'].values())}",', '.join(d['topic'] for d in r['diagnostics'] if d['flag']) or 'Tidak ada'] for f,s in S.items() for r in [selected(f)]],[87,44,50,66,66,63,127])
add('<b>Interpretasi:</b> overfitting=gap akurasi &gt;0.10 atau val loss naik&gt;15% sementara train loss turun; underfitting=kedua akurasi&lt;0.70; unstable=perubahan absolut val loss&gt;0.30; convergence=rentang tiga val loss akhir&lt;0.025. Vanishing/large gradient tidak ter-flag pada model pilihan. Semua flags adalah heuristik, bukan bukti pasti.')
story.append(img(E/'banana_mobilenet_training_panel.png',503,260))
add('Screenshot Training & diagnostics: MobileNet pisang. Kurva menunjukkan train loss terus turun sedangkan validation loss tidak mengikuti; checkpoint epoch9 dipilih sebelum akhir training. Train dan validation memakai mode/input berbeda, sehingga gap harus dibaca bersama kurva.')
head('5. Hasil dashboard dan error (lanjutan)')
story.append(img(E/'strawberry_mobilenet_test_panel.png',503,265))
add('Screenshot Test analysis: MobileNet stroberi. Accuracy44.71%, macro F1 0.3781;47 dari85 gambar salah. Model cepat dan train accuracy tinggi belum menunjukkan generalisasi yang baik.')
tab([['Buah','True','Prediksi','Skor','Analisis']]+[[f,row.true_label,row.prediction,f'{row.confidence:.4f}','Salah; overlap visual/ambiguitas label perlu review, bukan penyebab yang terbukti.'] for f in S for row in [pd.read_csv(E/f/S[f]['selected_model']/'test_predictions.csv').query('correct == False').sort_values('confidence',ascending=False).iloc[0]]],[60,65,65,55,258])
add('Contoh aktual diambil dari kesalahan dengan skor tertinggi tiap buah. Skor tinggi tetap dapat salah. Brightness/edge detail pada aplikasi adalah indikator kualitas sederhana; sistem tidak membuktikan occlusion atau penyebab kesalahan secara otomatis.')
pics=[]
for f in ['banana','strawberry','pineapple']:
 d=pd.read_csv(E/f/S[f]['selected_model']/'test_predictions.csv');row=d[~d.correct].sort_values('confidence',ascending=False).iloc[0]
 pics.append([p(f'<b>{f}</b>: {row.true_label} menjadi {row.prediction}'),img(ROOT/row.path,150,95)])
story.append(Table([[x[0] for x in pics],[x[1] for x in pics]],colWidths=[167]*3))

head('6. Kelayakan penggunaan dunia nyata')
tab([['Aspek','Penilaian proyek'],['Performance','Pisang:95.56% accuracy; apel94.29%; jeruk92.31%; nanas90.28%; stroberi44.71% pada model pilihan validation. Test berbeda antarbuah; bukan ranking langsung. Stroberi perlu perbaikan sebelum penggunaan operasional.'],['Robustness','Uji tambahan brightness0.6/1.4 dan blur radius2 dilakukan pada test yang ada. Ini simulasi, bukan validasi kebun/toko independen. Kamera, occlusion, varietas dan latar baru belum diuji sistematis.'],['Computational cost','Training frozen-backbone cepat karena hanya head dilatih; waktu tercatat mengecualikan persiapan data/download. Memori proses memuat PyTorch/cache/aktivasi, bukan bobot saja.'],['Scalability','Flask development server + satu lock inferensi; request dapat mengantre. Cache model tidak memiliki eviction. Belum ada load test/QPS/SLA. Produksi perlu server WSGI, cache terbatas dan benchmark concurrency.'],['Limitations','Jenis buah dipilih manual; tidak ada deteksi identitas/OOD; foto mixed-stage, resize128, label ambigu, shared-fruit leakage dan satu seed membatasi generalisasi. Tidak mengukur rasa/gula/keamanan pangan.'],['Practical suitability','Cocok untuk demo edukasi/pilot terkontrol. Pemenang akademik belum tentu deployment terbaik: pilih berdasarkan recall kelas kritis, latency/RAM target dan validasi lapangan. MobileNet lebih kecil; ResNet tidak otomatis lebih baik untuk semua buah.']],[105,398])
tab([['Buah/model pilihan','Train s','Forward P50 ms','Checkpoint MiB','RSS proses MiB']]+[[f+'/'+s['selected_model'],f"{selected(f)['training_seconds']:.2f}",f"{b['p50_ms']:.2f}",f"{b['checkpoint_mib']:.2f}",f"{b['rss_after_mib']:.1f}"] for f,s in S.items() for b in [next(v for v in M['benchmarks'] if v['fruit']==f and v['model']==s['selected_model'])]])
add('Benchmark tambahan: CPU Windows,4 thread;5 warmup+30 forward batch1; tidak termasuk decode/network/antrian. RSS seluruh proses, bukan peak training. Model FP32: CNN sekitar0.09MiB, ResNet42.64MiB, MobileNet5.80MiB. GPU tidak wajib; minimum RAM deployment belum dipastikan. Hardware saat ukur:16 core fisik/32 logical, RAM15.21GiB.')
add('<b>Apakah sustain untuk real life?</b> Belum terbukti untuk penggunaan luas. Caching/frozen features mengurangi biaya training, tetapi energi/CO2 tidak diukur. Diperlukan test independen per physical fruit/session, audit label stroberi, kalibrasi, monitoring drift, human fallback dan pembaruan model berbasis kebutuhan.')

dest=ROOT.parent.parent/'DeepLearning/Laporan_Ringkas_6_Aspek_Ripeness_Lab.pdf'
def footer(c,d):
 c.setFont('Cal',8);c.setFillColor(colors.HexColor('#667667'));c.drawString(46,25,'RIPENESS LAB | Ringkasan 6 aspek | Data: 23 November 2026');c.drawRightString(549,25,str(d.page))
SimpleDocTemplate(str(dest),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=40,bottomMargin=42,title='Ripeness Lab - Ringkasan Enam Aspek').build(story,onFirstPage=footer,onLaterPages=footer)
print(dest)
