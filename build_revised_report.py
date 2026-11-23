from pathlib import Path
exec((Path(__file__).parent/'build_brief_report.py').read_text(encoding='utf-8').split("head('1. Dataset")[0])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz
from reportlab.platypus import KeepTogether
NAMES={'baseline':'CNN baseline','cnn_low_lr':'CNN eksperimen','resnet18':'ResNet18','mobilenet':'MobileNetV3 Small'}
FR={'banana':'Pisang','apple':'Apel','orange':'Jeruk','strawberry':'Stroberi','pineapple':'Nanas'}
DIAG={'Overfitting':'Overfitting','Underfitting':'Underfitting','Unstable training':'Training tidak stabil','Convergence':'Mulai mendatar','Vanishing gradients':'Gradien terlalu kecil','Large / exploding gradients':'Gradien sangat besar'}
G=E/'grafik_revisi';G.mkdir(exist_ok=True)
head('1. Dataset proyek',False)
add('<b>Laporan revisi - Ripeness Lab</b><br/>Enam pokok pembahasan: dataset; arsitektur dan pengaturan model; perbandingan pretrained; hasil eksperimen; proses training dan hasil pengujian; kelayakan penggunaan nyata. Angka diambil dari hasil proyek yang tersimpan, bukan hasil perkiraan.')
tab([['Buah','Kategori yang diprediksi','Training','Validasi','Pengujian']]+[[FR[f],', '.join(s['classes']),*[sum(s['audit']['counts'][k].values()) for k in ['train','val','test']]] for f,s in S.items()],[60,230,71,71,71])
for f in S:
 a=S[f]['audit'];add(f'<b>{FR[f]}</b><br/><link href="{a["source"]}" color="#245440">{escape(a["source"])}</link><br/>Lisensi menurut sumber: {escape(a["license"])}.')
add('<b>Persiapan data:</b> gambar rusak/hilang dikeluarkan, duplikat dan konflik label dibersihkan, gambar diubah menjadi RGB ukuran128 x128 piksel. Nilai piksel dibagi255, lalu dinormalisasi dengan mean/std ImageNet. Augmentasi hanya untuk training: flip horizontal dan rotasi sampai12 derajat. Tokenization tidak diperlukan untuk CNN.')
add('Data training dipakai untuk belajar; data validasi untuk memilih checkpoint dan model; data pengujian untuk mengukur hasil akhirnya. Jenis buah dipilih pengguna. Foto stroberi dapat berisi beberapa buah dengan tingkat matang berbeda; nanas dilatih memakai potongan gambar dari kotak anotasi.')

head('2. Arsitektur dan pengaturan model')
tab([['Pengaturan','CNN baseline','CNN eksperimen'],['Arsitektur','3 blok convolution','Sama dengan baseline'],['Input','Gambar RGB128 x128 piksel','Sama'],['Output','Kategori kematangan buah','Sama'],['Layer utama','Conv3x3:16,32,64 filter; MaxPool; average pooling; linear','Sama'],['Aktivasi','ReLU','ReLU'],['Loss function','CrossEntropyLoss','CrossEntropyLoss'],['Optimizer','Adam','Adam'],['Learning rate','0,001','0,0001'],['Batch size','32 gambar','32 gambar'],['Jumlah epoch','12','12'],['Bobot awal','Acak; seed42','Acak; seed42'],['Parameter yang dilatih','Seluruh model','Seluruh model']],[123,190,190])
add('<b>Apa bedanya dan mengapa dibagi dua?</b> Baseline adalah hasil acuan. Eksperimen CNN adalah uji pengaturan baru pada arsitektur yang sama. Hanya learning rate yang diubah agar kita dapat melihat pengaruh perubahan tersebut. Ini dua kondisi percobaan, bukan dua jenis arsitektur CNN yang berbeda.')
add('Learning rate0,001 membuat langkah pembaruan bobot lebih besar;0,0001 membuatnya10 kali lebih kecil. Pada12 epoch yang sama, learning rate lebih kecil mungkin belum cukup lama belajar. Karena itu hasilnya bisa lebih rendah. Perlu training lebih panjang untuk menguji apakah selisih itu bertahan.')
add('<b>Urutan training:</b> baca gambar dan label, preprocessing, prediksi model, hitung loss, hitung gradien, lalu Adam memperbarui bobot. Evaluasi pada validasi dilakukan setiap epoch. Gradien dibatasi pada norm5. Semua kondisi menjalani12 epoch; checkpoint dipilih dari F1 validasi terbaik, bukan otomatis epoch terakhir.')
add('Output mengikuti kategori buah pada bagian1. CrossEntropyLoss menerima nilai keluaran model sebelum softmax. Saat prediksi ditampilkan ke pengguna, softmax mengubahnya menjadi skor tiap kategori.')

head('3. Perbandingan ResNet dan MobileNet')
tab([['Yang dibandingkan','ResNet18','MobileNetV3 Small'],['Arsitektur','Residual blocks: jalur utama + skip connection','Inverted residual blocks dan depthwise convolution'],['Komponen utama','Stem Conv7x7;8 BasicBlocks; pooling; linear','11 blocks; squeeze-and-excitation; pooling;2 linear'],['Aktivasi','ReLU','ReLU dan Hardswish; Hardsigmoid pada SE'],['Data pretrained','ImageNet-1K,1.000 kategori objek','ImageNet-1K,1.000 kategori objek'],['Tujuan pretraining','Mengenali kategori objek umum','Mengenali kategori objek umum'],['Bobot yang digunakan','IMAGENET1K_V1','IMAGENET1K_V1'],['Strategi transfer','Backbone dibekukan; linear terakhir dilatih','Backbone dibekukan; seluruh classifier dilatih'],['Classifier proyek','512 fitur -> kategori kematangan','576 fitur ->1.024 fitur -> kategori kematangan'],['Dropout','Tidak ada pada classifier','0,2 pada classifier'],['Loss / optimizer','CrossEntropy / Adam','CrossEntropy / Adam'],['LR / batch / epoch','0,001 /32 /12','0,001 /32 /12'],['Ukuran bobot FP32','Sekitar42,64MiB','Sekitar5,80MiB'],['Kelebihan yang dicari','Fitur residual untuk klasifikasi','Model lebih kecil dan inferensi ringan'],['Batasan','Lebih besar; backbone tidak menyesuaikan fitur buah','Fitur lebih kompak; belum tentu paling akurat']],[128,187,188])
add('<b>Cara transfer learning:</b> kedua model memakai convolution dan pooling pretrained. Classifier1.000 kategori diganti untuk kategori kematangan buah. ResNet melatih satu linear baru. MobileNet melatih linear awal classifier yang sudah pretrained dan linear terakhir yang baru. Fitur backbone dihitung sekali lalu disimpan sementara agar training CPU lebih cepat.')
add('Keduanya sesuai sebagai percobaan pada dataset kecil karena sudah memiliki fitur visual awal. Keduanya tetap dapat gagal ketika foto baru berbeda dari data training. Pada proyek ini tidak dilakukan fine-tuning seluruh backbone. Pretraining ImageNet tidak berarti model sudah mengetahui kematangan buah.')
add('Referensi arsitektur: <link href="https://arxiv.org/abs/1512.03385">ResNet - He dkk.</link> dan <link href="https://arxiv.org/abs/1905.02244">MobileNetV3 - Howard dkk.</link>. Detail implementasi diperiksa dari models.py dan torchvision0.25.0.')

head('4. Perbandingan hasil tiap model')
tab([['Buah','Model','Akurasi','Presisi*','Recall*','F1*','Selisih F1**']]+[[FR[f],NAMES[r['config']['id']],pct(r['test']['accuracy']),f"{r['test']['precision']:.3f}",f"{r['test']['recall']:.3f}",f"{r['test']['f1']:.4f}",f"{100*(r['test']['f1']-s['experiments'][0]['test']['f1']):+.2f}"] for f,s in S.items() for r in s['experiments']],[56,104,67,65,65,68,78])
add('*Presisi, recall dan F1 menggunakan rata-rata macro: setiap kategori diberi bobot sama. **Selisih F1 dalam poin persentase dibanding CNN baseline pada buah yang sama. Akurasi adalah bagian prediksi yang benar. Presisi melihat ketepatan prediksi kategori; recall melihat berapa banyak anggota kategori yang berhasil ditemukan.')
add('CNN eksperimen selalu lebih rendah dari baseline pada budget12 epoch. ResNet dan MobileNet memperbaiki hasil untuk pisang, apel, jeruk dan nanas. Pada stroberi, keduanya tidak meningkatkan F1 test dibanding baseline. Perubahan model transfer juga mengubah bobot awal dan bagian yang dilatih, sehingga hasilnya tidak dapat dijelaskan oleh arsitektur saja.')
head('4. Model mana yang lebih baik?')
tab([['Buah','Pilihan berdasarkan validasi','F1 validasi','F1 test','F1 test tertinggi']]+[[FR[f],NAMES[s['selected_model']],f"{selected(f)['validation_f1']:.4f}",f"{selected(f)['test']['f1']:.4f}",NAMES[max(s['experiments'],key=lambda r:r['test']['f1'])['config']['id']]] for f,s in S.items()],[60,150,85,85,123])
add('<b>Pisang dan apel:</b> MobileNet memiliki F1 tertinggi pada validasi dan test. Fitur pretrained membantu model mengenali pola warna/tekstur dengan lebih baik pada data ini. Ini penjelasan yang masuk akal, bukan hasil pengujian terpisah untuk membuktikan penyebabnya.')
add('<b>Jeruk:</b> MobileNet dipilih karena F1 validasi lebih tinggi. ResNet ternyata lebih tinggi pada test. Test hanya26 gambar; satu kesalahan mengubah akurasi sekitar3,85 poin persentase. Selisih kecil perlu diuji pada lebih banyak gambar.')
add('<b>Nanas:</b> ResNet dipilih berdasarkan validasi, tetapi MobileNet lebih tinggi pada test. Pilihan tidak diganti setelah melihat test agar test tetap menjadi evaluasi akhir, bukan bahan memilih model.')
add('<b>Stroberi:</b> CNN baseline memberi F1 test lebih tinggi daripada model transfer, tetapi semua hasil masih rendah. MobileNet menunjukkan training hampir100% dan validasi63,53%, sehingga terlihat overfitting. Foto berisi banyak buah dan label tingkat gambar dapat menimbulkan ambiguitas; label dan crop perlu diperiksa untuk membuktikannya.')
add('Jadi, "lebih baik" harus menyebut ukurannya: terbaik pada validasi, terbaik pada test, atau paling ringan. Perbandingan arsitektur memakai dataset test yang sama dalam satu buah. Skor antarbuah memakai dataset dan kategori berbeda, sehingga tidak langsung menunjukkan buah mana yang paling mudah.')

head('5. Proses training dan hasil pengujian')
add('Bagian ini menampilkan setiap model untuk setiap buah. Grafik dipisahkan agar loss, akurasi dan F1 terbaca. Tabel menampilkan nilai pada seluruh12 epoch. Setelah empat model training tiap buah, tersedia hasil pengujian serta contoh gambar benar/salah.')
tab([['Istilah','Cara membacanya'],['Training loss','Kesalahan model pada data belajar. Semakin kecil umumnya semakin baik.'],['Validation loss','Kesalahan pada data validasi. Dipantau bersama training loss.'],['Training / validation accuracy','Bagian prediksi benar pada masing-masing data.'],['Training / validation F1','Keseimbangan presisi dan recall; kategori dirata-ratakan secara macro.'],['Overfitting','Hasil training membaik, tetapi validasi tertinggal atau memburuk.'],['Underfitting','Model belum belajar cukup baik; training dan validasi sama-sama rendah.'],['Training tidak stabil','Loss berubah besar antar-epoch; perubahan cepat ke arah baik juga dapat memicu indikator.'],['Mulai mendatar','Loss validasi akhir berubah sedikit. Tidak otomatis berarti model sudah optimal.']],[155,348])
add('Aturan indikator aplikasi: overfitting jika gap akurasi akhir&gt;0,10 atau validation loss naik&gt;15% dari minimum ketika training loss turun. Underfitting jika kedua akurasi akhir&lt;0,70. Tidak stabil jika perubahan absolut validation loss&gt;0,30. Mendatar jika rentang tiga validation loss terakhir&lt;0,025. Indikator gradien kecil/sangat besar memakai mean&lt;1e-6 /max&gt;100. Ini alat bantu membaca hasil, bukan diagnosis pasti.')
add('Train memakai gambar augmentasi; validation tidak diberi augmentasi tambahan. MobileNet juga memakai dropout saat training. Karena itu perbedaan training-validasi perlu dilihat bersama seluruh kurva. Model dipilih menggunakan F1 validasi terbaik; evaluasi test menggunakan checkpoint tersebut.')

for f,s in S.items():
 for r in s['experiments']:
  id=r['config']['id'];hist=r['history'];title=FR[f]+' - '+NAMES[id]
  head('5. Training / '+title)
  add(f"Learning rate: {r['config']['lr']}. Batch:32. Epoch:12. Checkpoint terbaik: epoch{r['best_epoch']}. F1 validasi terbaik: <b>{r['validation_f1']:.4f}</b>.")
  figs=[]
  for metric,label in [('loss','Loss'),('accuracy','Akurasi'),('f1','F1 macro')]:
   fig,ax=plt.subplots(figsize=(5,2.1));ax.plot([x['epoch'] for x in hist],[x['train_'+metric] for x in hist],label='Training');ax.plot([x['epoch'] for x in hist],[x['val_'+metric] for x in hist],label='Validasi');ax.set(title=label,xlabel='Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();file=G/f'{f}_{id}_{metric}.png';fig.savefig(file,dpi=150);plt.close(fig);figs.append(img(file,245,115))
  diag='<br/>'.join(f"<b>{DIAG[x['topic']]}:</b> {'terlihat' if x['flag'] else 'tidak terdeteksi'}" for x in r['diagnostics'])
  story.append(Table([[figs[0],figs[1]],[figs[2],p(diag)]],colWidths=[251,252],style=[('VALIGN',(0,0),(-1,-1),'TOP')]))
  tab([['Epoch','Train loss','Val loss','Train acc','Val acc','Train F1','Val F1']]+[[x['epoch'],*[f"{x[k]:.4f}" for k in ['train_loss','val_loss','train_accuracy','val_accuracy','train_f1','val_f1']]] for x in hist],[43,78,78,76,76,76,76])
  last=hist[-1];flags=[DIAG[x['topic']] for x in r['diagnostics'] if x['flag']]
  add(f"<b>Kesimpulan:</b> akurasi training akhir {pct(last['train_accuracy'])}; validasi {pct(last['val_accuracy'])}. Indikator yang muncul: {', '.join(flags) if flags else 'tidak ada menurut ambang aplikasi'}. Test F1 dari checkpoint terbaik: {r['test']['f1']:.4f}.")
 head('5. Pengujian / '+FR[f])
 add('Confusion matrix: baris menunjukkan label sebenarnya, kolom menunjukkan prediksi. Angka pada diagonal adalah prediksi benar. Semua model diuji pada gambar yang sama untuk buah ini.')
 cms=[img(E/f/r['config']['id']/'confusion_matrix.png',245,140) for r in s['experiments']]
 story.append(Table([[p('<b>'+NAMES[s['experiments'][0]['config']['id']]+'</b>'),p('<b>'+NAMES[s['experiments'][1]['config']['id']]+'</b>')],[cms[0],cms[1]],[p('<b>ResNet18</b>'),p('<b>MobileNetV3 Small</b>')],[cms[2],cms[3]]],colWidths=[251,252]))
 tab([['Model','Kategori','Presisi','Recall','F1','Jumlah']]+[[NAMES[r['config']['id']],c,*[f"{r['test']['per_class'][c][k]:.3f}" for k in ['precision','recall','f1-score']],int(r['test']['per_class'][c]['support'])] for r in s['experiments'] for c in s['classes']],[115,100,72,72,72,72])
 head('5. Contoh prediksi dan kesalahan / '+FR[f])
 add('Satu contoh salah dan satu contoh benar untuk setiap model. Contoh salah diambil dari kesalahan dengan skor tertinggi. Skor adalah keluaran softmax; skor tinggi tetap dapat salah. Contoh gambar ini membantu inspeksi, bukan membuktikan penyebab kesalahannya.')
 rows=[['Model','Gambar','Label benar','Prediksi','Skor','Analisis']]
 for r in s['experiments']:
  d=pd.read_csv(E/f/r['config']['id']/'test_predictions.csv')
  for correct in [False,True]:
   subset=d[d.correct==correct].sort_values('confidence',ascending=False)
   if subset.empty:continue
   row=subset.iloc[0]
   rows.append([NAMES[r['config']['id']],img(ROOT/row.path,70,64),row.true_label,row.prediction,f'{row.confidence:.3f}','Benar' if correct else 'Salah; kemiripan visual atau label ambigu perlu diperiksa.'])
 cells=[]
 for row in rows:cells.append([x if isinstance(x,Image) else Paragraph(escape(str(x)),ss['Cell']) for x in row])
 t=Table(cells,colWidths=[90,80,65,65,50,153],repeatRows=1);t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfebdc')),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#d8dfd5')),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t)

head('6. Kelayakan untuk penggunaan nyata')
add('<b>Performance - Seberapa baik model bekerja?</b> Hasil model pilihan validasi:')
tab([['Buah','Model','Akurasi test','F1 test']]+[[FR[f],NAMES[s['selected_model']],pct(selected(f)['test']['accuracy']),f"{selected(f)['test']['f1']:.4f}"] for f,s in S.items()],[75,175,125,128])
add('Pisang, apel, jeruk dan nanas memberikan hasil internal yang lebih baik daripada stroberi. Apel dan jeruk memiliki test kecil. Hasil ini belum membuktikan performa pada foto lapangan baru.')
add('<b>Robustness - Bagaimana saat kondisi data berubah?</b> Berikut F1 test model pilihan saat gambar dibuat lebih gelap/terang dan blur. Pengujian ini memakai gambar test yang ada, bukan dataset lapangan baru.')
tab([['Buah','Normal','Lebih gelap','Lebih terang','Blur']]+[[FR[v['fruit']],*[f"{z['f1']:.4f}" for z in v['conditions']]] for v in M['robustness']],[75,107,107,107,107])
add('Gelap=brightness0,6; terang=1,4; blur=Gaussian radius2 setelah resize. Perubahan skor menunjukkan model sensitif terhadap kondisi gambar. Kamera, latar, varietas baru dan occlusion belum diuji secara khusus.')
add('<b>Computational cost - Berapa biaya komputasinya?</b> Waktu training dari catatan eksperimen; waktu inferensi dan memori dari pengukuran tambahan pada CPU yang sama untuk seluruh model.')
tab([['Buah/model pilihan','Training (s)','Inferensi (ms)*','Memori (MiB)**','File model (MiB)']]+[[FR[f]+'/'+NAMES[s['selected_model']],f"{selected(f)['training_seconds']:.2f}",f"{b['p50_ms']:.2f}",f"{b['rss_after_mib']:.1f}",f"{b['checkpoint_mib']:.2f}"] for f,s in S.items() for b in [next(v for v in M['benchmarks'] if v['fruit']==f and v['model']==s['selected_model'])]],[155,75,88,93,92])
add('*Median30 forward batch1 setelah5 warmup,4 thread CPU; tidak termasuk baca gambar dan jaringan. **Memori seluruh proses Python, bukan bobot saja. Peak memori training tidak tercatat. Hardware pengukuran: CPU16 core fisik/32 logical, RAM15,21GiB. GPU tidak wajib; minimum RAM belum diuji.')
head('6. Kelayakan untuk penggunaan nyata (lanjutan)')
add('<b>Scalability - Dapatkah dipakai pada skala lebih besar?</b><br/>Saat ini aplikasi berjalan di Flask lokal. Satu lock membuat inferensi bergantian saat banyak permintaan datang. Model yang pernah dimuat disimpan di memori tanpa batas penghapusan cache. Belum ada pengukuran jumlah pengguna bersamaan atau kapasitas request. Skala lebih besar memerlukan server produksi, pengaturan cache dan pengujian beban.')
add('<b>Limitations - Apa kelemahannya?</b><br/>Pengguna harus memilih jenis buah sendiri. Aplikasi belum memastikan gambar benar-benar buah yang dipilih. Foto diperkecil menjadi128 x128, sehingga detail dapat hilang. Label stroberi per gambar bisa ambigu jika ada beberapa buah berbeda kematangan. Nanas dilatih pada crop, sedangkan upload tidak dicrop otomatis. Kemiripan frame dan buah yang sama dapat tersisa antarsplit. Semua eksperimen memakai satu seed. Penampilan luar tidak mengukur kemanisan internal atau keamanan konsumsi.')
add('<b>Practical suitability - Apakah terbaik secara akademik juga terbaik untuk deployment?</b><br/>Tidak selalu. ResNet lebih besar; MobileNet lebih kecil dan cocok menjadi pilihan ketika memori terbatas. Namun model yang ringan belum tentu akurat untuk semua buah. Model terbaik perlu dipilih dengan hasil validasi, ketepatan kategori penting, kecepatan dan memori pada perangkat tujuan. Test tertinggi tidak dipakai untuk memilih ulang model setelah evaluasi.')
add('<b>Penilaian akhir:</b> proyek dapat dipakai untuk demonstrasi dan percobaan terkontrol. Penggunaan operasional yang luas belum didukung evaluasi lapangan independen. Stroberi membutuhkan perbaikan data dan validasi sebelum dipakai untuk keputusan nyata.')

dest=ROOT.parent.parent/'DeepLearning/Laporan_Revisi_6_Aspek_Ripeness_Lab.pdf'
def footer(c,d):
 c.setFont('Cal',8);c.setFillColor(colors.HexColor('#667667'));c.drawString(46,25,'RIPENESS LAB | Enam aspek proyek | Data:23 November2026');c.drawRightString(549,25,str(d.page))
SimpleDocTemplate(str(dest),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=40,bottomMargin=42,title='Ripeness Lab - Revisi Enam Aspek').build(story,onFirstPage=footer,onLaterPages=footer)
doc=fitz.open(dest);print('PDF',dest,'pages',len(doc))
toc=[]
for i,page in enumerate(doc):
 for line in page.get_text().splitlines():
  if line.startswith(tuple(str(n)+'. ' for n in range(1,7))):toc.append([1,line,i+1]);break
doc.set_toc(toc);temp=dest.with_suffix('.tmp.pdf');doc.save(temp,garbage=4,deflate=True);doc.close();temp.replace(dest)
