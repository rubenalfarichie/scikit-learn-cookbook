# scikit-learn-cookbook

# scikit-learn Cookbook (Third Edition) - Code Reproduction & Theoretical Deep-Dive

Repositori ini berisi reproduksi kode dan penjelasan teoretis mendalam (*theoretical deep-dive*) dari buku **"scikit-learn Cookbook, Third Edition: Over 80 recipes for machine learning in Python with scikit-learn"** karya **John Sukup** (Packt Publishing, 2025). Disusun untuk memenuhi Tugas Individu Mata Kuliah Machine Learning.

---

## Daftar Isi & Ringkasan Bab (Chapter Summary)

### Chapter 1: Common Conventions and API Elements of scikit-learn
Bab ini membahas filosofi desain dasar dari pustaka `scikit-learn` yang berpusat pada konsistensi, kesederhanaan, modularitas, dan *reusability*. Konsep inti yang dibahas meliputi:
* **Estimators**: Objek dasar yang mempelajari parameter dari data menggunakan metode `fit()` dan menghasilkan prediksi dengan `predict()` (serta jalan pintas `fit_predict()` untuk *unsupervised learning*).
* **Transformers**: Komponen pemrosesan data yang mempelajari statistik fitur melalui `fit()` dan mengubah skala/format data melalui `transform()` atau `fit_transform()`.
* **Pipelines & MLOps**: Penggabungan berbagai tahapan transformasi dan pemodelan ke dalam satu objek `Pipeline()` untuk mencegah kebocoran data (*data leakage*) dan mendukung otomatisasi alur kerja *Machine Learning Operations* (MLOps).
* **Atribut, Hyperparameter, & Metadata Routing**: Penggunaan atribut terlatih (seperti `coef_` dan `intercept_`), evaluasi cepat via `score()`, pengaturan parameter melalui `set_params()`/`get_params()`, serta pengelolaan aliran metadata antara *routers* dan *consumers*.

### Chapter 2: Pre-Model Workflow and Data Preprocessing
Membahas pentingnya kualitas data (*garbage in, garbage out*) dan teknik penyiapan data mentah sebelum dilatih ke dalam model:
* **Penanganan Data Hilang (*Missing Data*)**: Strategi imputasi menggunakan nilai statistik dasar (`SimpleImputer`), kemiripan tetangga terdekat (`KNNImputer`), dan pemodelan regresi multivariat berulang (`IterativeImputer`).
* **Teknik Penyekalaan (*Feature Scaling*)**: Perbedaan matematis dan implementasi antara standardisasi Z-score (`StandardScaler`), normalisasi rentang nilai ke $[0, 1]$ (`MinMaxScaler`), serta normalisasi norma unit L1/L2 pada sampel (`Normalizer`).
* **Encoding Variabel Kategorikal**: Transformasi fitur nominal dan ordinal menjadi representasi numerik menggunakan `OneHotEncoder`, `LabelEncoder`, dan penggabungan pemrosesan kolom campuran (numerik & kategorikal) secara paralel dengan `ColumnTransformer`.
* **Feature Engineering**: Ekstraksi fitur baru dengan `PolynomialFeatures` dan diskritisasi rentang (`KBinsDiscretizer`), serta seleksi fitur otomatis menggunakan *Recursive Feature Elimination* (`RFE`) dan `SelectFromModel`.

### Chapter 3: Dimensionality Reduction Techniques
Membahas teknik mereduksi jumlah fitur pada dataset berdimensi tinggi untuk menekan biaya komputasi, menghindari *overfitting* (*curse of dimensionality*), dan memfasilitasi visualisasi data:
* **Principal Component Analysis (PCA)**: Teknik *unsupervised* berbasis nilai eigen dan matriks kovarians yang memproyeksikan data ke sumbu ortogonal (*principal components*) dengan varians terbesar.
* **Linear Discriminant Analysis (LDA)**: Teknik *supervised* yang mencari kombinasi linear fitur untuk memaksimalkan jarak antar-kelas (*between-class variance*) sekaligus meminimalkan sebaran di dalam kelas (*within-class variance*).
* **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: Teknik reduksi dimensi non-linear berbasis probabilitas kondisional yang mempertahankan struktur lokal data untuk memvisualisasikan klaster kompleks dalam ruang 2D atau 3D.

### Chapter 4: Building Models with Distance Metrics and Nearest Neighbors
Mengkaji algoritma berbasis kemiripan jarak spasial dan implementasi *k-Nearest Neighbors* (KNN):
* **Metrik Jarak (*Distance Metrics*)**: Formulasi matematis dan karakteristik penggunaan jarak *Euclidean* (garis lurus / norma L2), jarak *Manhattan* (*city block* / norma L1 yang lebih tahan terhadap *outlier*), dan generalisasinya dalam jarak *Minkowski*.
* **Algoritma KNN**: Mekanisme pemungutan suara mayoritas (*majority voting*) untuk klasifikasi dan rata-rata tetangga terdekat untuk regresi, serta pengaruh pemilihan nilai $k$ terhadap batas keputusan (*decision boundary*).
* **Optimasi & Evaluasi Kinerja**: Pencarian kombinasi hyperparameter terbaik (`n_neighbors`, `weights`, `metric`) menggunakan `GridSearchCV`, serta evaluasi komprehensif melalui kurva pembelajaran (*learning curves*), *confusion matrix*, *precision*, *recall*, dan *F1-score*.

### Chapter 5: Linear Models and Regularization
Mengupas pemodelan hubungan linear antara fitur independen dan target kontinu, serta pengendalian kompleksitas model melalui penalti regularisasi:
* **Ordinary Least Squares (OLS)**: Regresi linear klasik yang meminimalkan jumlah kuadrat residu (*Mean Squared Error* / MSE) dan dievaluasi dengan koefisien determinasi ($R^2$).
* **Ridge Regression (Regularisasi L2)**: Penambahan penalti kuadrat magnitudo koefisien ($\alpha \sum \beta_j^2$) untuk menyusutkan bobot fitur yang berkorelasi tinggi (*multicollinearity*) mendekati nol.
* **Lasso Regression (Regularisasi L1)**: Penambahan penalti nilai absolut koefisien ($\alpha \sum |\beta_j|$) yang mampu menyusutkan koefisien fitur tidak relevan menjadi tepat nol (*sparsity* / seleksi fitur otomatis).
* **ElasticNet & Regresi Non-Linear**: Kombinasi penalti L1 dan L2 yang dikontrol melalui `l1_ratio`, analisis *coefficient paths*, serta pemodelan hubungan kurvilinear menggunakan regresi polinomial dan *spline interpolation* (`SplineTransformer`).

### Chapter 6: Advanced Logistic Regression and Extensions
Memperluas konsep regresi untuk menyelesaikan masalah klasifikasi biner, multikelas, dan multilabel:
* **Regresi Logistik Biner**: Transformasi kombinasi linear fitur menjadi probabilitas $[0, 1]$ menggunakan fungsi sigmoid/logistik dan konsep *log-odds* dengan estimasi *Maximum Likelihood* (MLE).
* **Klasifikasi Multikelas (*Multiclass*)**: Perbandingan strategi *One-vs-Rest* (OvR) yang melatih pengklasifikasi biner terpisah untuk tiap kelas, dengan pendekatan *Multinomial* (*SoftMax*) yang mengoptimalkan probabilitas seluruh kelas secara simultan.
* **Regularisasi pada Regresi Logistik**: Pengendalian parameter $C$ (invers kekuatan regularisasi) dengan penalti L1 dan L2 untuk mengatur *trade-off* antara bias dan varians.
* **Klasifikasi Multilabel & Evaluasi ROC-AUC**: Penanganan sampel yang memiliki lebih dari satu label target sekaligus, analisis *label co-occurrence matrix*, serta evaluasi ambang batas probabilitas dengan kurva *Receiver Operating Characteristic* (ROC) dan *Area Under the Curve* (AUC).

### Chapter 7: Support Vector Machines and Kernel Methods
Membahas pembangunan batas keputusan optimal pada data berdimensi tinggi dan data yang tidak terpisahkan secara linear:
* **Konsep Hyperplane & Margin**: Cara kerja *Support Vector Classifier* (`SVC`) dan *Regressor* (`SVR`) dalam mencari *hyperplane* yang memaksimalkan margin terhadap titik data terdekat (*support vectors*), serta konsep *soft margin* (variabel *slack*).
* **Metode Kernel (*The Kernel Trick*)**: Pemetaan implisit data ke dimensi yang lebih tinggi menggunakan fungsi kernel *Linear*, *Polynomial*, dan *Radial Basis Function* (RBF / Gaussian) untuk membentuk batas keputusan non-linear.
* **Penyetelan Parameter & Ruang Dimensi Tinggi**: Optimasi parameter regularisasi $C$, derajat polinomial (`degree`), dan koefisien kernel (`gamma`) melalui validasi silang.

### Chapter 8: Tree-Based Algorithms and Ensemble Methods
Membahas algoritma berbasis pohon keputusan dan penggabungan beberapa model (*ensemble*) untuk meningkatkan akurasi serta stabilitas prediksi:
* **Decision Trees**: Pembentukan aturan percabangan berdasarkan metrik kemurnian simpul (*Gini impurity* atau *entropy* / *information gain*) dan pengendalian kedalaman pohon.
* **Random Forests & Bagging**: Teknik *Bootstrap Aggregating* yang melatih banyak pohon keputusan secara paralel pada subset data dan fitur acak untuk menurunkan varians (*overfitting*).
* **Gradient Boosting Machines (GBM)**: Pelatihan pohon keputusan secara sekuensial di mana setiap pohon baru berusaha mengoreksi kesalahan residu dari pohon sebelumnya.

### Chapter 9: Text Processing and Multiclass Classification
Membahas alur pemrosesan data teks tidak terstruktur (*Natural Language Processing* dasar) untuk tugas klasifikasi dokumen:
* **Text Preprocessing & Vectorization**: Pembersihan teks dan konversi dokumen menjadi matriks numerik menggunakan *Bag-of-Words* (`CountVectorizer`) serta pembobotan kepentingan kata dengan *Term Frequency-Inverse Document Frequency* (`TfidfVectorizer`).
* **Ekstraksi Fitur N-Grams**: Penangkapan konteks urutan kata lokal menggunakan rentang *unigram*, *bigram*, dan *n-grams* dalam model klasifikasi multikelas.

### Chapter 10: Clustering Techniques
Mengkaji metode *unsupervised learning* untuk menemukan pengelompokan alami pada data tanpa label:
* **K-Means Clustering**: Partisi iteratif data ke dalam $k$ klaster berdasarkan jarak terdekat terhadap titik pusat (*centroid*).
* **Hierarchical Clustering & DBSCAN**: Pembentukan hierarki klaster berbasis pohon (*dendrogram*) serta pengelompokan berbasis kepadatan titik data (`DBSCAN`) yang mampu mengenali bentuk klaster arbitrer dan derau (*noise*).
* **Gaussian Mixture Models (GMM) & Evaluasi Klaster**: Pendekatan klasterisasi probabilistik dan evaluasi kualitas klaster menggunakan metrik seperti *Silhouette Score*.

### Chapter 11: Novelty and Outlier Detection
Membahas teknik identifikasi anomali, baik berupa *outlier* (data ekstrem pada data latih) maupun *novelty* (pola baru yang belum pernah terlihat pada data uji):
* **Isolation Forest**: Algoritma berbasis pohon acak yang mengisolasi pengamatan anomali berdasarkan panjang jalur pemisahan yang lebih pendek.
* **One-Class SVM & Local Outlier Factor (LOF)**: Pemodelan batas wilayah data normal menggunakan *One-Class SVM* serta deteksi anomali berdasarkan perbandingan kepadatan lokal suatu titik terhadap tetangganya (`LocalOutlierFactor`).

### Chapter 12: Cross-Validation and Model Evaluation Techniques
Mendalami strategi validasi untuk memastikan kemampuan generalisasi model pada data baru:
* **Strategi Cross-Validation Lanjutan**: Implementasi *K-Fold*, *Stratified K-Fold* (untuk menjaga proporsi kelas pada data timpang), *Leave-One-Out* (LOO), dan validasi data deret waktu.
* **Kurva Diagnostik Generalisasi**: Analisis *learning curves* dan *validation curves* untuk mendiagnosis masalah *underfitting* (bias tinggi) dan *overfitting* (varians tinggi).

### Chapter 13: Deploying scikit-learn Models in Production
Membahas praktik terbaik dalam membawa model yang telah dilatih ke lingkungan produksi nyata:
* **Serialisasi & Persistensi Model**: Penyimpanan dan pemuatan kembali seluruh objek *pipeline* menggunakan pustaka `joblib` dan `pickle`.
* **Skalabilitas, Pemantauan, & Siklus Hidup Model**: Pembuatan *deployment pipelines*, pemantauan penurunan akurasi (*model drift*) seiring waktu, serta otomatisasi pemeriksaan ambang batas validasi sebelum rilis produksi.

---

## Cara Menjalankan Kode (Instalasi)
1. Clone repositori ini:
   ```bash
   git clone [https://github.com/](https://github.com/)<username-anda>/scikit-learn-cookbook.git
   cd scikit-learn-cookbook
