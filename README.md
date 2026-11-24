# IF3170_G19_kagglemethis

Tugas Besar 2 IF3170 Inteligensi Artifisial implementasi model from scratch pada KNN, Decision Tree Learning, Logistic Regression, SVM, dan Naive Bayes. Setiap algoritma diimplementasikan from scratch dan dibandingkan dengan scikit-learn.

## Setup

Jalankan dari direktori utama repository:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter ipython
python -m pip install -e ./src
```

Untuk Windows, aktivasi virtual environment menggunakan `venv\Scripts\activate`.

## Menjalankan

1. Pastikan `train.csv`, `test.csv`, dan `sample_submission.csv` tersedia di folder `data/`.
2. Jalankan `jupyter notebook` dari direktori utama repository.
3. Buka `main.ipynb`, pilih kernel dari virtual environment, lalu jalankan sel secara berurutan.

Notebook mencakup preprocessing, training, evaluasi, tuning, dan pembuatan file submission. Hasil disimpan di folder `output/`. Tuning dapat membutuhkan waktu cukup lama.

## Struktur

- `src/kagglemethis/`: implementasi algoritma dan dashboard bonus Logistic Regression.
- `main.ipynb`: alur eksperimen utama.
- `data/`: dataset training dan test.
- `output/`: hasil eksperimen dan submission.
- `Laporan Tubes 2 AI.md`: laporan tugas besar.

## Pembagian Tugas

| Nama | Tugas |
| --- | --- |
| Raysha Erviandika Putra | DTL |
| Mahmudia Kimdaro Amin | KNN |
| Billie Bhaskara Wibawa | Naive Bayes |
| Anindya Naufal Pinasthika | Logistic Regression |
| M. Fatih Irkham Mauludi | SVM |
