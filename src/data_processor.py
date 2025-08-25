import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Pemroses data lengkap untuk dataset penyakit hati dengan semua metode - VERSI YANG DIPERBAIKI"""
    
    def __init__(self):
        """Inisialisasi objek DataProcessor dengan semua komponen yang diperlukan"""
        self.scaler = StandardScaler()  # Untuk normalisasi fitur
        self.label_encoder = LabelEncoder()  # Untuk encoding label kategorikal
        self.feature_names = None  # Menyimpan nama fitur setelah preprocessing
        # Kolom yang diharapkan ada di dataset
        self.expected_columns = [
            'Age', 'Gender', 'Total Bilirubin', 'Direct Bilirubin', 
            'Alkphos', 'Sgpt', 'Sgot', 'Total Protiens', 'ALB', 'A/G Ratio', 'Result'
        ]
        
    def load_data(self, filepath):
        """Memuat dataset dari file CSV dengan berbagai encoding yang fleksibel - DIPERBAIKI"""
        # Mencoba berbagai encoding karena dataset bisa memiliki encoding berbeda
        encodings_to_try = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]
        
        for enc in encodings_to_try:
            try:
                data = pd.read_csv(filepath, encoding=enc)
                print(f"✅ Data berhasil dimuat dengan encoding '{enc}'! Ukuran: {data.shape}")
                print(f"📋 Kolom yang ditemukan: {list(data.columns)}")
                return data
            except Exception as e:
                print(f"⚠️ Gagal dengan encoding {enc}: {e}")
        
        print("❌ Semua percobaan encoding gagal.")
        return None

    def explore_data(self, data):
        """Eksplorasi data dasar dengan analisis target yang tepat"""
        print("\n" + "="*50)
        print("📊 RINGKASAN EKSPLORASI DATA")
        print("="*50)
        
        # Informasi dasar dataset
        print(f"Ukuran Dataset: {data.shape}")
        print(f"Jumlah Fitur: {data.shape[1]}")
        print(f"Jumlah Sampel: {data.shape[0]}")
        
        # Mengecek nilai yang hilang (missing values)
        print("\n📈 Nilai yang Hilang:")
        missing = data.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                print(f"  {col}: {count} ({count/len(data)*100:.1f}%)")
        
        # Analisis kolom target (Result)
        if 'Result' in data.columns:
            print(f"\n🎯 Distribusi Target (Result):")
            target_counts = data['Result'].value_counts().sort_index()
            for value, count in target_counts.items():
                label = "Penyakit Hati" if value == 1 else "Tidak Ada Penyakit" 
                print(f"  {label} ({value}): {count} ({count/len(data)*100:.1f}%)")
        
        # Analisis distribusi Gender
        if 'Gender' in data.columns:
            print(f"\n👥 Distribusi Gender:")
            gender_counts = data['Gender'].value_counts()
            for gender, count in gender_counts.items():
                print(f"  {gender}: {count} ({count/len(data)*100:.1f}%)")
        
        # Menampilkan tipe data setiap kolom
        print("\n📋 Tipe Data:")
        for col, dtype in data.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # Menampilkan contoh data
        print(f"\n📄 Contoh Data:")
        print(data.head(3).to_string())
            
        return {
            'shape': data.shape,
            'missing_values': missing.sum(),
            'target_distribution': data['Result'].value_counts() if 'Result' in data.columns else None
        }
    
    def clean_data(self, data):
        """Membersihkan dan memproses data awal - DIPERBAIKI UNTUK MENANGANI KOLOM KHUSUS"""
        print("\n🧹 MEMBERSIHKAN DATA...")
        
        cleaned_data = data.copy()  # Buat salinan untuk menghindari perubahan data asli
        
        # CRITICAL FIX: Clean column names (menangani karakter \xa0)
        print("  🔧 Membersihkan nama kolom...")
        original_columns = list(cleaned_data.columns)
        
        # Mapping kolom yang bermasalah
        column_mapping = {
            'Age of the patient': 'Age',
            'Gender of the patient': 'Gender', 
            'Total Bilirubin': 'Total Bilirubin',
            'Direct Bilirubin': 'Direct Bilirubin',
            '\xa0Alkphos Alkaline Phosphotase': 'Alkphos',
            '\xa0Sgpt Alamine Aminotransferase': 'Sgpt',
            'Sgot Aspartate Aminotransferase': 'Sgot',
            'Total Protiens': 'Total Protiens',
            '\xa0ALB Albumin': 'ALB',
            'A/G Ratio Albumin and Globulin Ratio': 'A/G Ratio',
            'Result': 'Result'
        }
        
        # Rename kolom yang bermasalah
        cleaned_data = cleaned_data.rename(columns=column_mapping)
        print(f"  ✅ Kolom dibersihkan: {list(cleaned_data.columns)}")
        
        # Show missing values before cleaning
        missing_before = cleaned_data.isnull().sum()
        print("  📊 Nilai kosong sebelum dibersihkan:")
        for col, count in missing_before.items():
            if count > 0:
                print(f"     {col}: {count}")
        
        # Menangani nilai yang hilang untuk kolom numerik
        numeric_columns = ['Age', 'Total Bilirubin', 'Direct Bilirubin', 'Alkphos', 
                          'Sgpt', 'Sgot', 'Total Protiens', 'ALB', 'A/G Ratio']
        
        for col in numeric_columns:
            if col in cleaned_data.columns and cleaned_data[col].isnull().sum() > 0:
                median_val = cleaned_data[col].median()
                cleaned_data[col].fillna(median_val, inplace=True)
                print(f"  ✅ Mengisi nilai kosong {col} dengan median: {median_val:.2f}")
        
        # Menangani kolom Gender secara khusus
        if 'Gender' in cleaned_data.columns:
            if cleaned_data['Gender'].isnull().sum() > 0:
                mode_val = cleaned_data['Gender'].mode()[0] if not cleaned_data['Gender'].mode().empty else 'Male'
                cleaned_data['Gender'].fillna(mode_val, inplace=True)
                print(f"  ✅ Mengisi nilai kosong Gender dengan modus: {mode_val}")
        
        # CRITICAL FIX: Handle Result column properly
        if 'Result' in cleaned_data.columns:
            # Remove rows where Result is missing
            if cleaned_data['Result'].isnull().sum() > 0:
                before_len = len(cleaned_data)
                cleaned_data = cleaned_data.dropna(subset=['Result'])
                removed = before_len - len(cleaned_data)
                print(f"  ✅ Menghapus {removed} baris dengan Result kosong")
        
        # Menghapus data duplikat
        initial_shape = cleaned_data.shape[0]
        cleaned_data = cleaned_data.drop_duplicates()
        removed_duplicates = initial_shape - cleaned_data.shape[0]
        if removed_duplicates > 0:
            print(f"  ✅ Menghapus {removed_duplicates} baris duplikat")
        
        print(f"  📊 Ukuran final: {cleaned_data.shape}")
        return cleaned_data
    
    def prepare_features(self, data, target_column='Result'):
        """Mempersiapkan fitur dan variabel target - VERSI YANG DIPERBAIKI"""
        print("\n⚙️ MEMPERSIAPKAN FITUR...")
        
        # Pastikan kolom target ada
        if target_column not in data.columns:
            raise ValueError(f"Kolom target '{target_column}' tidak ditemukan!")
        
        print(f"  🎯 Menggunakan kolom target: {target_column}")
        
        # Memisahkan fitur (X) dan target (y)
        X = data.drop(columns=[target_column])  # Fitur = semua kolom kecuali target
        y = data[target_column].copy()  # Target = kolom Result
        
        # CRITICAL FIX: Proper target encoding
        print(f"  📊 Nilai target asli: {sorted(y.unique())}")
        
        # Convert Result: 2->0 (Healthy), 1->1 (Disease) 
        print(f"  🔍 Nilai target unik yang ditemukan: {sorted(y.unique())}")
        
        if set(sorted(y.unique())) == {1, 2}:
            # Convert to binary: 1=Disease, 0=Healthy
            print("  📝 Mengkonversi: 2=Sehat -> 0=Sehat, 1=Penyakit -> 1=Penyakit")
            y = (y == 1).astype(int)
        elif set(sorted(y.unique())).issubset({0, 1}):
            # Already in correct format
            print("  ✅ Target sudah dalam format yang benar: 1=Penyakit, 0=Sehat")
            y = y.astype(int)
        else:
            # For other values, assume max = disease
            print(f"  ⚠️ Nilai target tidak terduga: {sorted(y.unique())}")
            max_val = max(y.unique())
            y = (y == max_val).astype(int)
            print(f"  📝 Menganggap nilai {max_val} = Penyakit")
        
        print(f"  ✅ Nilai target final: {sorted(y.unique())}")
        
        # Show final distribution
        target_counts = y.value_counts()
        for value, count in target_counts.items():
            label = "Penyakit" if value == 1 else "Sehat"
            percentage = (count/len(y)) * 100
            print(f"     {label}: {count} ({percentage:.1f}%)")
        
        # CRITICAL FIX: Proper Gender encoding
        if 'Gender' in X.columns:
            print("  🔄 Encoding kolom Gender...")
            # Create Gender_Male (1=Male, 0=Female)
            X['Gender_Male'] = (X['Gender'] == 'Male').astype(int)
            X = X.drop('Gender', axis=1)  # Remove original Gender column
            print(f"  ✅ Gender berhasil di-encode sebagai Gender_Male")
        
        # Menyimpan nama fitur untuk digunakan nanti
        self.feature_names = X.columns.tolist()
        
        print(f"  📋 Fitur final ({len(X.columns)}): {self.feature_names}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Membagi data menjadi set latih dan set uji - DIPERBAIKI"""
        # Cek distribusi kelas sebelum split
        disease_count = sum(y)
        healthy_count = len(y) - disease_count
        disease_ratio = disease_count / len(y)
        
        print(f"\n📊 Sebelum split - Distribusi kelas:")
        print(f"  Penyakit: {disease_count} ({disease_ratio:.1%})")
        print(f"  Sehat: {healthy_count} ({(1-disease_ratio):.1%})")
        
        # CRITICAL WARNING: Check for class imbalance
        if disease_ratio > 0.7:
            print(f"  🚨 PERINGATAN: Dataset bias ke PENYAKIT! Rasio: {disease_ratio:.1%}")
            print(f"  💡 Model akan menggunakan class_weight='balanced' untuk mengatasi ini")
        elif disease_ratio < 0.3:
            print(f"  🚨 PERINGATAN: Dataset bias ke SEHAT! Rasio: {disease_ratio:.1%}")
            print(f"  💡 Model akan menggunakan class_weight='balanced' untuk mengatasi ini")
        
        # Melakukan split dengan stratify untuk menjaga proporsi kelas
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n📊 PEMBAGIAN DATA:")
        print(f"  Set latih: {X_train.shape[0]} sampel")
        print(f"  Set uji: {X_test.shape[0]} sampel")
        print(f"  Latih - Penyakit: {sum(y_train)}, Sehat: {len(y_train)-sum(y_train)}")
        print(f"  Uji - Penyakit: {sum(y_test)}, Sehat: {len(y_test)-sum(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Melakukan scaling fitur menggunakan StandardScaler"""
        print("\n📏 MELAKUKAN SCALING FITUR...")
        
        # Fit scaler dengan data latih dan transform keduanya
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Konversi kembali ke DataFrame agar mudah digunakan
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print("  ✅ Fitur berhasil di-scale!")
        print(f"  📊 Mean data latih yang di-scale: {X_train_scaled.mean().mean():.6f}")
        print(f"  📊 Std data latih yang di-scale: {X_train_scaled.std().mean():.6f}")
        
        return X_train_scaled, X_test_scaled
    
    def get_feature_info(self, data):
        """Mendapatkan statistik fitur untuk kolom numerik - METODE YANG DITAMBAHKAN"""
        print("\n📊 MENGHITUNG STATISTIK FITUR...")
        
        feature_info = {}
        
        # Ambil kolom numerik (kecuali kolom target jika ada)
        if 'Result' in data.columns:
            numeric_data = data.drop('Result', axis=1).select_dtypes(include=[np.number])
        else:
            numeric_data = data.select_dtypes(include=[np.number])
        
        # Hitung statistik untuk setiap kolom numerik
        for col in numeric_data.columns:
            feature_info[col] = {
                'mean': numeric_data[col].mean(),      # Rata-rata
                'std': numeric_data[col].std(),        # Standar deviasi
                'min': numeric_data[col].min(),        # Nilai minimum
                'max': numeric_data[col].max(),        # Nilai maximum
                'median': numeric_data[col].median(),  # Median (nilai tengah)
                'q25': numeric_data[col].quantile(0.25),  # Kuartil 25%
                'q75': numeric_data[col].quantile(0.75)   # Kuartil 75%
            }
        
        print(f"  ✅ Statistik dihitung untuk {len(feature_info)} fitur")
        return feature_info
    
    def prepare_single_prediction(self, input_data):
        """Mempersiapkan satu sampel untuk prediksi - DIPERBAIKI"""
        # Pastikan scaler sudah di-fit
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler belum di-fit! Latih model terlebih dahulu.")
        
        print(f"\n🔍 MEMPERSIAPKAN PREDIKSI UNTUK: {input_data}")
        
        # Konversi ke DataFrame jika input berupa dictionary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # CRITICAL FIX: Handle Gender encoding properly
        if 'Gender' in input_df.columns:
            print("  🔄 Encoding Gender...")
            # Create Gender_Male (1 if Male, 0 if Female)
            input_df['Gender_Male'] = (input_df['Gender'] == 'Male').astype(int)
            input_df = input_df.drop('Gender', axis=1)
            print(f"  ✅ Gender berhasil di-encode: {input_df['Gender_Male'].values[0]}")
        
        # Pastikan semua fitur yang diperlukan ada
        if self.feature_names:
            missing_features = set(self.feature_names) - set(input_df.columns)
            if missing_features:
                print(f"  ⚠️ Fitur yang hilang: {missing_features}")
                for feature in missing_features:
                    input_df[feature] = 0  # Isi dengan nilai default
                    print(f"  📝 Set {feature} = 0")
            
            # Pilih hanya fitur yang diperlukan dalam urutan yang benar
            input_df = input_df[self.feature_names]
        
        print(f"  📊 Ukuran input final: {input_df.shape}")
        print(f"  📋 Nilai input: {input_df.values[0]}")
        
        # Scale fitur
        input_scaled = self.scaler.transform(input_df)
        
        return pd.DataFrame(input_scaled, columns=self.feature_names if self.feature_names else input_df.columns)
    
    def get_normal_ranges(self, X, y):
        """Mendapatkan rentang normal dari pasien sehat"""
        healthy_samples = X[y == 0]  # Ambil sampel dari pasien sehat
        
        if len(healthy_samples) == 0:
            print("  ⚠️ Tidak ada sampel sehat yang ditemukan!")
            return None
        
        print(f"  📊 Ditemukan {len(healthy_samples)} sampel sehat")
        
        normal_ranges = {}
        # Hitung rentang normal untuk setiap fitur
        for col in X.columns:
            if 'Gender' not in col:  # Skip kolom gender untuk rentang
                q25, q50, q75 = healthy_samples[col].quantile([0.25, 0.5, 0.75])
                mean_val = healthy_samples[col].mean()
                normal_ranges[col] = {
                    'q25': q25,           # Kuartil 25%
                    'median': q50,        # Median
                    'q75': q75,           # Kuartil 75%
                    'mean': mean_val,     # Rata-rata
                    'min': healthy_samples[col].min(),    # Minimum
                    'max': healthy_samples[col].max()     # Maximum
                }
        
        return normal_ranges
    
    def create_normal_test_case(self):
        """Membuat kasus uji dengan nilai normal/sehat"""
        # Berdasarkan literatur medis untuk nilai normal
        normal_case = {
            'Age': 35,                    # Dewasa usia menengah
            'Gender': 'Male',             # Akan di-encode
            'Total Bilirubin': 0.8,       # Normal: 0.1-1.2 mg/dL
            'Direct Bilirubin': 0.2,      # Normal: 0.0-0.3 mg/dL  
            'Alkphos': 80,                # Normal: 44-147 IU/L
            'Sgpt': 25,                   # Normal: 7-56 IU/L (ALT)
            'Sgot': 25,                   # Normal: 10-40 IU/L (AST)
            'Total Protiens': 7.2,        # Normal: 6.0-8.3 g/dL
            'ALB': 4.2,                   # Normal: 3.5-5.0 g/dL
            'A/G Ratio': 1.5              # Normal: 1.1-2.5
        }
        
        return normal_case