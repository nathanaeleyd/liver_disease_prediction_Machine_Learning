import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

class ModelTrainer:
    """Enhanced model trainer for liver disease prediction - VERSI YANG DIPERBAIKI"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def initialize_models(self):
        """Initialize different ML models - DIPERBAIKI DENGAN CLASS BALANCING"""
        print("🤖 Menginisialisasi model dengan pengaturan seimbang...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',  # CRITICAL FIX: Handle class imbalance
                random_state=42,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt'
            ),
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',  # CRITICAL FIX: Handle class imbalance
                random_state=42,
                max_iter=1000
            ),
            'SVM': SVC(
                class_weight='balanced',  # CRITICAL FIX: Handle class imbalance
                random_state=42, 
                probability=True,
                kernel='rbf'
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance'  # Give more weight to closer neighbors
            )
        }
        
        print(f"✅ Diinisialisasi {len(self.models)} model dengan class balancing")
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train all models and compare performance - DIPERBAIKI"""
        print("\n" + "="*60)
        print("🚀 TRAINING MODELS")
        print("="*60)
        
        # Check class distribution
        disease_count = sum(y_train)
        healthy_count = len(y_train) - disease_count
        disease_ratio = disease_count / len(y_train)
        
        print(f"Distribusi data latih:")
        print(f"   Penyakit: {disease_count} ({disease_ratio:.1%})")
        print(f"   Sehat: {healthy_count} ({(1-disease_ratio):.1%})")
        
        if disease_ratio > 0.6:
            print("   🚨 Dataset bias ke penyakit - balanced weights diterapkan")
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate comprehensive metrics
                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)
                precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'accuracy': test_acc,  # For backward compatibility
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_test_pred,
                    'overfitting': abs(train_acc - test_acc),
                    'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
                }
                
                print(f"  ✅ Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
                print(f"  📊 CV Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
                print(f"  📈 F1 Score: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
                
            except Exception as e:
                print(f"  ❌ Error training {name}: {e}")
                continue
        
        print(f"\n✅ Training selesai untuk {len(self.results)} model")
    
    def select_best_model(self):
        """Select the best performing model - DIPERBAIKI DENGAN COMPOSITE SCORING"""
        print("\n🏆 Memilih model terbaik...")
        
        if not self.results:
            print("❌ Tidak ada model yang dilatih!")
            return None, None
        
        best_score = -1
        best_name = None
        
        print("\n📊 Perbandingan Model:")
        for name, result in self.results.items():
            # Composite score: balance accuracy, F1, and overfitting penalty
            composite_score = (
                result['test_accuracy'] * 0.4 + 
                result['f1_score'] * 0.4 + 
                (1 - result['overfitting']) * 0.2  # Penalize overfitting
            )
            
            print(f"   {name}:")
            print(f"      Test Accuracy: {result['test_accuracy']:.3f}")
            print(f"      F1 Score: {result['f1_score']:.3f}")
            print(f"      Overfitting: {result['overfitting']:.3f}")
            print(f"      Composite Score: {composite_score:.3f}")
            
            if composite_score > best_score:
                best_score = composite_score
                best_name = name
                self.best_model = result['model']
        
        self.best_model_name = best_name
        print(f"\n🏆 BEST MODEL: {best_name} (Score: {best_score:.3f})")
        
        return self.best_model, best_name
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from the best model"""
        if self.best_model is None:
            print("❌ Tidak ada best model yang dipilih!")
            return None
            
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n📋 TOP 5 FITUR PENTING:")
            for idx, row in feature_importance.head().iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
            return feature_importance
        else:
            print(f"⚠️ {self.best_model_name} tidak mendukung feature importance")
            return None
    
    def save_model(self, filepath="models/liver_model_fixed.joblib"):
        """Save the best model - DIPERBAIKI UNTUK KONSISTENSI DENGAN FIX.PY"""
        if self.best_model is None:
            print("❌ Tidak ada best model untuk disimpan!")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            joblib.dump(self.best_model, filepath)
            
            # Save model info
            info_filepath = filepath.replace('.joblib', '_info.joblib')
            model_info = {
                'model_name': self.best_model_name,
                'performance': self.results[self.best_model_name],
                'feature_names': None  # Will be updated by caller
            }
            joblib.dump(model_info, info_filepath)
            
            print(f"✅ Model disimpan ke {filepath}")
            print(f"✅ Model info disimpan ke {info_filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error menyimpan model: {e}")
            return False
    
    def load_model(self, filepath="models/liver_model_fixed.joblib"):
        """Load a saved model"""
        try:
            self.best_model = joblib.load(filepath)
            print(f"✅ Model berhasil dimuat dari {filepath}")
            return self.best_model
        except Exception as e:
            print(f"❌ Error memuat model: {e}")
            return None
    
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            print("❌ Tidak ada model untuk prediksi")
            return None
        
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)
        
        return predictions, probabilities
    
    def test_specific_case(self, X_test_scaled, case_data, case_name="Test Case"):
        """Test model on specific case - METODE BARU DARI FIX.PY"""
        if self.best_model is None:
            print("❌ Tidak ada model yang tersedia!")
            return None
        
        print(f"🧪 Testing {case_name}...")
        
        # Make prediction
        prediction = self.best_model.predict(case_data)[0]
        probabilities = self.best_model.predict_proba(case_data)[0]
        
        result_text = "PENYAKIT" if prediction == 1 else "SEHAT"
        confidence_healthy = probabilities[0] * 100
        confidence_disease = probabilities[1] * 100
        
        print(f"🎯 {case_name} Result:")
        print(f"   Prediksi: {result_text}")
        print(f"   Confidence Sehat: {confidence_healthy:.1f}%")
        print(f"   Confidence Penyakit: {confidence_disease:.1f}%")
        
        return {
            'prediction': prediction,
            'result_text': result_text,
            'probabilities': probabilities,
            'confidence_healthy': confidence_healthy,
            'confidence_disease': confidence_disease
        }
    
    def get_model_summary(self):
        """Get summary of all model performances - DIPERBAIKI"""
        print("\n" + "="*60)
        print("📊 RINGKASAN PERBANDINGAN MODEL")
        print("="*60)
        
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'Model': name,
                'Test Accuracy': f"{result['test_accuracy']:.3f}",
                'Precision': f"{result['precision']:.3f}",
                'Recall': f"{result['recall']:.3f}",
                'F1 Score': f"{result['f1_score']:.3f}",
                'CV Score': f"{result['cv_mean']:.3f} (±{result['cv_std']:.3f})",
                'Overfitting': f"{result['overfitting']:.3f}",
                'Best': "✅" if name == self.best_model_name else ""
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def generate_insights(self):
        """Generate business insights - DIPERBAIKI"""
        print("\n" + "="*60)
        print("💼 BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        if self.best_model_name is None:
            return {}
        
        best_result = self.results[self.best_model_name]
        
        print(f"\n🔍 TEMUAN KUNCI:")
        print(f"1. PERFORMA MODEL:")
        print(f"   • Model terbaik: {self.best_model_name}")
        print(f"   • Akurasi yang dicapai: {best_result['test_accuracy']:.1%}")
        print(f"   • F1 Score: {best_result['f1_score']:.3f}")
        print(f"   • Stabilitas CV: ±{best_result['cv_std']:.3f}")
        
        # Determine reliability
        reliability = 'Tinggi' if best_result['overfitting'] < 0.1 else 'Sedang'
        
        # Generate recommendation
        acc = best_result['test_accuracy']
        overfitting = best_result['overfitting']
        
        if acc > 0.85 and overfitting < 0.1:
            recommendation = "Siap untuk implementasi produksi"
        elif acc > 0.75 and overfitting < 0.15:
            recommendation = "Baik untuk implementasi dengan monitoring"
        else:
            recommendation = "Perlu optimasi lebih lanjut sebelum implementasi"
        
        print(f"\n📈 REKOMENDASI:")
        print(f"1. APLIKASI KLINIS:")
        print(f"   • Implementasikan {self.best_model_name} untuk skrining penyakit hati")
        print(f"   • Fokus pada biomarker kunci untuk deteksi dini")
        print(f"   • Reliabilitas model: {reliability}")
        print(f"   • Status: {recommendation}")
        
        print(f"\n2. LANGKAH SELANJUTNYA:")
        print(f"   • Validasi model dengan dataset eksternal")
        print(f"   • Kembangkan strategi deployment")
        print(f"   • Buat dashboard monitoring")
        print(f"   • Training ulang berkala dengan data baru")
        
        insights = {
            'best_model': self.best_model_name,
            'accuracy': best_result['test_accuracy'],
            'precision': best_result['precision'],
            'recall': best_result['recall'],
            'f1_score': best_result['f1_score'],
            'cv_stability': best_result['cv_std'],
            'reliability': reliability,
            'recommendation': recommendation,
            'recommendations': [
                f'Implementasikan {self.best_model_name} untuk skrining',
                'Fokus pada biomarker kunci',
                'Training ulang berkala diperlukan'
            ]
        }
        
        return insights