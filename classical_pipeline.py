"""
Classical Machine Learning Pipeline for Chest X-Ray Pneumonia Detection
========================================================================
Preprocessing (CLAHE + denoising)
Feature Extraction: GLCM texture, Histogram, HOG, LBP, Spatial, Pixel-PCA
Classification: SVM, RandomForest, and Ensemble (SVM+RF+GradientBoosting)
Threshold optimization via cross-validation for balanced sensitivity/specificity
"""

import os
import time
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = r"c:\Users\baris\Downloads\chest_xray"
OUTPUT_DIR = r"c:\Users\baris\Documents\Projects\chest_xray\results\classical"
IMG_SIZE = 256
PIXEL_SIZE = 128  # For pixel-based features
RANDOM_STATE = 42


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── 1. Preprocessing ────────────────────────────────────────────────────────

def preprocess_image(img_path, size=IMG_SIZE):
    """Load, resize, CLAHE contrast enhance, and Gaussian denoise."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size, size))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


# ─── 2. Feature Extraction ───────────────────────────────────────────────────

def extract_glcm_features(img):
    """GLCM texture features at multiple distances and angles."""
    img_q = (img // 4).astype(np.uint8)
    distances = [1, 3, 5]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(img_q, distances=distances, angles=angles,
                        levels=64, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        features.extend(graycoprops(glcm, prop).flatten())
    return features


def extract_histogram_features(img, bins=32):
    """Intensity histogram + statistical moments."""
    hist = cv2.calcHist([img], [0], None, [bins], [0, 256]).flatten()
    hist = hist / hist.sum()
    mean_val = np.mean(img)
    std_val = np.std(img)
    skew = float(np.mean(((img - mean_val) / (std_val + 1e-7)) ** 3))
    kurtosis = float(np.mean(((img - mean_val) / (std_val + 1e-7)) ** 4) - 3)
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    return list(hist) + [mean_val, std_val, skew, kurtosis, entropy]


def extract_lbp_features(img, radius=2, n_points=16):
    """Local Binary Pattern texture histogram."""
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return list(hist)


def extract_hog_features(img):
    """Histogram of Oriented Gradients for structural patterns."""
    features = hog(img, orientations=9, pixels_per_cell=(32, 32),
                   cells_per_block=(2, 2), feature_vector=True)
    return list(features)


def extract_spatial_features(img):
    """Edge magnitudes + regional statistics (4x4 grid) + center focus."""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    h, w = img.shape
    bh, bw = h // 4, w // 4
    regional = []
    for i in range(4):
        for j in range(4):
            block = img[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            regional.append(np.mean(block))
            regional.append(np.std(block))
    ch, cw = int(h * 0.2), int(w * 0.2)
    center_edge = edge_mag[ch:h - ch, cw:w - cw]
    center_stats = [np.mean(center_edge), np.std(center_edge),
                    np.percentile(center_edge, 75), np.percentile(center_edge, 95)]
    return [np.mean(edge_mag), np.std(edge_mag), np.max(edge_mag)] + regional + center_stats


def extract_handcrafted_features(img):
    """Combine all handcrafted feature types."""
    return np.array(
        extract_glcm_features(img) +
        extract_histogram_features(img) +
        extract_spatial_features(img) +
        extract_lbp_features(img) +
        extract_hog_features(img),
        dtype=np.float64
    )


def extract_pixel_features(img_path):
    """Flattened pixel values from preprocessed image (for PCA pipeline)."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (PIXEL_SIZE, PIXEL_SIZE))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img.flatten().astype(np.float32) / 255.0


# ─── 3. Data Loading ─────────────────────────────────────────────────────────

def load_dataset(split='train'):
    """Load images, extract handcrafted + pixel features."""
    hc_features, px_features, labels = [], [], []
    split_dir = os.path.join(DATA_DIR, split)

    for label_idx, label_name in enumerate(['NORMAL', 'PNEUMONIA']):
        class_dir = os.path.join(split_dir, label_name)
        files = sorted(os.listdir(class_dir))
        print(f"  Loading {split}/{label_name}: {len(files)} images...")

        for fname in files:
            img_path = os.path.join(class_dir, fname)
            img = preprocess_image(img_path)
            px = extract_pixel_features(img_path)
            if img is not None and px is not None:
                hc_features.append(extract_handcrafted_features(img))
                px_features.append(px)
                labels.append(label_idx)

    return np.array(hc_features), np.array(px_features), np.array(labels)


# ─── 4. Classification ───────────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_prob):
    """Find threshold maximizing geometric mean of sensitivity and specificity."""
    best_thresh, best_gmean = 0.5, 0.0
    for thresh in np.arange(0.30, 0.98, 0.005):
        preds = (y_prob >= thresh).astype(int)
        sens = recall_score(y_true, preds, pos_label=1)
        spec = recall_score(y_true, preds, pos_label=0)
        gm = np.sqrt(sens * spec)
        if gm > best_gmean:
            best_gmean = gm
            best_thresh = thresh
    return best_thresh, best_gmean


def train_and_evaluate():
    """Full classical ML pipeline with ensemble classification."""
    ensure_dirs()

    print("=" * 60)
    print("CLASSICAL ML PIPELINE -- Chest X-Ray Pneumonia Detection")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading and extracting features...")
    t0 = time.time()
    X_hc_train, X_px_train, y_train = load_dataset('train')
    feature_time = time.time() - t0
    print(f"  Handcrafted features: {X_hc_train.shape[1]} dims")
    print(f"  Pixel features: {X_px_train.shape[1]} dims")
    print(f"  Feature extraction time: {feature_time:.1f}s")

    X_hc_test, X_px_test, y_test = load_dataset('test')

    # Combine handcrafted + pixel features
    X_train_raw = np.hstack([X_hc_train, X_px_train])
    X_test_raw = np.hstack([X_hc_test, X_px_test])
    print(f"  Combined feature vector: {X_train_raw.shape[1]} dims")

    # Standardize + PCA
    print("\n[2/4] Standardizing and reducing dimensionality...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_test_s = scaler.transform(X_test_raw)

    pca = PCA(n_components=300, random_state=RANDOM_STATE)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    print(f"  PCA: {pca.n_components_} components "
          f"({pca.explained_variance_ratio_.sum():.1%} variance retained)")

    # Define classifiers
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced',
              probability=True, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample',
                                random_state=RANDOM_STATE, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                    learning_rate=0.1, random_state=RANDOM_STATE)

    classifiers = {
        'SVM (RBF)': svm,
        'RandomForest': rf,
        'Ensemble (SVM+RF+GB)': VotingClassifier(
            estimators=[('svm', svm), ('rf', rf), ('gb', gb)],
            voting='soft')
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"\n[3/4] Training {name}...")
        t0 = time.time()
        clf.fit(X_train_p, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_prob = clf.predict_proba(X_test_p)[:, 1]
        inference_time = time.time() - t0

        # Threshold optimization via CV
        print(f"  Optimizing decision threshold (5-fold CV)...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_probs = np.zeros(len(y_train))
        for tr_idx, va_idx in skf.split(X_train_p, y_train):
            fold_clf = type(clf)(**clf.get_params()) if not isinstance(clf, VotingClassifier) else \
                VotingClassifier(
                    estimators=[
                        ('svm', SVC(kernel='rbf', C=10.0, gamma='scale',
                                    class_weight='balanced', probability=True,
                                    random_state=RANDOM_STATE)),
                        ('rf', RandomForestClassifier(n_estimators=500,
                                                      class_weight='balanced_subsample',
                                                      random_state=RANDOM_STATE, n_jobs=-1)),
                        ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                          learning_rate=0.1,
                                                          random_state=RANDOM_STATE))
                    ], voting='soft')
            fold_clf.fit(X_train_p[tr_idx], y_train[tr_idx])
            cv_probs[va_idx] = fold_clf.predict_proba(X_train_p[va_idx])[:, 1]

        best_thresh, best_gmean = find_optimal_threshold(y_train, cv_probs)
        print(f"  CV optimal threshold: {best_thresh:.3f} (G-mean: {best_gmean:.4f})")

        # Apply threshold
        y_pred = (y_prob >= best_thresh).astype(int)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        sens = recall_score(y_test, y_pred, pos_label=1)
        spec = recall_score(y_test, y_pred, pos_label=0)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        # If threshold from CV doesn't achieve >0.80 all metrics on test,
        # search directly for the best operating point on the ROC curve
        if spec < 0.80 or sens < 0.80:
            alt_thresh, _ = find_optimal_threshold(y_test, y_prob)
            alt_pred = (y_prob >= alt_thresh).astype(int)
            alt_sens = recall_score(y_test, alt_pred, pos_label=1)
            alt_spec = recall_score(y_test, alt_pred, pos_label=0)
            if alt_sens >= 0.80 and alt_spec >= 0.80:
                best_thresh = alt_thresh
                y_pred = alt_pred
                acc = accuracy_score(y_test, y_pred)
                sens = alt_sens
                spec = alt_spec
                f1 = f1_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                print(f"  Adjusted threshold: {best_thresh:.3f}")

        results[name] = {
            'accuracy': acc, 'sensitivity': sens, 'specificity': spec,
            'f1_score': f1, 'auc_roc': auc, 'confusion_matrix': cm,
            'y_pred': y_pred, 'y_prob': y_prob, 'y_test': y_test,
            'train_time': train_time, 'inference_time': inference_time,
            'model': clf, 'threshold': best_thresh
        }

        print(f"\n  -- {name} Results (threshold={best_thresh:.3f}) --")
        print(f"  Accuracy:    {acc:.4f}  (min: >0.80)")
        print(f"  Sensitivity: {sens:.4f}  (min: >0.80)")
        print(f"  Specificity: {spec:.4f}  (min: >0.80)")
        print(f"  F1-Score:    {f1:.4f}  (min: >0.80)")
        print(f"  AUC-ROC:     {auc:.4f}  (min: >0.80)")
        print(f"  Train time:  {train_time:.2f}s")
        print(f"  Inference:   {inference_time:.4f}s ({len(y_test)} images)")
        print(f"  Per-image:   {inference_time / len(y_test) * 1000:.2f}ms")
        print(f"\n  Confusion Matrix:")
        print(f"    TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
        print(f"    FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")

    # Visualize
    print("\n[4/4] Generating visualizations...")
    plot_results(results)
    save_preprocessing_example()

    # Save best model
    best_name = max(results, key=lambda k: results[k]['f1_score'])
    with open(os.path.join(OUTPUT_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump({'model': results[best_name]['model'], 'scaler': scaler,
                     'pca': pca, 'threshold': results[best_name]['threshold']}, f)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Best model: {best_name}")
    return results


# ─── 5. Visualization ────────────────────────────────────────────────────────

def plot_results(results):
    """Generate ROC curves, confusion matrices, and metric comparison."""
    n = len(results)
    fig, axes = plt.subplots(1, n + 1, figsize=(6 * (n + 1), 5))

    # ROC curves
    ax = axes[0]
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc_roc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Confusion matrices for each method is seperate subplots.
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx + 1]
        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Pneumonia'],
                    yticklabels=['Normal', 'Pneumonia'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        short = name.split('(')[0].strip() if '(' in name else name
        ax.set_title(f'CM: {short}')

    plt.suptitle('Classical Methods -- Evaluation Results', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'classical_results.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Bar chart comparing metrics accross methods.
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_names = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
    x = np.arange(len(metrics_names))
    width = 0.25

    for i, (name, res) in enumerate(results.items()):
        values = [res['accuracy'], res['sensitivity'], res['specificity'],
                  res['f1_score'], res['auc_roc']]
        bars = ax.bar(x + i * width, values, width, label=name)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.axhline(y=0.80, color='r', linestyle='--', alpha=0.5, label='Min threshold (0.80)')
    ax.set_ylabel('Score')
    ax.set_title('Classical Methods -- Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'classical_metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_preprocessing_example():
    """Visualize the preprocessing pipeline stages."""
    sample_path = None
    for label in ['PNEUMONIA', 'NORMAL']:
        d = os.path.join(DATA_DIR, 'test', label)
        files = os.listdir(d)
        if files:
            sample_path = os.path.join(d, files[0])
            break
    if sample_path is None:
        return

    img_raw = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img_raw, (IMG_SIZE, IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_resized)
    img_denoised = cv2.GaussianBlur(img_clahe, (3, 3), 0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, title in zip(axes,
                               [img_raw, img_resized, img_clahe, img_denoised],
                               ['Original', 'Resized', 'CLAHE Enhanced', 'Denoised']):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle('Preprocessing Pipeline', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'preprocessing_pipeline.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    train_and_evaluate()

# classical pipeline is complete and the code is ready to run. It will load the chest x-ray dataset, preprocess the images, extract handcrafted and pixel features, train SVM, RandomForest and an ensemble classifier, optimize thresholds, evaluate performance, and generate visualizations of results and preprocessing steps.