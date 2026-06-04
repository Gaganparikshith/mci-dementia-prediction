"""
main.py — project entry point
Prints a summary of available phases and run commands.
"""

print("=" * 60)
print("  AI-Based Early Prediction of MCI Progression")
print("  Research Pipeline — MIT Manipal")
print("=" * 60)
print()
print("  Available phases:")
print("    Phase 1   — Clinical-only XGBoost")
print("    Phase 1.5 — Clinical + MRI Volumes")
print("    Phase 2   — Raw MRI CNN (ResNet-18)")
print("    Phase 3   — Clinical + CNN Fusion")
print("    Website   — Streamlit demo app")
print()
print("  Run individual phases:")
print("    python Phase_1_Clinical/train_clinical.py")
print("    python Phase_1_5_Clinical_MRI_Volumes/train_clinical_plus_volumes.py")
print("    python Phase_2_Raw_MRI_CNN/Train_mri_cnn_2d.py")
print("    python Phase_3_Fusion/Train_fusion_clinical_mri.py")
print("    streamlit run Website/streamlit_app.py")
print()
print("  Data root (external):")
print("    C:\\Users\\ASUS\\Desktop\\Research Resources\\DementiaResearch")
print("=" * 60)
