import xgboost as xgb

try:
    # 1. Check if XGBoost was built with CUDA support
    build_info = xgb.build_info()
    cuda_support = build_info.get('USE_CUDA', False) or build_info.get('use_cuda', False)
    
    print(f"XGBoost Version: {xgb.__version__}")
    print(f"Built with CUDA support: {cuda_support}")

    # 2. Try to initialize a tiny model on the GPU
    # This is the 'Acid Test' - it will crash here if drivers aren't right
    test_model = xgb.XGBClassifier(tree_method='hist', device='cuda')
    print("GPU initialization successful! Your hardware is detected.")
    
except Exception as e:
    print("\n[!] GPU check failed.")
    print(f"Error details: {e}")
    print("\nTroubleshooting steps:")
    print("- Ensure NVIDIA drivers are installed (run 'nvidia-smi' in cmd).")
    print("- Check if CUDA Toolkit is installed.")
    print("- If 'USE_CUDA' is False, reinstall with: pip install xgboost")