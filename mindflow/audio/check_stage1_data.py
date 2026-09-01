import pandas as pd

for name in ["crema_d", "ravdess", "savee", "tess", "meld", "iemocap"]:
    path = f"D:/Capstone116_Vaish/Audio/processed/metadata/metadata_{name}.csv"
    df = pd.read_csv(path)
    n_speakers = df["speaker"].nunique() if "speaker" in df.columns else "no speaker col"
    print(f"{name}: {len(df)} rows, {n_speakers} speakers")
    if "emotion" in df.columns:
        print("  emotions:", df["emotion"].value_counts().to_dict())
    else:
        print("  no emotion column")
    print()
