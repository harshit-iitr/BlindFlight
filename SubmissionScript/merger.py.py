
import pandas as pd
import re
import os

# ==========================================
# CONFIGURATION
# ==========================================
MAIN_FILE   = "submission.csv"      # The V13 file you just generated
BACKUP_FILE = "Backup Submission (By RN34 ungridded).csv"          # Your backup file (any format)
OUTPUT_FILE = "final_submission.csv"

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def clean_id(val):
    """ Converts '0001.png', '1.png', '0001' -> 1 """
    s = str(val).lower().strip()
    s = s.replace(".png", "").replace(".jpg", "")
    try:
        return int(s)
    except:
        return None # Should not happen if data is clean

def clean_path(val):
    """ Converts Coords OR uppercase dirs to 'ldru' """
    if pd.isna(val) or val == "" or str(val).lower() == "nan":
        return None

    val = str(val).strip()

    # CASE 1: Coordinate String like "(0,0)->(0,1)"
    if "->" in val and "(" in val:
        matches = re.findall(r"\((\d+),(\d+)\)", val)
        path = [(int(r), int(c)) for r, c in matches]
        dirs = []
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            if r2 > r1: dirs.append("d")
            elif r2 < r1: dirs.append("u")
            elif c2 > c1: dirs.append("r")
            elif c2 < c1: dirs.append("l")
        return "".join(dirs)

    # CASE 2: Direction String like "RRDD" or "rrdd"
    return val.lower()

# ==========================================
# 2. EXECUTION
# ==========================================
print("🔄 Loading files...")

# --- LOAD MAIN ---
if not os.path.exists(MAIN_FILE):
    print(f"❌ Error: {MAIN_FILE} not found. Generate it first!")
    exit()

df_main = pd.read_csv(MAIN_FILE)
# Standardize Columns
if "Image" in df_main.columns: df_main.rename(columns={"Image": "image_id"}, inplace=True)
if "Path" in df_main.columns: df_main.rename(columns={"Path": "path"}, inplace=True)

# --- LOAD BACKUP ---
if os.path.exists(BACKUP_FILE):
    df_backup = pd.read_csv(BACKUP_FILE)
    # Heuristic to find columns
    cols = df_backup.columns
    # Rename 1st col to image_id, 2nd to path (assuming standard structure)
    df_backup.rename(columns={cols[0]: "image_id", cols[1]: "path"}, inplace=True)
    print(f"✅ Backup loaded ({len(df_backup)} rows)")
else:
    print("⚠️  Backup file not found. Creating empty backup.")
    df_backup = pd.DataFrame(columns=["image_id", "path"])

# ==========================================
# 3. STANDARDIZATION
# ==========================================
print("🔧 Standardizing IDs and Paths...")

# Clean IDs to Integer
df_main['clean_id'] = df_main['image_id'].apply(clean_id)
df_backup['clean_id'] = df_backup['image_id'].apply(clean_id)

# Clean Paths to 'ldru'
df_main['clean_path'] = df_main['path'].apply(clean_path)
df_backup['clean_path'] = df_backup['path'].apply(clean_path)

# ==========================================
# 4. MERGE & FILL
# ==========================================
print("🩹 Merging and patching holes...")

# Merge on the clean Integer ID
df_final = df_main.merge(df_backup[['clean_id', 'clean_path']],
                         on='clean_id',
                         how='left',
                         suffixes=('', '_backup'))

# Logic: Use Main Path. If None, use Backup Path.
def fill_strategy(row):
    main_p = row['clean_path']
    back_p = row['clean_path_backup']

    if main_p is not None and len(main_p) > 0:
        return main_p
    return back_p

df_final['final_path'] = df_final.apply(fill_strategy, axis=1)

# ==========================================
# 5. OUTPUT FORMATTING
# ==========================================
# Select only required columns
output_df = df_final[['clean_id', 'final_path']].copy()
output_df.columns = ['image_id', 'path']

# Sort by ID
output_df = output_df.sort_values(by="image_id")

# Check for remaining nulls
missing = output_df['path'].isna().sum()
if missing > 0:
    print(f"⚠️  Warning: {missing} paths are still empty (missing in both main and backup).")
else:
    print("✅ All paths filled successfully.")

# Save
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"🚀 Saved Final Submission to: {OUTPUT_FILE}")
print(output_df.head())