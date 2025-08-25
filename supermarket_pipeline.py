import argparse
import os
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
from textblob import TextBlob

# ----------------------------
# Utility: Safe column matching
# ----------------------------
def normalize_header(h: str) -> str:
    return re.sub(r'\s+', ' ', str(h).strip().lower().replace('%', 'percent'))

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {col: normalize_header(col) for col in df.columns}
    cand_norm = [normalize_header(c) for c in candidates]
    for original, normed in norm_map.items():
        if normed in cand_norm:
            return original
    for original, normed in norm_map.items():
        for c in cand_norm:
            if c in normed:
                return original
    return None

# ----------------------------
# Text cleaning helpers
# ----------------------------
def clean_text(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r'[^A-Za-z0-9\s\-\.\&]', '', s)  # remove junk chars
    s = re.sub(r'\s+', ' ', s)
    return s

def spell_correct(text):
    return str(text) if pd.notna(text) else ""


def clean_supplier_name(x: str) -> str:
    s = clean_text(x)
    s = re.sub(r'\b(pvt\.?|private)\s+(ltd\.?|limited)\b', 'Pvt Ltd', s, flags=re.I)
    s = re.sub(r'\b(ltd\.?|limited)\b', 'Ltd', s, flags=re.I)
    s = re.sub(r'\b(co\.?|company)\b', 'Co', s, flags=re.I)
    return s.strip()

# ----------------------------
# Supplier alias handling
# ----------------------------
def load_aliases(path: Optional[str]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return alias_map
    df = pd.read_csv(path)
    if 'alias' not in df.columns or 'canonical' not in df.columns:
        raise ValueError("Alias file must have columns: alias, canonical")
    for _, row in df.iterrows():
        alias = str(row['alias']).strip()
        canonical = clean_supplier_name(str(row['canonical']))
        if alias:
            alias_map[normalize_header(alias)] = canonical
    return alias_map

def apply_alias(supplier: str, alias_map: Dict[str, str]) -> str:
    key = normalize_header(supplier)
    return alias_map.get(key, supplier)

# ----------------------------
# Heuristic supplier inference
# ----------------------------
def infer_supplier_from_product(product: str) -> str:
    if not product:
        return "Unknown Supplier"
    tokens = product.split()
    if tokens:
        return clean_supplier_name(tokens[0])
    return "Unknown Supplier"

# ----------------------------
# Expanded Category rules (40+ groups)
# ----------------------------
CATEGORY_RULES: List[Tuple[str, List[str]]] = [
    ("Beverages - Tea & Coffee", ["tea","coffee","cappuccino","latte"]),
    ("Beverages - Soft Drinks", ["cola","soda","sprite","fanta","coke"]),
    ("Beverages - Juices", ["juice","mango drink","orange drink","fruit drink"]),
    ("Beverages - Water", ["water","mineral","aqua"]),
    ("Snacks - Chips", ["chips","lays","kurkure"]),
    ("Snacks - Biscuits & Cookies", ["biscuit","cookie","cracker"]),
    ("Snacks - Namkeen", ["namkeen","mixture","sev","bhujia"]),
    ("Snacks - Bakery", ["cake","pastry","donut"]),
    ("Dairy - Milk", ["milk","toned milk","curd","paneer"]),
    ("Dairy - Cheese & Butter", ["butter","cheese","cream","ghee"]),
    ("Dairy - Ice Cream", ["ice cream","kulfi","frozen dessert"]),
    ("Staples - Grains", ["rice","wheat","atta","flour","maida"]),
    ("Staples - Pulses", ["dal","lentil","pulses","moong","chana"]),
    ("Staples - Sugar & Salt", ["sugar","salt","jaggery"]),
    ("Staples - Oils", ["oil","sunflower","mustard","groundnut","sesame","coconut"]),
    ("Spices", ["masala","spice","turmeric","chili","cumin","coriander","pepper","cardamom","clove"]),
    ("Breakfast", ["corn flakes","muesli","oats","cereal","porridge"]),
    ("Bakery", ["bread","bun","roll"]),
    ("Frozen Foods", ["frozen","nuggets","fries","paratha","peas"]),
    ("Meat & Seafood", ["chicken","mutton","fish","prawn","egg","beef"]),
    ("Condiments & Sauces", ["ketchup","sauce","mayonnaise","chutney","vinegar","soy","mustard sauce"]),
    ("Sweets & Confectionery", ["chocolate","candy","toffee","sweet","laddu","barfi"]),
    ("Health & Nutrition", ["supplement","protein","vitamin","herbal","glucose","whey"]),
    ("Personal Care - Soap", ["soap","facewash","cleanser"]),
    ("Personal Care - Hair", ["shampoo","conditioner","hair oil"]),
    ("Personal Care - Oral", ["toothpaste","toothbrush","mouthwash"]),
    ("Personal Care - Skin", ["cream","lotion","deodorant","perfume"]),
    ("Baby Care", ["diaper","baby","wipes","infant","formula"]),
    ("Household - Cleaning", ["detergent","cleaner","dishwash","toilet","phenyl","disinfectant"]),
    ("Household - Paper", ["tissue","napkin","paper towel","foil","cup","plate"]),
    ("Stationery", ["pen","pencil","notebook","marker","eraser","glue","tape","stapler"]),
    ("Electronics - Small", ["battery","bulb","charger","cable"]),
    ("Pet Care", ["dog","cat","pet","litter","treat"]),
    ("Alcohol", ["whisky","vodka","beer","rum","wine"]),
    ("Cigarettes & Tobacco", ["cigarette","tobacco","cigar"]),
    ("Medicines & OTC", ["paracetamol","tablet","syrup","capsule"]),
    ("Clothing & Apparel", ["shirt","pant","saree","jeans","dress"]),
    ("Footwear", ["shoe","slipper","sandal"]),
    ("Accessories", ["bag","belt","wallet","cap","watch"]),
    ("Others", [""])
]
CATEGORY_REGEXES = [(cat, re.compile("|".join([re.escape(k) for k in kws]), re.I)) for cat, kws in CATEGORY_RULES if kws]

def classify_category(product_name: str) -> str:
    if not product_name:
        return "Others"
    for cat, creg in CATEGORY_REGEXES:
        if cat == "Others":
            continue
        if creg.search(product_name.lower()):
            return cat
    return "Others"

# ----------------------------
# Data cleaning
# ----------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Remove blanks
    df = df[df["Product Name"].str.strip() != ""]
    # Drop obvious test/fake records (non-capturing group → avoids warning)
    fake_patterns = re.compile(r"(?:test|dummy|sample|asdf|qwerty|xxx|abc|1234)", re.I)
    df = df[~df["Product Name"].str.contains(fake_patterns, na=False)]
    # Deduplicate
    df = df.drop_duplicates()
    # Spell correct product names (optional, can be slow)
    df["Product Name"] = df["Product Name"].apply(spell_correct)
    return df

def clean_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[col] = df[col].round(2)
            df[col] = df[col].clip(lower=0)  # remove negatives
    return df

# ----------------------------
# Pipeline
# ----------------------------
def run_pipeline(infile: str, outdir: str, sheet: Optional[str], supplier_alias_file: Optional[str]) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Read raw (no header)
    if infile.lower().endswith((".xlsx", ".xls")):
        raw_df = pd.read_excel(infile, sheet_name=sheet or 0, header=None)
    else:
        raw_df = pd.read_csv(infile, header=None)

    # Find header row automatically
    header_row = None
    for i, row in raw_df.iterrows():
        if row.astype(str).str.contains("Product", case=False).any() or row.astype(str).str.contains("S.No", case=False).any():
            header_row = i
            break
    if header_row is None:
        raise ValueError("❌ Could not detect header row with Product Name or S.No")

    # Re-read with correct header
    if infile.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(infile, sheet_name=sheet or 0, header=header_row, dtype=str)
    else:
        df = pd.read_csv(infile, header=header_row, dtype=str)

    # Normalize column names safely
    df.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in df.columns]

    # Find required columns
    col_map = {
        "S.No": find_col(df, ["S.No","S No","Serial"]),
        "Barcode": find_col(df, ["Barcode","Bar Code"]),
        "Product Name": find_col(df, ["Product Name","Item","Product"]),
        "Quantity": find_col(df, ["Quantity","Qty"]),
        "Stock Value": find_col(df, ["Stock Value","Value"]),
        "Profit": find_col(df, ["Profit"]),
        "Profit %": find_col(df, ["Profit %","Profit Percent"]),
        "Supplier Name": find_col(df, ["Supplier Name","Agency","Vendor","Brand"])
    }

    if not col_map["Product Name"]:
        raise ValueError(f"❌ Could not find Product Name column. Columns: {list(df.columns)}")

    # Rename standardized columns
    for std, actual in col_map.items():
        if actual and std != actual:
            df.rename(columns={actual: std}, inplace=True)

    # Clean text
    df["Product Name"] = df["Product Name"].apply(clean_text)
    if "Supplier Name" not in df:
        df["Supplier Name"] = df["Product Name"].apply(infer_supplier_from_product)
    else:
        df["Supplier Name"] = df["Supplier Name"].fillna("").apply(lambda s: clean_supplier_name(s) if s else "Unknown Supplier")

    # Aliases
    alias_map = load_aliases(supplier_alias_file)
    if alias_map:
        df["Supplier Name"] = df["Supplier Name"].apply(lambda s: apply_alias(s, alias_map))

    # Category classification
    df["Category"] = df["Product Name"].apply(classify_category)
    df["Sales Rank"] = range(1, len(df) + 1)

    # Data cleaning
    df = clean_dataframe(df)
    df = clean_numeric(df, ["Quantity","Stock Value","Profit","Profit %"])

    # Supplier mapping
    supplier_map_df = df[["Product Name","Supplier Name"]].drop_duplicates()

    # Export
    supplier_map_df.to_csv(os.path.join(outdir,"supplier_mapping.csv"), index=False, encoding="utf-8-sig")
    df.to_csv(os.path.join(outdir,"final_dataset.csv"), index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(os.path.join(outdir,"final_dataset.xlsx"), engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Cleaned Data")
        supplier_map_df.to_excel(xw, index=False, sheet_name="Supplier Mapping")

    print(f"[OK] Rows: {len(df)} | Products: {supplier_map_df['Product Name'].nunique()} | Suppliers: {supplier_map_df['Supplier Name'].nunique()}")
    print(f"Files written to: {outdir}")

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--infile", required=True)
    p.add_argument("--sheet", default=None)
    p.add_argument("--outdir", required=True)
    p.add_argument("--supplier-aliases", default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.infile, args.outdir, args.sheet, args.supplier_aliases)
