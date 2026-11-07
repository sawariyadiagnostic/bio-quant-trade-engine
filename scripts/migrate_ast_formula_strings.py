import sqlite3
from src.domain.formula_ast import ASTNode

def migrate_formula_strings(db_path: str = "audit/alpha_elites.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute("SELECT rowid, formula_str FROM resident_elites WHERE formula_str LIKE 'ASTNode(%'").fetchall()
    print(f"Found {len(rows)} records with ASTNode repr strings.")
    fixed = 0
    for rowid, f_str in rows:
        try:
            node = eval(f_str, {"ASTNode": ASTNode})
            clean_f = node.to_formula_str()
            cur.execute("UPDATE resident_elites SET formula_str = ? WHERE rowid = ?", (clean_f, rowid))
            fixed += 1
        except Exception as e:
            pass
    conn.commit()
    print(f"Successfully converted {fixed}/{len(rows)} formulas to clean infix representation.")
    conn.close()

if __name__ == "__main__":
    migrate_formula_strings()
