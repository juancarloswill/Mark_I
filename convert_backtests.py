# guarda como: convert_backtests.py
import argparse, os
import pandas as pd

# importa la misma load_backtest_csv robusta (o pega aquí su código)
from audit_resultados import load_backtest_csv  # si lo dejaste ahí

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Lista de CSVs backtest crudos")
    ap.add_argument("--outdir", default="outputs/backtest_plots/normalized", help="Carpeta salida")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for path in args.inputs:
        name = os.path.splitext(os.path.basename(path))[0]
        df = load_backtest_csv(path)
        if df is None or df.empty:
            print(f"[WARN] No pude normalizar {path}")
            continue
        out_csv = os.path.join(args.outdir, f"{name}_normalized.csv")
        out_xlsx = os.path.join(args.outdir, f"{name}_normalized.xlsx")
        # preserva índice si es datetime
        if isinstance(df.index, pd.DatetimeIndex):
            df.to_csv(out_csv, index=True)
        else:
            df.to_csv(out_csv, index=False)
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as wr:
            df.to_excel(wr, index=True if isinstance(df.index, pd.DatetimeIndex) else False, sheet_name="data")
        print(f"✅ {name}: guardado {out_csv} y {out_xlsx}")

if __name__ == "__main__":
    main()
