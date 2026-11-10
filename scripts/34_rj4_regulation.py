#!/usr/bin/env python
"""RJ-4: regulation strata from RiverATLAS.

Each of the 62 pilot stations gets the HydroRIVERS v1.0 reach with the largest upstream
area within 0.01 degrees, read directly from HydroRIVERS; the nearest reach is the
sensitivity. RiverATLAS v1.0 supplies the degree of regulation of those reaches. The
catalogue states that `dor_pc_pva` stores percent multiplied by 10, so DOR >= 2 % is a
stored value of at least 20. The analysis is descriptive; see docs/revision_log.md.
"""
from __future__ import annotations

from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from hidroxmx.data.streams import LocalRoots
from hidroxmx.revision import (
    BASIN_SLUGS,
    TYPE_DIAGNOSTIC,
    check_snapshot,
    clean,
    folds_of,
    read_manifest,
    update_numbers,
)

SOURCE = "scripts/34_rj4_regulation.py"
RADIUS_DEG = 0.01
DOR_STORED_FACTOR = 10.0
DOR_REGULATED_PCT = 2.0
DELTA_H = (5, 7)
TOP_GAINS = 4
MECHANISMS = {
    "S-SIG t=1.0": "F0txfr-sig-alto-lerma-sweep-01",
    "S-SIG t=0.3": "F0txfr-sig-alto-lerma-temp03-01",
    "S-ATTR t=1.0": "F0txfr-attr-alto-lerma-sweep-01",
}


def pilot_stations(roots: LocalRoots) -> pd.DataFrame:
    cat = pd.read_csv(roots.selected_hidro_csv, dtype={"clave": str})
    pool = cat[cat["cuenca"].isin(BASIN_SLUGS)].copy()
    if len(pool) != 62:
        raise SystemExit(f"expected 62 pilot stations, found {len(pool)}")
    return pool


def assign_reaches(pool: pd.DataFrame, shapefile: Path) -> pd.DataFrame:
    pad = 2 * RADIUS_DEG
    bbox = (pool.longitud.min() - pad, pool.latitud.min() - pad,
            pool.longitud.max() + pad, pool.latitud.max() + pad)
    reaches = gpd.read_file(shapefile, bbox=bbox, columns=["HYRIV_ID", "UPLAND_SKM"])
    click.echo(f"[34_rj4] {len(reaches)} HydroRIVERS reaches in the pilot box")
    rows = []
    for _, st in pool.iterrows():
        point = gpd.points_from_xy([st.longitud], [st.latitud])[0]
        near = reaches.iloc[reaches.sindex.query(point.buffer(RADIUS_DEG), predicate="intersects")]
        row = {"clave": st.clave, "basin": st.cuenca, "nombre": st.nombre,
               "lat": st.latitud, "lon": st.longitud}
        if len(near):
            dist = near.distance(point)
            largest = near.loc[near["UPLAND_SKM"].idxmax()]
            closest = near.loc[dist.idxmin()]
            row.update({"hyriv_id": int(largest["HYRIV_ID"]), "upland_km2": largest["UPLAND_SKM"],
                        "hyriv_id_nearest": int(closest["HYRIV_ID"]),
                        "upland_nearest_km2": closest["UPLAND_SKM"]})
        rows.append(row)
    return pd.DataFrame(rows)


def read_dor(atlas: Path, ids: list[int]) -> pd.Series:
    listed = ",".join(str(i) for i in ids)
    table = pyogrio.read_dataframe(atlas, columns=["HYRIV_ID", "dor_pc_pva"], read_geometry=False,
                                   where=f"HYRIV_ID IN ({listed})")
    return table.set_index("HYRIV_ID")["dor_pc_pva"] / DOR_STORED_FACTOR


def fold_deltas(mirror: Path) -> pd.DataFrame:
    rows = []
    for basin, slug in BASIN_SLUGS.items():
        run = f"F0pub-{slug}-sweep-01"
        for clave in folds_of(mirror, run):
            m = read_manifest(mirror, run, clave)["metrics"]
            for h in DELTA_H:
                delta = m.get(f"nse_h{h}", np.nan) - m.get(f"persist_nse_h{h}", np.nan)
                rows.append({"clave": clave, "basin": basin, "h": h,
                             "comparison": "F0-PUB - persistence", "delta_nse": delta})
    lumped = "F0pub-alto-lerma-sweep-01"
    for name, run in MECHANISMS.items():
        for clave in folds_of(mirror, run):
            m = read_manifest(mirror, run, clave)["metrics"]
            ref = read_manifest(mirror, lumped, clave)["metrics"]
            for h in DELTA_H:
                delta = m.get(f"nse_h{h}", np.nan) - ref.get(f"nse_h{h}", np.nan)
                rows.append({"clave": clave, "basin": "Alto Lerma", "h": h,
                             "comparison": f"{name} - lumped", "delta_nse": delta})
    return pd.DataFrame(rows)


def strata_table(st: pd.DataFrame, deltas: pd.DataFrame, column: str) -> pd.DataFrame:
    joined = deltas.merge(st[["clave", column]], on="clave", how="left")
    joined = joined[joined["delta_nse"].notna()]
    joined["stratum"] = np.where(joined[column].isna(), "no reach",
                                 np.where(joined[column] >= DOR_REGULATED_PCT,
                                          "regulated", "not regulated"))
    rows = []
    scopes = [("All", joined[joined["comparison"].str.startswith("F0-PUB")])]
    scopes += [(b, joined[(joined["basin"] == b) & joined["comparison"].str.startswith("F0-PUB")])
               for b in BASIN_SLUGS]
    scopes += [("Alto Lerma", joined[~joined["comparison"].str.startswith("F0-PUB")])]
    for scope, frame in scopes:
        for keys, sub in frame.groupby(["comparison", "h", "stratum"], sort=False):
            rows.append({"scope": scope, "comparison": keys[0], "h": keys[1], "stratum": keys[2],
                         "n": len(sub), "median_delta_nse": sub["delta_nse"].median()})
    return pd.DataFrame(rows)


@click.command()
@click.option("--snapshot-root", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--hydrorivers", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--riveratlas", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out-dir", default="outputs/revision_joh", show_default=True,
              type=click.Path(path_type=Path))
@click.option("--results-dir", default="results/revision_joh", show_default=True,
              type=click.Path(path_type=Path))
def main(snapshot_root: Path, hydrorivers: Path, riveratlas: Path, out_dir: Path,
         results_dir: Path):
    roots = LocalRoots(snapshot_root)
    check_snapshot(roots)
    tables = results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    numbers = results_dir / "numbers_revision.json"

    st = assign_reaches(pilot_stations(roots), hydrorivers)
    ids = sorted({int(i) for i in pd.concat([st["hyriv_id"], st["hyriv_id_nearest"]]).dropna()})
    dor = read_dor(riveratlas, ids)
    click.echo(f"[34_rj4] DOR read for {len(dor)} of {len(ids)} reaches")
    st["dor_pct"] = st["hyriv_id"].map(dor)
    st["dor_nearest_pct"] = st["hyriv_id_nearest"].map(dor)
    st.to_csv(tables / "T_rj4_station_reaches.csv", index=False)

    deltas = fold_deltas(out_dir / "r2")
    strata = strata_table(st, deltas, "dor_pct")
    strata_near = strata_table(st, deltas, "dor_nearest_pct")
    strata.to_csv(tables / "T_rj4_regulation.csv", index=False)
    strata_near.to_csv(tables / "T_rj4_regulation_nearest.csv", index=False)

    lerma = deltas[(deltas["basin"] == "Alto Lerma") & (deltas["h"] == 5)
                   & (deltas["comparison"] == "F0-PUB - persistence")]
    lerma = lerma.merge(st[["clave", "nombre", "dor_pct"]], on="clave")
    lerma = lerma.sort_values("delta_nse", ascending=False)
    lerma.to_csv(tables / "T_rj4_alto_lerma_stations.csv", index=False)
    top = lerma.head(TOP_GAINS)
    gains = lerma["delta_nse"].clip(lower=0.0)

    def nested(frame: pd.DataFrame) -> dict:
        out: dict = {}
        for _, r in frame.iterrows():
            leaf = out.setdefault(r["scope"], {}).setdefault(r["comparison"], {}) \
                .setdefault(f"h{r['h']}", {})
            leaf[r["stratum"]] = {"n": r["n"], "mediana_delta_nse": r["median_delta_nse"]}
        return out

    regulated = st["dor_pct"] >= DOR_REGULATED_PCT
    update_numbers(numbers, "rj4", TYPE_DIAGNOSTIC, SOURCE, clean({
        "unidad": "dor_pc_pva / 10 = percent (RiverATLAS catalogue v1.0, sheet H07)",
        "umbral_pct": DOR_REGULATED_PCT,
        "estaciones": {
            "reguladas": int(regulated.sum()),
            "no_reguladas": int((st["dor_pct"] < DOR_REGULATED_PCT).sum()),
            "sin_tramo": int(st["dor_pct"].isna().sum()),
            "por_cuenca": st.assign(reg=regulated).groupby("basin")["reg"].sum().to_dict(),
        },
        "estratos": nested(strata), "estratos_tramo_cercano": nested(strata_near),
        "alto_lerma_mayores_ganancias": {
            "h": 5, "estaciones": top["clave"].tolist(),
            "parte_de_la_ganancia_positiva": gains.head(TOP_GAINS).sum() / gains.sum(),
            "reguladas": int((top["dor_pct"] >= DOR_REGULATED_PCT).sum()),
        },
    }))

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 100)
    click.echo(st.groupby("basin")["dor_pct"].describe().round(1).to_string())
    click.echo(strata.round(3).to_string(index=False))
    click.echo("no reach: " + str(st.loc[st["hyriv_id"].isna(), "clave"].tolist()))


if __name__ == "__main__":
    main()
