import argparse
import logging
from pathlib import Path

import duckdb


def main(
    path_to_modern_gold: Path | str,
    path_to_archival_gold: Path | str,
    output_path: Path | str,
):
    logger = logging.getLogger(__name__)

    logger.info("Regenerating latest_comprehensive.parquet")

    all_gold_modern = duckdb.read_parquet(
        str(Path(path_to_modern_gold, "*.parquet"))
    )
    latest_gold_archival_path = str(
        max(path_to_archival_gold.glob("*.parquet"))
    )
    latest_gold_archival = duckdb.read_parquet(latest_gold_archival_path)

    comprehensive = duckdb.sql("""
    WITH latest_report_dates AS
    (SELECT reference_date,
    MAX(report_date) AS latest_report_date
    FROM all_gold_modern
    GROUP BY reference_date),
    modern_vintages_mega AS
    (SELECT ag.report_date,
    ag.reference_date,
    ag.asof,
    ag.metric,
    ag.geo_type,
    ag.geo_value,
    ag.run_id,
    ag.disease,
          SUM(ag.value) AS value
    FROM all_gold_modern AS ag
    JOIN latest_report_dates AS lrd ON ag.reference_date = lrd.reference_date
   AND ag.report_date = lrd.latest_report_date
   GROUP BY ag.report_date,
            ag.reference_date,
            ag.asof,
            ag.metric,
            ag.geo_type,
            ag.geo_value,
            ag.run_id,
            ag.disease)
SELECT report_date,
       reference_date,
       metric,
       geo_type,
       geo_value,
       disease,
       value
FROM modern_vintages_mega
UNION ALL
SELECT report_date,
       reference_date,
       metric,
       geo_type,
       geo_value,
       disease,
       value
FROM latest_gold_archival
WHERE reference_date < (SELECT MIN(reference_date) FROM modern_vintages_mega)
""")

    comprehensive.to_parquet(str(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_modern_gold", type=Path)
    parser.add_argument("path_to_archival_gold", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    main(**vars(args))
