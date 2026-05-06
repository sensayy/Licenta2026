import duckdb


duckdb.query("COPY (SELECT * FROM 'unigram_bytes.csv') TO 'unigrams.parquet' (FORMAT PARQUET)")


# This joins the files and streams the result directly to a new CSV
duckdb.query("""
    COPY (
        SELECT 
            u.*, 
            b.* EXCLUDE (Id, filesize, Class)
        FROM 'unigrams.parquet' u
        INNER JOIN 'bigram_bytes.parquet' b ON u.Id = b.Id
    ) TO 'merged_features.parquet' (FORMAT PARQUET)
""")