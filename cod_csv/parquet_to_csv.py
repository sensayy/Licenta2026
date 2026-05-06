import polars as pl

print("Converting bigram_bytes.parquet to bigram_bytes.csv... this may take a while")
pl.scan_parquet("bigram_bytes.parquet").sink_csv("bigram_bytes.csv")
print("Done! bigram_bytes.csv has been created.")