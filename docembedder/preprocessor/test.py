import conversion
import csv

with open("/data/breakthrough-patents/year_index/year_df.csv",) as file:
    has_header = csv.Sniffer().has_header(file.read(1024))
    file.seek(0)  # Rewind.
    reader = csv.reader(file,delimiter='\t')
    if has_header:
        next(reader)
    years = {int(rows[0]):rows[1] for rows in reader}

bla = conversion.parse_raw(patent_input_fp="/data/breakthrough-patents/raw_datafiles/test.txt",year_lookup=years)

print(bla)
