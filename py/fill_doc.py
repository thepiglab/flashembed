"""Fill the generated sections of docs/PROFILING.md from `final_report.py --doc` output.

  python final_report.py $EMBED_ROOT/out/final --doc > report.json
  python fill_doc.py report.json ../docs/PROFILING.md

Placeholders replaced in place: __SUMMARY__, __PARITY__, __TABLES__, __CTX__, __MM__, __OPS__.
"""

import json
import sys
from pathlib import Path

rep, doc = json.load(open(sys.argv[1])), Path(sys.argv[2])
s = doc.read_text()
fill = {
    "__SUMMARY__": rep["by_family"] + "\n" + rep["summary"].replace("### Summary", "Summary"),
    "__PARITY__": rep["parity"],
    "__TABLES__": rep["tables"],
    "__CTX__": rep["ctx"],
    "__MM__": rep["mm"],
    "__OPS__": rep["ops"],
}
missing = [k for k in fill if k not in s]
if missing:
    sys.exit(f"placeholders not found: {missing}")
for k, v in fill.items():
    s = s.replace(k, v.strip())
doc.write_text(s)
print(f"filled {len(fill)} sections into {doc}")
