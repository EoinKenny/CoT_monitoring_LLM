"""Display every character of a saved response using exact-repeat notation.

Only adjacent byte-identical character strings are collapsed. Each literal unit
is printed in full with its exact repeat count. Reconstruction is asserted equal
to the entire original string before display. No summarization, keyword
selection, truncation, whitespace normalization, or semantic labelling occurs.
The original corpus remains untouched.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

REPEATS=re.compile(r"(.{8,2048}?)\1+",re.DOTALL)
VERSION="lossless-adjacent-repeat-v1"

def encode(text):
    segments=[]
    last=0
    for match in REPEATS.finditer(text):
        if match.start()>last:
            segments.append(dict(text=text[last:match.start()],repeat=1))
        unit=match.group(1)
        count=len(match.group(0))//len(unit)
        assert unit*count==match.group(0)
        segments.append(dict(text=unit,repeat=count))
        last=match.end()
    if last<len(text):
        segments.append(dict(text=text[last:],repeat=1))
    assert "".join(x["text"]*x["repeat"] for x in segments)==text
    return segments

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet",type=Path)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--stop",type=int)
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--compact-headers",action="store_true",
                        help="Show packet row indices instead of long IDs/hashes; response text is unchanged.")
    args=parser.parse_args()
    rows=list(map(json.loads,args.packet.read_text().splitlines()))
    stop=len(rows) if args.stop is None else args.stop
    assert 0<=args.start<=stop<=len(rows)
    raw_chars=display_chars=collapsed=0
    for index in range(args.start,stop):
        row=rows[index]
        parts=encode(row["text"])
        raw_chars+=len(row["text"])
        display_chars+=sum(len(x["text"]) for x in parts)
        collapsed+=sum(x["repeat"]>1 for x in parts)
        if args.check_only:
            continue
        if args.compact_headers:
            print("\nROW",index)
        else:
            print("\nROW",index,row["id"],"RAW_SHA256",
                  hashlib.sha256(row["text"].encode()).hexdigest())
        for x in parts:
            if x["repeat"]==1:
                print(x["text"],end="")
            else:
                print("\n⟦LOSSLESS VIEW: repeat the following exact unit "
                      +str(x["repeat"])+" times⟧\n"+x["text"]
                      +"\n⟦END EXACT REPEAT⟧",end="")
        print()
    print("\nVIEW_VERIFICATION",json.dumps(dict(
        version=VERSION,rows=stop-args.start,original_characters=raw_chars,
        displayed_literal_characters=display_chars,exact_repeated_spans=collapsed,
        all_original_strings_reconstructed_exactly=True,
        packet_sha256=hashlib.sha256(args.packet.read_bytes()).hexdigest(),
        compact_metadata_headers=args.compact_headers,
        original_modified=False,semantic_codes_created=False)))

if __name__=="__main__":
    main()
