"""Dataset helpers. Roboflow augmented copies of one source photo share a 32-hex hash in the file name;
that hash is the grouping key used for every leak-free split and photo-level statistic."""

import re
from collections import defaultdict
from pathlib import Path

_SOURCE_GROUP_PATTERN=re.compile(r"^(.*rf\.[0-9a-f]{32})")

def extract_source_group(stem:str)->str:
    """Return the stable grouping key for a file stem: everything up to and including the Roboflow hash,..."""
    m=_SOURCE_GROUP_PATTERN.match(stem)
    return m.group(1) if m else stem

def group_label_stems(labels_dir:Path,skip_dup:bool=True):
    """Scan a labels/ directory and group stems by source image."""
    groups=defaultdict(list)
    group_classes=defaultdict(set)
    for label_file in sorted(labels_dir.glob("*.txt")):
        stem=label_file.stem
        if skip_dup and "_dup" in stem:
            continue
        group_key=extract_source_group(stem)
        classes_here=set()
        for line in label_file.read_text().splitlines():
            if line.strip():
                classes_here.add(int(line.split()[0]))
        groups[group_key].append(stem)
        group_classes[group_key]|=classes_here
    return dict(groups),dict(group_classes)

def label_path_for(img_path)->Path:
    """Ultralytics rule: swap the LAST '/images/' path component for '/labels/' and the extension for .txt..."""
    p=str(img_path)
    sa,sb="/images/","/labels/"
    head,sep,tail=p.rpartition(sa)
    if not sep:
        raise ValueError(f"{p} has no '/images/' component")
    return Path(head+sb+tail).with_suffix(".txt")
