# One-time script to populate Arelle's local taxonomy cache by loading
# representative instances (one per taxonomy year) in online mode.
# After this, all instances can be loaded in offline mode using the cache.

import os
import re
import tempfile

from executable_finmr.configs.settings import OUTPUT_DIR
from executable_finmr.data.load_finmr import load_finmr

SECTION_FILE_MAP = {
    "Schema document": "schema.xsd",
    "Presentation linkbase document": "presentation.xml",
    "Calculation linkbase document": "calculation.xml",
    "Definition linkbase document": "definition.xml",
    "Label linkbase document": "label.xml",
    "Instance document": "instance.xml",
}

LINKBASE_REFS = """
  <annotation><appinfo>
    <link:linkbaseRef xlink:href="presentation.xml" xlink:role="http://www.xbrl.org/2003/role/presentationLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
    <link:linkbaseRef xlink:href="calculation.xml" xlink:role="http://www.xbrl.org/2003/role/calculationLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
    <link:linkbaseRef xlink:href="definition.xml" xlink:role="http://www.xbrl.org/2003/role/definitionLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
    <link:linkbaseRef xlink:href="label.xml" xlink:role="http://www.xbrl.org/2003/role/labelLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
  </appinfo></annotation>
"""


def reconstruct_package(inst, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    instance_doc = inst.sections.get("Instance document", "")
    m = re.search(r'xlink:href="([^"]+\.xsd)"', instance_doc)
    orig_schema = m.group(1) if m else None

    for section_name, filename in SECTION_FILE_MAP.items():
        content = inst.sections.get(section_name, "")
        if not content:
            continue
        if orig_schema:
            content = content.replace(orig_schema, "schema.xsd")
        with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)

    schema_path = os.path.join(out_dir, "schema.xsd")
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            sc = f.read()
        if "linkbaseRef" not in sc:
            for tag in ["</schema>", "</xs:schema>"]:
                sc = sc.replace(tag, LINKBASE_REFS + tag)
            with open(schema_path, "w") as f:
                f.write(sc)


def main():
    from arelle.api.Session import Session
    from arelle.RuntimeOptions import RuntimeOptions

    instances = load_finmr()
    inst_by_id = {i.id: i for i in instances}

    rep_ids = {
        "2021": 0,
        "2022": 11,
        "2023": 63,
        "2024": 87,
    }

    for year, iid in sorted(rep_ids.items()):
        inst = inst_by_id[iid]
        print(f"\n{'='*60}")
        print(f"Caching taxonomy year {year} (instance {iid}, {inst.dqc_id})")
        print(f"{'='*60}")

        tmpdir = os.path.join(str(OUTPUT_DIR), f"taxonomy_cache_tmp/{year}")
        reconstruct_package(inst, tmpdir)

        session = Session()
        opts = RuntimeOptions(
            entrypointFile=os.path.join(tmpdir, "instance.xml"),
            internetConnectivity="online",
            keepOpen=True,
            logFile="logToBuffer",
        )
        success = session.run(opts)
        models = session.get_models()
        if models:
            model = models[0]
            print(f"  Success: {success}")
            print(f"  Facts: {len(model.facts)}")
            print(f"  Concepts: {len(model.qnameConcepts)}")
        else:
            print(f"  FAILED to load model")
            logs = session.get_logs("text")
            if logs:
                print(f"  Logs: {logs[:500]}")
        session.close()

    print("\nVerifying offline mode works...")
    for year, iid in sorted(rep_ids.items()):
        tmpdir = os.path.join(str(OUTPUT_DIR), f"taxonomy_cache_tmp/{year}")
        session = Session()
        opts = RuntimeOptions(
            entrypointFile=os.path.join(tmpdir, "instance.xml"),
            internetConnectivity="offline",
            keepOpen=True,
            logFile="logToBuffer",
        )
        success = session.run(opts)
        models = session.get_models()
        if models:
            model = models[0]
            print(f"  {year} offline: facts={len(model.facts)} concepts={len(model.qnameConcepts)}")
        else:
            print(f"  {year} offline: FAILED")
        session.close()

    print("\nDone. Taxonomy cache populated.")


if __name__ == "__main__":
    main()
