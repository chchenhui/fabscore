# Reconstructs per-instance XBRL packages from FinMR query sections.
# Writes 6 files (instance.xml, schema.xsd, presentation/calculation/definition/label.xml)
# with href rewriting so Arelle can resolve all DTS references locally.

import os
import re
from pathlib import Path

from executable_finmr.data.load_finmr import FinMRInstance

SECTION_FILE_MAP = {
    "Schema document": "schema.xsd",
    "Presentation linkbase document": "presentation.xml",
    "Calculation linkbase document": "calculation.xml",
    "Definition linkbase document": "definition.xml",
    "Label linkbase document": "label.xml",
    "Instance document": "instance.xml",
}

LINKBASE_REFS_TEMPLATE = """<annotation xmlns="http://www.w3.org/2001/XMLSchema" xmlns:link="http://www.xbrl.org/2003/linkbase" xmlns:xlink="http://www.w3.org/1999/xlink"><appinfo>
    <link:linkbaseRef xlink:href="presentation.xml" xlink:role="http://www.xbrl.org/2003/role/presentationLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
    <link:linkbaseRef xlink:href="calculation.xml" xlink:role="http://www.xbrl.org/2003/role/calculationLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
    <link:linkbaseRef xlink:href="definition.xml" xlink:role="http://www.xbrl.org/2003/role/definitionLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
    <link:linkbaseRef xlink:href="label.xml" xlink:role="http://www.xbrl.org/2003/role/labelLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
  </appinfo></annotation>"""

LINKBASE_REFS_WITH_LINK_NS = """<link:annotation><link:appinfo>
    <link:linkbaseRef xlink:href="presentation.xml" xlink:role="http://www.xbrl.org/2003/role/presentationLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
    <link:linkbaseRef xlink:href="calculation.xml" xlink:role="http://www.xbrl.org/2003/role/calculationLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
    <link:linkbaseRef xlink:href="definition.xml" xlink:role="http://www.xbrl.org/2003/role/definitionLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
    <link:linkbaseRef xlink:href="label.xml" xlink:role="http://www.xbrl.org/2003/role/labelLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
  </link:appinfo></link:annotation>"""


def _detect_original_schema_name(inst: FinMRInstance) -> str | None:
    instance_doc = inst.sections.get("Instance document", "")
    m = re.search(r'xlink:href="([^"]+\.xsd)"', instance_doc)
    if m:
        href = m.group(1)
        if "://" not in href:
            return href
    schema_doc = inst.sections.get("Schema document", "")
    m = re.search(r'targetNamespace="([^"]*?/(\w[\w.-]+))"', schema_doc)
    if m:
        return None
    return None


def _sanitize_xml(content: str) -> str:
    content = content.replace("&amp;lt;", "&lt;")
    content = content.replace("&amp;gt;", "&gt;")
    content = content.replace("&amp;amp;", "&amp;")
    content = content.replace("&amp;quot;", "&quot;")
    content = content.replace("&amp;apos;", "&apos;")
    return content


def _repair_truncated_xml(content: str, root_tag: str = "") -> str:
    stripped = content.rstrip()
    if not stripped:
        return content

    expected_roots = {
        "instance.xml": ["xbrl", "xbrli:xbrl"],
        "schema.xsd": ["schema", "xs:schema", "xsd:schema"],
        "presentation.xml": ["link:linkbase", "linkbase"],
        "calculation.xml": ["link:linkbase", "linkbase"],
        "definition.xml": ["link:linkbase", "linkbase"],
        "label.xml": ["link:linkbase", "linkbase"],
    }

    possible_roots = expected_roots.get(root_tag, [])
    for rt in possible_roots:
        if stripped.endswith(f"</{rt}>"):
            return content

    if not stripped.endswith(">"):
        last_lt = stripped.rfind("<")
        if last_lt >= 0:
            stripped = stripped[:last_lt].rstrip()

    open_tags = []
    for m in re.finditer(r"<(/?)([a-zA-Z][\w:.-]*)[^>]*/?>|<(/?)([a-zA-Z][\w:.-]*)[^>]*>", stripped):
        is_close = m.group(1) or m.group(3)
        tag_name = m.group(2) or m.group(4)
        full = m.group()
        if full.endswith("/>"):
            continue
        if is_close:
            if open_tags and open_tags[-1] == tag_name:
                open_tags.pop()
        else:
            open_tags.append(tag_name)

    closing = "\n".join(f"</{t}>" for t in reversed(open_tags))
    return stripped + "\n" + closing


def _inject_linkbase_refs(schema_content: str) -> str:
    if "linkbaseRef" in schema_content:
        return schema_content

    uses_xs_prefix = "<xs:schema" in schema_content or "xmlns:xs=" in schema_content
    uses_xsd_prefix = "<xsd:schema" in schema_content

    if "<annotation" in schema_content or "<xs:annotation" in schema_content or "<xsd:annotation" in schema_content:
        has_link_ns = 'xmlns:link="http://www.xbrl.org/2003/linkbase"' in schema_content
        has_xlink_ns = 'xmlns:xlink="http://www.w3.org/1999/xlink"' in schema_content

        if not has_link_ns:
            schema_content = re.sub(
                r'(<(?:xs:|xsd:)?schema\b)',
                r'\1 xmlns:link="http://www.xbrl.org/2003/linkbase"',
                schema_content,
                count=1,
            )
        if not has_xlink_ns:
            schema_content = re.sub(
                r'(<(?:xs:|xsd:)?schema\b)',
                r'\1 xmlns:xlink="http://www.w3.org/1999/xlink"',
                schema_content,
                count=1,
            )

        appinfo_prefix = ""
        if uses_xs_prefix:
            appinfo_prefix = "xs:"
        elif uses_xsd_prefix:
            appinfo_prefix = "xsd:"

        for pattern in [
            r"(</(?:xs:|xsd:)?annotation>)",
            r"(<(?:xs:|xsd:)?annotation/>)",
        ]:
            m = re.search(pattern, schema_content)
            if m:
                lb_ref_block = f"""<{appinfo_prefix}appinfo xmlns:link="http://www.xbrl.org/2003/linkbase" xmlns:xlink="http://www.w3.org/1999/xlink">
    <link:linkbaseRef xlink:href="presentation.xml" xlink:role="http://www.xbrl.org/2003/role/presentationLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
    <link:linkbaseRef xlink:href="calculation.xml" xlink:role="http://www.xbrl.org/2003/role/calculationLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
    <link:linkbaseRef xlink:href="definition.xml" xlink:role="http://www.xbrl.org/2003/role/definitionLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
    <link:linkbaseRef xlink:href="label.xml" xlink:role="http://www.xbrl.org/2003/role/labelLinkbaseRef" xlink:type="simple" xlink:arcrole="http://www.w3.org/1999/xlink/properties/linkbase"/>
  </{appinfo_prefix}appinfo>"""
                insert_pos = m.start()
                schema_content = schema_content[:insert_pos] + lb_ref_block + schema_content[insert_pos:]
                return schema_content

    has_link_ns = 'xmlns:link="http://www.xbrl.org/2003/linkbase"' in schema_content
    has_xlink_ns = 'xmlns:xlink="http://www.w3.org/1999/xlink"' in schema_content

    if not has_link_ns:
        schema_content = re.sub(
            r'(<(?:xs:|xsd:)?schema\b)',
            r'\1 xmlns:link="http://www.xbrl.org/2003/linkbase"',
            schema_content,
            count=1,
        )
    if not has_xlink_ns:
        schema_content = re.sub(
            r'(<(?:xs:|xsd:)?schema\b)',
            r'\1 xmlns:xlink="http://www.w3.org/1999/xlink"',
            schema_content,
            count=1,
        )

    if uses_xs_prefix:
        close_tag = "</xs:schema>"
    elif uses_xsd_prefix:
        close_tag = "</xsd:schema>"
    else:
        close_tag = "</schema>"

    refs = LINKBASE_REFS_TEMPLATE
    schema_content = schema_content.replace(close_tag, refs + "\n" + close_tag)
    return schema_content


def _is_valid_xml_content(content: str) -> bool:
    stripped = content.strip()
    if not stripped or stripped == "None":
        return False
    if not stripped.startswith("<?xml") and not stripped.startswith("<"):
        return False
    return True


def _generate_minimal_schema(inst: FinMRInstance) -> str:
    instance_doc = inst.sections.get("Instance document", "")
    ns_map = {}
    for m in re.finditer(r'xmlns:?([\w-]*)="([^"]+)"', instance_doc):
        prefix = m.group(1)
        uri = m.group(2)
        if prefix and uri and "xbrl.org" not in uri and "w3.org" not in uri:
            ns_map[prefix] = uri

    target_ns = ""
    for prefix, uri in ns_map.items():
        if prefix not in ("us-gaap", "dei", "srt", "iso4217", "link", "xbrli", "xsi", "country"):
            target_ns = uri
            break

    if not target_ns:
        target_ns = "http://example.com/schema"

    imports = []
    for prefix, uri in ns_map.items():
        if prefix == "us-gaap":
            year = re.search(r"/(\d{4})", uri)
            yr = year.group(1) if year else "2022"
            imports.append(
                f'  <import namespace="{uri}" schemaLocation="https://xbrl.fasb.org/us-gaap/{yr}/elts/us-gaap-{yr}-01-31.xsd"/>'
            )
        elif prefix == "dei":
            year = re.search(r"/(\d{4})", uri)
            yr = year.group(1) if year else "2022"
            imports.append(
                f'  <import namespace="{uri}" schemaLocation="https://xbrl.sec.gov/dei/{yr}/dei-{yr}.xsd"/>'
            )
        elif prefix == "srt":
            year = re.search(r"/(\d{4})", uri)
            yr = year.group(1) if year else "2022"
            imports.append(
                f'  <import namespace="{uri}" schemaLocation="https://xbrl.fasb.org/srt/{yr}/elts/srt-{yr}-01-31.xsd"/>'
            )

    imports_str = "\n".join(imports)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<schema xmlns="http://www.w3.org/2001/XMLSchema"
        xmlns:xlink="http://www.w3.org/1999/xlink"
        xmlns:link="http://www.xbrl.org/2003/linkbase"
        xmlns:xbrli="http://www.xbrl.org/2003/instance"
        targetNamespace="{target_ns}"
        elementFormDefault="qualified">
  <import namespace="http://www.xbrl.org/2003/instance" schemaLocation="http://www.xbrl.org/2003/xbrl-instance-2003-12-31.xsd"/>
{imports_str}
{LINKBASE_REFS_TEMPLATE}
</schema>"""


def reconstruct_xbrl_package(inst: FinMRInstance, out_dir: str, skip_href_rewrite: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    orig_schema = _detect_original_schema_name(inst)
    written_files = {}
    errors = []

    for section_name, filename in SECTION_FILE_MAP.items():
        content = inst.sections.get(section_name, "")

        if not _is_valid_xml_content(content):
            if section_name == "Instance document":
                errors.append(f"missing_section:{section_name}")
                continue
            elif section_name == "Schema document":
                content = _generate_minimal_schema(inst)
            else:
                continue

        content = _sanitize_xml(content)
        content = _repair_truncated_xml(content, filename)

        if not skip_href_rewrite:
            if orig_schema and orig_schema != "schema.xsd":
                content = content.replace(orig_schema, "schema.xsd")

            if filename == "schema.xsd" and "linkbaseRef" not in content:
                content = _inject_linkbase_refs(content)

        filepath = os.path.join(out_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        written_files[section_name] = filepath

    return {
        "out_dir": out_dir,
        "instance_path": os.path.join(out_dir, "instance.xml"),
        "written_files": written_files,
        "orig_schema_name": orig_schema,
        "errors": errors,
    }
