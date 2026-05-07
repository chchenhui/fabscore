"""Sanity check for entity anonymization pipeline. Validates the first 100
instances: checks replacement correctness, collision-free IDs, no residual
entity names, and prints diagnostic statistics."""

import json
import os
import re
import sys

PROJ_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJ_DIR)

from eacp.data.confiqa_loader import extract_path_entities, load_confiqa
from eacp.data.entity_anonymizer import anonymize_instance
from eacp.data.entity_inventory import build_entity_inventory
from eacp.prompts.condition_c import build_chat_messages_c


def run_sanity_check(n: int = 100):
    data = load_confiqa("MC")[:n]
    print(f"Running anonymization sanity check on {len(data)} instances...\n")

    total_entities = 0
    total_replacements = 0
    all_entities_replaced_count = 0
    cf_entities_all_replaced_count = 0
    collision_issues = []
    residual_name_issues = []
    cf_answer_issues = []
    orig_answer_issues = []
    sample_prompts = []

    for idx, inst in enumerate(data):
        result = anonymize_instance(inst)
        entity_map = result["entity_map"]
        anon_ctx = result["anon_context"]
        anon_q = result["anon_question"]
        stats = result["replacement_stats"]

        n_entities = len(entity_map)
        n_replacements = sum(stats.values())
        total_entities += n_entities
        total_replacements += n_replacements

        ent_ids = [eid for _, (eid, _) in entity_map.items()]
        if len(ent_ids) != len(set(ent_ids)):
            collision_issues.append({"instance_id": idx, "ids": ent_ids})

        all_replaced = all(stats.get(eid, 0) > 0 for _, (eid, _) in entity_map.items())
        if all_replaced:
            all_entities_replaced_count += 1

        from eacp.data.entity_inventory import _parse_path_labeled
        cf_labels = set()
        for triple in _parse_path_labeled(inst.get("cf_path_labeled", "[]")):
            cf_labels.add(triple[0])
            cf_labels.add(triple[2])
        cf_answer_label = inst.get("cf_answer", "")
        if cf_answer_label:
            cf_labels.add(cf_answer_label)
        cf_all_replaced = all(
            stats.get(entity_map[lbl][0], 0) > 0
            for lbl in cf_labels if lbl in entity_map
        )
        if cf_all_replaced:
            cf_entities_all_replaced_count += 1

        cf_answer = inst.get("cf_answer", "")
        orig_answer = inst.get("orig_answer", "")
        if cf_answer and cf_answer in entity_map:
            eid = entity_map[cf_answer][0]
            if stats.get(eid, 0) == 0:
                cf_answer_issues.append({
                    "instance_id": idx,
                    "answer": cf_answer,
                    "ent_id": eid,
                })
        if orig_answer and orig_answer in entity_map:
            eid = entity_map[orig_answer][0]
            if stats.get(eid, 0) == 0:
                orig_answer_issues.append({
                    "instance_id": idx,
                    "answer": orig_answer,
                    "ent_id": eid,
                })

        entities_with_aliases = extract_path_entities(inst)
        combined_text = anon_ctx + " " + anon_q
        for label, aliases in entities_with_aliases.items():
            for name in {label} | aliases:
                if len(name) < 2:
                    continue
                pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
                if pattern.search(combined_text):
                    residual_name_issues.append({
                        "instance_id": idx,
                        "residual_name": name,
                        "entity_label": label,
                    })

        if idx < 3:
            msgs, emap, _ = build_chat_messages_c(inst)
            sample_prompts.append({
                "instance_id": idx,
                "user_prompt": msgs[1]["content"],
                "entity_map": {k: v[0] for k, v in emap.items()},
            })

    avg_entities = total_entities / len(data)
    avg_replacements = total_replacements / len(data)
    frac_all_replaced = all_entities_replaced_count / len(data)
    frac_cf_all_replaced = cf_entities_all_replaced_count / len(data)

    report = {
        "n_instances": len(data),
        "avg_entities_per_instance": round(avg_entities, 2),
        "avg_replacements_per_instance": round(avg_replacements, 2),
        "fraction_all_entities_replaced": round(frac_all_replaced, 4),
        "fraction_cf_entities_all_replaced": round(frac_cf_all_replaced, 4),
        "n_collision_issues": len(collision_issues),
        "n_residual_name_issues": len(residual_name_issues),
        "n_cf_answer_replacement_issues": len(cf_answer_issues),
        "n_orig_answer_not_in_cf_context": len(orig_answer_issues),
        "collision_details": collision_issues[:5],
        "residual_name_details": residual_name_issues[:10],
        "cf_answer_issue_details": cf_answer_issues[:5],
        "orig_answer_not_in_cf_details": orig_answer_issues[:5],
        "sample_prompts": sample_prompts,
    }

    print("=== Anonymization Sanity Check Report ===")
    print(f"  Instances checked: {report['n_instances']}")
    print(f"  Avg entities/instance: {report['avg_entities_per_instance']}")
    print(f"  Avg replacements/instance: {report['avg_replacements_per_instance']}")
    print(f"  Fraction ALL entities replaced: {report['fraction_all_entities_replaced']}")
    print(f"  Fraction CF entities all replaced: {report['fraction_cf_entities_all_replaced']}")
    print(f"  Collision issues: {report['n_collision_issues']}")
    print(f"  Residual name issues: {report['n_residual_name_issues']}")
    print(f"  CF answer replacement issues: {report['n_cf_answer_replacement_issues']}")
    print(f"  Orig answer not in CF context (expected): {report['n_orig_answer_not_in_cf_context']}")

    if residual_name_issues:
        print("\n  Sample residual name issues:")
        for r in residual_name_issues[:5]:
            print(f"    Instance {r['instance_id']}: '{r['residual_name']}' ({r['entity_label']})")

    if sample_prompts:
        print("\n  Sample prompt (instance 0):")
        print(sample_prompts[0]["user_prompt"][:500])

    out_dir = os.path.join(PROJ_DIR, "eacp", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "anonymization_sanity_check.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    run_sanity_check()
