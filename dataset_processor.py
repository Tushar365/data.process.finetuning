import os
import json
from glob import glob
from collections import Counter
import random

# ---- CONFIGURATION ----
DATASET_DIR = "train"  # Your dataset folder containing images/ and labels/
OUTPUT_JSONL = "training_dataset.jsonl"

def parse_disaster_json(json_path):
    '''Parse pre/post disaster JSON and extract building information.'''
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        building_map = {}
        features = data.get("features", {})
        buildings = features.get("lng_lat", [])

        for building in buildings:
            props = building.get("properties", {})
            uid = props.get("uid")
            if uid:
                # Post-disaster JSONs have subtype, pre-disaster don't
                subtype = props.get("subtype", "building")
                building_map[uid] = subtype
        return building_map
    except Exception as e:
        print(f"Error parsing {json_path}: {e}")
        return {}

def generate_detailed_assessment(pre_json_path, post_json_path, base_filename):
    '''Generate detailed, natural language damage assessment for LLM training.'''
    pre_buildings = parse_disaster_json(pre_json_path)
    post_buildings = parse_disaster_json(post_json_path)

    damage_counts = Counter()
    total_buildings = len(post_buildings) if post_buildings else len(pre_buildings)
    changed_buildings = []

    # Analyze each building
    for uid in set(list(pre_buildings.keys()) + list(post_buildings.keys())):
        if uid in pre_buildings and uid in post_buildings:
            damage_state = post_buildings[uid]
            damage_counts[damage_state] += 1
            if damage_state != "building":
                changed_buildings.append((uid, damage_state))
        elif uid in pre_buildings and uid not in post_buildings:
            damage_counts["disappeared"] += 1
            changed_buildings.append((uid, "disappeared"))
        elif uid not in pre_buildings and uid in post_buildings:
            damage_counts["new-building"] += 1
            changed_buildings.append((uid, "new-building"))

    if not damage_counts:
        return "No buildings were detected in either the pre-disaster or post-disaster satellite images."

    # Generate comprehensive natural language report
    report_parts = []

    # 1. Overview statement
    disaster_type = "flooding" if "flooding" in base_filename else "disaster"
    location = base_filename.replace("midwest-flooding_", "").replace("_", " ") if "midwest" in base_filename else "affected area"

    report_parts.append(f"**Disaster Impact Assessment Report**\n")
    report_parts.append(f"Disaster Type: {disaster_type.title()}\n")

    # 2. Summary statistics
    report_parts.append(f"**Building Analysis Summary:**")
    report_parts.append(f"Total buildings analyzed: {total_buildings}")

    if total_buildings > 0:
        # Calculate percentages
        no_damage = damage_counts.get("no-damage", 0)
        minor_damage = damage_counts.get("minor-damage", 0)
        major_damage = damage_counts.get("major-damage", 0)
        destroyed = damage_counts.get("destroyed", 0)
        unclassified = damage_counts.get("un-classified", 0)
        disappeared = damage_counts.get("disappeared", 0)

        if no_damage > 0:
            pct = round((no_damage / total_buildings) * 100, 1)
            report_parts.append(f"• No damage: {no_damage} buildings ({pct}%) - These structures show no visible signs of damage")

        if minor_damage > 0:
            pct = round((minor_damage / total_buildings) * 100, 1)
            report_parts.append(f"• Minor damage: {minor_damage} buildings ({pct}%) - Slight structural damage, likely repairable")

        if major_damage > 0:
            pct = round((major_damage / total_buildings) * 100, 1)
            report_parts.append(f"• Major damage: {major_damage} buildings ({pct}%) - Significant structural damage requiring extensive repairs")

        if destroyed > 0:
            pct = round((destroyed / total_buildings) * 100, 1)
            report_parts.append(f"• Completely destroyed: {destroyed} buildings ({pct}%) - Structures are completely demolished or collapsed")

        if unclassified > 0:
            pct = round((unclassified / total_buildings) * 100, 1)
            report_parts.append(f"• Unclassified damage: {unclassified} buildings ({pct}%) - Damage extent unclear from satellite imagery")

        if disappeared > 0:
            pct = round((disappeared / total_buildings) * 100, 1)
            report_parts.append(f"• Missing structures: {disappeared} buildings ({pct}%) - Buildings present before disaster but not visible after")

    # 3. Impact severity assessment
    total_affected = minor_damage + major_damage + destroyed + disappeared
    if total_buildings > 0:
        affected_percentage = round((total_affected / total_buildings) * 100, 1)

        report_parts.append(f"\n**Impact Severity:**")
        if affected_percentage == 0:
            severity = "Minimal Impact"
            description = "The disaster appears to have caused minimal structural damage to buildings in this area."
        elif affected_percentage < 25:
            severity = "Low Impact"
            description = f"Limited damage observed with {affected_percentage}% of buildings affected. Most structures remain intact."
        elif affected_percentage < 50:
            severity = "Moderate Impact"
            description = f"Moderate damage levels with {affected_percentage}% of buildings affected. Significant recovery efforts will be needed."
        elif affected_percentage < 75:
            severity = "High Impact"
            description = f"Severe damage observed with {affected_percentage}% of buildings affected. Extensive reconstruction required."
        else:
            severity = "Catastrophic Impact"
            description = f"Catastrophic damage with {affected_percentage}% of buildings affected. Area requires major rebuilding efforts."

        report_parts.append(f"Classification: {severity}")
        report_parts.append(f"Assessment: {description}")

    # 4. Specific observations
    if changed_buildings:
        report_parts.append(f"\n**Key Observations:**")
        if destroyed > 0:
            report_parts.append(f"• {destroyed} building(s) suffered complete destruction, indicating severe {disaster_type} impact")
        if major_damage > 0:
            report_parts.append(f"• {major_damage} building(s) show major structural damage, likely requiring demolition and rebuilding")
        if disappeared > 0:
            report_parts.append(f"• {disappeared} building(s) are no longer visible, possibly swept away or buried by {disaster_type}")
        if no_damage > 0 and total_affected > 0:
            report_parts.append(f"• {no_damage} building(s) remained undamaged, suggesting uneven {disaster_type} impact across the area")

    # 5. Recommended actions
    report_parts.append(f"\n**Recommended Response Actions:**")
    if destroyed > 0 or major_damage > 0:
        report_parts.append(f"• Immediate safety assessment and debris clearance required")
        report_parts.append(f"• Emergency shelter needed for displaced residents")
    if minor_damage > 0:
        report_parts.append(f"• Structural engineering assessment for damaged buildings")
    if total_affected > 0:
        report_parts.append(f"• Coordinate with emergency services for rescue and relief operations")
    else:
        report_parts.append(f"• Monitoring recommended, but immediate emergency response may not be critical")

    return "\n".join(report_parts)

def process_dataset():
    '''Main function to process the entire dataset and create enhanced training JSONL.'''
    print("🚀 Starting enhanced dataset processing for LLM training...")

    # Validate folder structure
    images_dir = os.path.join(DATASET_DIR, "images")
    labels_dir = os.path.join(DATASET_DIR, "labels")

    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return

    if not os.path.exists(labels_dir):
        print(f"❌ Labels directory not found: {labels_dir}")
        return

    print(f"✅ Found dataset structure:")
    print(f"   Images: {images_dir}")
    print(f"   Labels: {labels_dir}")

    # Find all pre-disaster JSON files
    pre_json_pattern = os.path.join(labels_dir, "*_pre_disaster.json")
    pre_json_files = glob(pre_json_pattern)

    print(f"\n📁 Found {len(pre_json_files)} pre-disaster JSON files")

    training_entries = []
    processed_count = 0
    skipped_count = 0

    # Enhanced instructions for better LLM training
    instructions = [
        "Analyze these pre-disaster and post-disaster satellite images and provide a comprehensive building damage assessment report.",
        "Compare the before and after satellite imagery to evaluate disaster impact on buildings and infrastructure.",
        "Examine these satellite images taken before and after a disaster event and generate a detailed damage assessment.",
        "Review these pre-disaster and post-disaster satellite images and create a thorough building damage analysis report.",
        "Assess the disaster impact by comparing these before and after satellite images, focusing on building damage patterns."
    ]

    for pre_json_path in pre_json_files:
        # Extract base filename
        base_filename = os.path.basename(pre_json_path).replace("_pre_disaster.json", "")

        # Use correct naming pattern  
        post_json_path = os.path.join(labels_dir, f"{base_filename}_post_disaster.json")
        pre_img_path = os.path.join(images_dir, f"{base_filename}_pre_disaster.png")
        post_img_path = os.path.join(images_dir, f"{base_filename}_post_disaster.png")

        # Check if all required files exist
        missing_files = []
        if not os.path.exists(post_json_path):
            missing_files.append("post JSON")
        if not os.path.exists(pre_img_path):
            missing_files.append("pre image")
        if not os.path.exists(post_img_path):
            missing_files.append("post image")

        if missing_files:
            print(f"⚠️  Skipping {base_filename}: missing {', '.join(missing_files)}")
            skipped_count += 1
            continue

        # Process this pair
        print(f"📊 Processing: {base_filename}")

        # Generate detailed damage assessment
        detailed_assessment = generate_detailed_assessment(pre_json_path, post_json_path, base_filename)

        # Create enhanced training entry
        entry = {
            "images": [pre_img_path, post_img_path],
            "instruction": random.choice(instructions),  # Vary instructions for better training
            "output": detailed_assessment
        }
        training_entries.append(entry)

        processed_count += 1

        # Show example output (truncated)
        if processed_count == 1:
            print(f"   📋 Sample detailed output:")
            lines = detailed_assessment.split("\n")[:8]  # First 8 lines
            for line in lines:
                print(f"      {line}")
            print(f"      ... (full report saved to dataset)")

    # Save to JSONL file
    print(f"\n💾 Saving {len(training_entries)} enhanced entries to {OUTPUT_JSONL}")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for entry in training_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Enhanced dataset processing complete!")
    print(f"   📈 Processed: {processed_count} pairs")
    print(f"   ⏭️  Skipped: {skipped_count} pairs")
    print(f"   📄 Output file: {OUTPUT_JSONL}")
    print(f"\n🎯 Ready for high-quality Qwen2-VL fine-tuning!")
    print(f"\n💡 Each entry now contains:")
    print(f"   ✅ Detailed damage assessment reports")
    print(f"   ✅ Impact severity classifications")
    print(f"   ✅ Specific observations and recommendations")
    print(f"   ✅ Natural language perfect for LLM training")

if __name__ == "__main__":
    process_dataset()
