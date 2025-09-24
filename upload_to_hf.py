import os
import json
from datasets import Dataset, DatasetDict, Image
import pandas as pd
from huggingface_hub import login

# ---- CONFIGURATION ----
DATASET_JSONL = "training_dataset.jsonl"  # Your processed dataset
HF_DATASET_NAME = "Tushar365/disaster-assessment-qwen2vl"  # Change this!

def load_processed_dataset():
    '''Load your processed training dataset.'''

    if not os.path.exists(DATASET_JSONL):
        print(f"❌ Dataset file not found: {DATASET_JSONL}")
        print("💡 Run dataset_processor_enhanced.py first!")
        return None

    print(f"📂 Loading dataset from: {DATASET_JSONL}")

    # Load JSONL file
    data_rows = []
    with open(DATASET_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_rows.append(json.loads(line))

    print(f"✅ Loaded {len(data_rows)} training samples")

    # Verify image files exist
    missing_images = []
    valid_rows = []

    for row in data_rows:
        images = row.get("images", [])
        if len(images) != 2:
            print(f"⚠️ Skipping row - expected 2 images, got {len(images)}")
            continue

        # Check if both images exist
        pre_img, post_img = images[0], images[1]
        if not os.path.exists(pre_img):
            missing_images.append(pre_img)
        if not os.path.exists(post_img):
            missing_images.append(post_img)

        if os.path.exists(pre_img) and os.path.exists(post_img):
            valid_rows.append(row)

    if missing_images:
        print(f"⚠️ Found {len(missing_images)} missing images:")
        for img in missing_images[:5]:  # Show first 5
            print(f"   {img}")
        if len(missing_images) > 5:
            print(f"   ... and {len(missing_images)-5} more")

    print(f"✅ Valid samples with existing images: {len(valid_rows)}")
    return valid_rows

def create_hf_dataset(data_rows):
    '''Create HF Dataset with proper image handling.'''

    print("🔄 Creating Hugging Face Dataset structure...")

    # Prepare data for HF Dataset
    processed_data = []

    for i, row in enumerate(data_rows):
        pre_img_path, post_img_path = row["images"]

        # Create entry with image objects
        entry = {
            "id": f"sample_{i+1:04d}",
            "pre_image": pre_img_path,
            "post_image": post_img_path, 
            "instruction": row["instruction"],
            "output": row["output"],
            "images": row["images"]  # Keep original paths for reference
        }
        processed_data.append(entry)

    # Convert to pandas DataFrame
    df = pd.DataFrame(processed_data)

    # Create Dataset
    dataset = Dataset.from_pandas(df)

    # Add image loading - this automatically uploads images to HF Hub
    def load_image_files(example):
        '''Load actual PIL Image objects for HF storage.'''
        try:
            from PIL import Image as PILImage

            pre_img = PILImage.open(example["pre_image"]).convert("RGB")
            post_img = PILImage.open(example["post_image"]).convert("RGB")

            # Store as image objects (HF will handle uploading)
            example["pre_image"] = pre_img
            example["post_image"] = post_img

            return example
        except Exception as e:
            print(f"⚠️ Error loading images for {example.get('id', 'unknown')}: {e}")
            return example

    print("🖼️ Processing images for upload...")
    dataset = dataset.map(load_image_files)

    # Create train/validation split (80/20) - FIXED VERSION
    if len(dataset) > 1:
        # Use seed for reproducibility instead of random_state
        dataset = dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
        print(f"📊 Created splits - Train: {len(dataset['train'])}, Validation: {len(dataset['test'])}")
    else:
        dataset = DatasetDict({"train": dataset})
        print(f"📊 Single split - Train: {len(dataset['train'])}")

    return dataset

def push_to_hugging_face(dataset):
    '''Upload dataset to Hugging Face Hub.'''

    if "your-username" in HF_DATASET_NAME:
        print("❌ Please change 'your-username' to your actual HF username in the script!")
        print(f"   Current: {HF_DATASET_NAME}")
        print(f"   Example: john-doe/disaster-assessment-qwen2vl")
        return False

    print(f"🚀 Uploading dataset to: {HF_DATASET_NAME}")

    try:
        # Push to hub - images are automatically uploaded
        dataset.push_to_hub(
            HF_DATASET_NAME,
            private=False,  # Set to True for private dataset
            commit_message="Upload disaster assessment dataset for Qwen2-VL fine-tuning",
            max_shard_size="500MB"  # Optimize for large image files
        )

        print(f"\n✅ SUCCESS! Dataset uploaded to:")
        print(f"🔗 https://huggingface.co/datasets/{HF_DATASET_NAME}")

        print(f"\n🎯 Ready for Oumi + Qwen2-VL fine-tuning!")
        print(f"\n📝 Use in your fine-tuning script:")
        print(f'from datasets import load_dataset')
        print(f'dataset = load_dataset("{HF_DATASET_NAME}")')

        return True

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(f"\n💡 Common fixes:")
        print(f"   • Login: huggingface-cli login")
        print(f"   • Check dataset name format: username/dataset-name")
        print(f"   • Verify internet connection")
        print(f"   • Try private=True if having permission issues")
        return False

def main():
    '''Main function to upload processed dataset to HF Hub.'''

    print("🚀 HUGGING FACE DATASET UPLOAD")
    print("=" * 40)

    # Step 1: Load processed dataset
    data_rows = load_processed_dataset()
    if not data_rows:
        return

    # Step 2: Create HF Dataset with images
    dataset = create_hf_dataset(data_rows)
    if not dataset:
        return

    # Step 3: Upload to HF Hub
    success = push_to_hugging_face(dataset)

    if success:
        print(f"\n🎉 COMPLETE! Your disaster assessment dataset is now live!")
        print(f"📚 Perfect for Qwen2-VL fine-tuning with Oumi")
        print(f"🔄 Images and training data automatically handled by HF Hub")
    else:
        print(f"\n❌ Upload failed. Check the error messages above.")

if __name__ == "__main__":
    # Check if user is logged in
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"✅ Logged in as: {user['name']}")
    except:
        print("❌ Not logged in to Hugging Face!")
        print("💡 Run: huggingface-cli login")
        exit(1)

    main()
