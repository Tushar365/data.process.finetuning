# data.process.finetuning

This repository contains scripts and documentation for processing raw disaster assessment data into a multi-turn conversational JSON format suitable for AI model fine-tuning. The workflow focuses on comparing pre- and post-disaster satellite imagery to generate damage assessments.

## Data Processing Workflow

The following diagram illustrates how the raw data (images and labels) are structured, processed by the `process.py` script, and transformed into the final conversational JSON output.

### Directory Structure & Processing Flow
# Dataset Preparation and Processing

This project processes raw disaster data into structured multi-turn conversations for analysis and training purposes.

## Workflow Overview

### Raw Data → Script Processing → Multi-turn Conversations

train/
├── images/
│   ├── area1_pre_disaster.png  ──┐
│   └── area1_post_disaster.png ──┼─→ Script reads actual images
└── labels/                       │
    ├── area1_pre_disaster.json ──┼─→ Script parses JSON building data
    └── area1_post_disaster.json ─┘
                                   │
                                   ▼
                    Creates conversation with:
                    • Real image paths
                    • Actual damage analysis from JSON
                    • Building count statistics
                    • Damage percentages# data.process.finetuning


## Data Preparation

To prepare the data for processing, ensure your `train` folder is organized according to the structure described above:

-   `train/images/`: Contains both pre-disaster (`*_pre_disaster.png`) and post-disaster (`*_post_disaster.png`) satellite images.
-   `train/labels/`: Contains corresponding pre-disaster (`*_pre_disaster.json`) and post-disaster (`*_post_disaster.json`) JSON files. These JSONs are expected to contain annotations or labels relevant to the damage assessment (e.g., building boundaries, damage classifications).

### Processing Steps

1.  **Load Data:** Ensure the `train` folder is correctly populated with the `images` and `labels` subfolders.
2.  **Run Processing Script:** Execute the Python script from the root of your project:
    ```bash
    python process.py
    ```
3.  **Output:** The script will generate the processed data in a file named `train_conversation.json` within the project root.

---

## Describing the Disaster Assessment Workflow JSON Structure

The `train_conversation.json` file adheres to a specific JSON structure designed to facilitate a two-stage disaster impact assessment workflow. This structure represents a sequence of conversational exchanges between a "user" (providing images and requests) and an "assistant" (providing acknowledgments and assessment reports).

The overall JSON is an array of individual conversation segments. Each segment contains a `"messages"` array that encapsulates a single turn (or a set of related turns) in the assessment process.

### Stage 1: Pre-Disaster Image Submission

This initial stage captures the user providing the baseline image.

*   **User Input (Pre-Image):**
    *   The first message object in a segment (`"role": "user"`) provides the initial context.
    *   It contains a `"content"` array with two elements:
        *   An object with `"type": "image_path"`, pointing to the **pre-disaster satellite image** (e.g., `images/midwest-flooding_00000001_pre_disaster.png`).
        *   An object with `"type": "text"`, typically instructing the assistant to acknowledge the image and await further input (e.g., "Here's a pre-disaster satellite image. Please wait for the post-disaster image to make the comparison.").

*   **Assistant Acknowledgment:**
    *   The subsequent message object (`"role": "assistant"`) within the same segment confirms receipt of the pre-disaster image.
    *   It contains a `"content"` array with a `"type": "text"` element, indicating readiness for the next stage (e.g., "Pre-disaster satellite image noted. Awaiting the post-disaster image for damage analysis.").

### Stage 2: Post-Disaster Image Submission & Damage Description Request

This second stage involves providing the disaster image and requesting an analysis.

*   **User Input (Post-Image & Request):**
    *   A separate conversation segment (another object in the top-level array) initiates the comparison phase.
    *   Its `"role": "user"` message again has a `"content"` array:
        *   An object with `"type": "image_path"`, providing the **post-disaster satellite image** (e.g., `images/midwest-flooding_00000001_post_disaster.png`).
        *   An object with `"type": "text"`, explicitly requesting a comparison with the previous image and an evaluation of damage (e.g., "This is the post-disaster satellite image. Compare with the previous pre-disaster image and evaluate building damage.").

*   **Assistant Output (Damage Description):**
    *   The final message object (`"role": "assistant"`) in this segment delivers the **damage assessment description**.
    *   It contains a `"content"` array with a single `"type": "text"` element. This text is the comprehensive report detailing the disaster type, building damage summary (e.g., "No damage", "Unclassified damage"), impact severity, key observations, and recommended response actions.

**In essence, this JSON structure models a sequential interaction where the user first provides initial data (pre-image), then follows up with the crucial comparison data (post-image), and receives a structured analytical "description" as the assistant's output.**
