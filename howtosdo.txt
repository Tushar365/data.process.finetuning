# Disaster Assessment Data Processing and JSON Structure

This document outlines the process for preparing disaster assessment data and describes the resulting JSON conversational structure used for AI model training or interaction.

## Data Preparation

To prepare the data for processing, ensure your `train` folder is organized as follows:

-   `train/images/`: Contains both pre-disaster and post-disaster satellite images.
-   `train/labels/`: Contains corresponding pre-disaster and post-disaster JSON files (likely containing annotations or labels for the images).

### Processing Steps

1.  **Load Data:** Ensure the `train` folder is correctly populated with the `images` and `labels` subfolders.
2.  **Run Processing Script:** Execute the Python script:
    ```bash
    python process.py
    ```
3.  **Output:** The script will generate the processed data in a file named `train_conversation.json`.

---

## Describing the Disaster Assessment Workflow JSON Structure

The `train_conversation.json` file adheres to a specific JSON structure designed to facilitate a two-stage disaster impact assessment workflow. This structure represents a sequence of conversational exchanges between a "user" (providing images and requests) and an "assistant" (providing acknowledgments and assessment reports).

The overall JSON is an array of individual conversation segments. Each segment contains a `"messages"` array that encapsulates a single turn (or a set of related turns) in the assessment process.

### Stage 1: Pre-Disaster Image Submission

This initial stage captures the user providing the baseline image.

*   **User Input (Pre-Image):**
    *   The first message object in a segment (`"role": "user"`) provides the initial context.
    *   It contains a `"content"` array with two elements:
        *   An object with `"type": "image_path"`, pointing to the **pre-disaster satellite image** (e.g., `train/images/midwest-flooding_00000001_pre_disaster.png`).
        *   An object with `"type": "text"`, typically instructing the assistant to acknowledge the image and await further input (e.g., "Here's a pre-disaster satellite image. Please wait for the post-disaster image to make the comparison.").

*   **Assistant Acknowledgment:**
    *   The subsequent message object (`"role": "assistant"`) within the same segment confirms receipt of the pre-disaster image.
    *   It contains a `"content"` array with a `"type": "text"` element, indicating readiness for the next stage (e.g., "Pre-disaster satellite image noted. Awaiting the post-disaster image for damage analysis.").

### Stage 2: Post-Disaster Image Submission & Damage Description Request

This second stage involves providing the disaster image and requesting an analysis.

*   **User Input (Post-Image & Request):**
    *   A separate conversation segment (another object in the top-level array) initiates the comparison phase.
    *   Its `"role": "user"` message again has a `"content"` array:
        *   An object with `"type": "image_path"`, providing the **post-disaster satellite image** (e.g., `train/images/midwest-flooding_00000001_post_disaster.png`).
        *   An object with `"type": "text"`, explicitly requesting a comparison with the previous image and an evaluation of damage (e.g., "This is the post-disaster satellite image. Compare with the previous pre-disaster image and evaluate building damage.").

*   **Assistant Output (Damage Description):**
    *   The final message object (`"role": "assistant"`) in this segment delivers the **damage assessment description**.
    *   It contains a `"content"` array with a single `"type": "text"` element. This text is the comprehensive report detailing the disaster type, building damage summary (e.g., "No damage", "Unclassified damage"), impact severity, key observations, and recommended response actions.

**In essence, this JSON structure models a sequential interaction where the user first provides initial data (pre-image), then follows up with the crucial comparison data (post-image), and receives a structured analytical "description" as the assistant's output.**