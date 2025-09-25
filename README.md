# Dataset Preparation and Processing

This project processes raw disaster data into structured multi-turn conversations for analysis and training purposes.

## Workflow Overview

### Raw Data → Script Processing → Multi-turn Conversations

The workflow begins with raw data consisting of images and JSON labels. The script processes this data to generate structured conversations containing:

- Real image paths
- Actual damage analysis from JSON
- Building count statistics
- Damage percentages

### Directory Structure
Your Raw Data → Script Processing → Multi-turn Conversations

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
