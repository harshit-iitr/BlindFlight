# Blind Flight: Autonomous Logistics Rescue
### Synapse Drive 2024 Solution

**Author:** Harshit Agrawal (Enrollment: 25125017)  
**Branch:** Data Science & AI (IIT Roorkee)

## 🎯 The Mission
Following an EMP blast, Stark Industries' logistics network is blind. [cite_start]This project rebuilds the perception and planning stack to navigate autonomous drones through **Labs, Forests, and Deserts** using only imagery[cite: 5, 7, 12].
## 🧠 The Solution: Hybrid "Gap-Filling" Ensemble
We achieved **81% accuracy** not by relying on a single model, but by engineering a **Two-Stage Perception Pipeline**. This approach mitigates the specific weaknesses of grid-based vision.

### 1. The Backup: Ungridded Global Model
* **Role:** High Recall / Safety Net.
* **Function:** A lightweight model that looks at the map globally without slicing.
* **Why it matters:** Grid slicing techniques sometimes fail on blurry or highly corrupted maps, resulting in broken paths. This model ensures that **most of the maps have a predicted path, though compromising accuracy**, preventing invalid submissions.

### 2. The Expert: Gridded ResNet-18
* **Role:** High Precision / Fine-Grained Classification.
* **Function:** Slices the $20 \times 20$ map into 400 individual tiles to classify terrain type (Road, Wall, Hazard, Start, Goal).
* **Key Innovations:**
    * **"Diamond Cutter" Augmentation:** We trained on images with randomly "cut" corners to force the model to recognize objects by internal texture rather than shape.
    * **Velocity-Aware A:** The pathfinding algorithm integrates the hidden **Velocity Boost Field** ($Cost = Base - Boost$), allowing drones to ride the wind for optimal energy efficiency.

### 🤝 The Merger
The final submission combines these two models. The Gridded model provides precise, optimal paths for the majority of maps. Wherever the Gridded model is uncertain or fails to find a valid route, the **Ungridded model fills the gap**, raising the overall system accuracy to **81.82%**.

## 📂 Repository Structure

```text
BlindFlight-Solution/
├── assets/                  # Diagrams, icons, and visual assets for documentation
├── Models/                  # Trained Model weights (.pth)
├── Notebooks/               # Jupyter notebooks for EDA, training, and analysis
├── Sample outputs/          # Visualizations of generated paths and debug images
├── Submissions/             # Generated CSV submission files (including Final & Backup)
├── SubmissionScript/        # The core inference logic (submission_generator.py)
├── .gitignore               
└── README.md                # Project documentation
