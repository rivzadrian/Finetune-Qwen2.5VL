from datasets import load_dataset
from pathlib import Path

# ✅ Import your message builder function from your data collator
from src.vlm.data.data_collactor import create_message_template

def main():
    # -------------------------------
    # 1. Load your dataset (adjust if needed)
    # -------------------------------
    dataset = load_dataset("rvzadrian/skin_flap_cropped_data_v2_fold_0", split="train")

    # -------------------------------
    # 2. Print out prompts for first few samples
    # -------------------------------
    for i in range(3):  # adjust number of samples to debug
        sample = dataset[i]

        # 🛠 Build the prompt using your project function
        message = create_message_template(sample)

        # 📢 Print to check
        print(f"\n🔎 Sample {i} Prompt:\n{message}\n{'-'*80}")

if __name__ == "__main__":
    main()
