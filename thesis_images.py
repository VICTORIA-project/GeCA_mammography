import os
import random
import argparse
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import ast

def extract_label(label_value):
    if isinstance(label_value, str) and label_value.startswith("[") and label_value.endswith("]"):
        try:
            parsed = ast.literal_eval(label_value)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]
        except Exception:
            pass
    return label_value

def create_comparison_grid(real_csv, real_folder, synth_csv, synth_folder, output_path, label_column="breast_density"):
    real_df = pd.read_csv(real_csv)
    real_df[label_column] = real_df[label_column].apply(extract_label)

    synth_df = pd.read_csv(synth_csv)
    synth_df = synth_df.loc[:, ~synth_df.columns.str.contains('^Unnamed')]

    real_labels = real_df[label_column].unique().tolist()
    
    known_cols = {"client_id", "filename", "patient_name", "multi_class_label"}
    label_cols = [col for col in synth_df.columns if col not in known_cols and col in real_labels]

    if label_column not in real_df.columns:
        raise ValueError(f"{label_column} not found in real CSV.")

    if label_column == "breast_density":
        expected_labels = ['DENSITY A', 'DENSITY B', 'DENSITY C', 'DENSITY D']
        common_labels = [l for l in expected_labels if l in label_cols]
    else:
        common_labels = label_cols

    print(f"Detected one-hot label columns: {label_cols}")
    print(f"Common labels: {common_labels}")

    if len(common_labels) < 4:
        raise ValueError(f"Expected at least 4 labels to sample from, found: {common_labels}")

    # shuffle common labels
    shuffled_labels = common_labels.copy()
    random.shuffle(shuffled_labels)

    selected_labels = []
    max_columns = 4
    for label in shuffled_labels:
        matching_real = real_df[real_df[label_column] == label]
        synth_candidates = synth_df[synth_df[label] == 1]
        if matching_real.empty:
            print(f"No real images found for label {label}")
            continue
        if synth_candidates.empty:
            print(f"No synthetic images found for label {label}")
            continue
        selected_labels.append(label)
        if len(selected_labels) == max_columns:
            break

    if len(selected_labels) < max_columns:
        print(f"Warning: only found {len(selected_labels)} labels with both real and synthetic images.")

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    img_size = 128
    spacing = 4

    num_images = 4
    label_height = 20
    vertical_label_width = 60
    combined_width = vertical_label_width + img_size * num_images + spacing * (num_images - 1)
    combined_height = label_height + 2 * img_size + spacing + spacing  # label + two images + spacing between rows

    combined_image = Image.new("RGB", (combined_width, combined_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(combined_image)

    for i, label in enumerate(selected_labels):
        matching_real = real_df[real_df[label_column] == label]
        if matching_real.empty:
            print(f"No real images found for label {label}")
            continue
        real_sample = matching_real.sample(1).iloc[0]
        real_model = real_sample['model']

        synth_df['assigned_model'] = real_model 
        synth_candidates = synth_df[synth_df[label] == 1]
        if synth_candidates.empty:
            print(f"No synthetic images found for label {label}")
            continue
        synth_sample = synth_candidates.sample(1).iloc[0]

        real_img = Image.open(os.path.join(real_folder, real_sample['image_id']))
        synth_img = Image.open(os.path.join(synth_folder, synth_sample['filename']))

        x_offset = vertical_label_width + i * (img_size + spacing)

        # label
        label_text = label.replace("DENSITY ", "")  # Clean up if density
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]

        draw.text((x_offset + (img_size - text_width) / 2, 0), label_text, fill=(0, 0, 0), font=font)

        # real image
        combined_image.paste(real_img, (x_offset, label_height))

        # synthetic image
        combined_image.paste(synth_img, (x_offset, label_height + img_size + spacing))

    vertical_labels = ["Real", "Synthetic"]
    for idx, vlabel in enumerate(vertical_labels):
        bbox = draw.textbbox((0, 0), vlabel, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        y_pos = label_height + idx * (img_size + spacing) + (img_size - text_height) / 2
        x_pos = (vertical_label_width - text_width) / 2
        draw.text((x_pos, y_pos), vlabel, fill=(0, 0, 0), font=font)


    combined_image.save(output_path)
    print(f"Saved comparison grid to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a side-by-side comparison grid for real and synthetic images.")
    parser.add_argument("real_csv", help="Path to real images CSV.")
    parser.add_argument("real_folder", help="Folder containing real PNG images.")
    parser.add_argument("synth_csv", help="Path to synthetic images CSV.")
    parser.add_argument("synth_folder", help="Folder containing synthetic PNG images.")
    parser.add_argument("output_path", help="Output PNG file path.")
    parser.add_argument("--label-column", type=str, default="breast_density", help="Label column to use.")

    args = parser.parse_args()

    create_comparison_grid(
        real_csv=args.real_csv,
        real_folder=args.real_folder,
        synth_csv=args.synth_csv,
        synth_folder=args.synth_folder,
        output_path=args.output_path,
        label_column=args.label_column
    )
