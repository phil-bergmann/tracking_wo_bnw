import sys
from mots_common.io import load_sequences, load_seqmap, write_sequences


if __name__ == "__main__":
  if len(sys.argv) != 4:
    print("Usage: python images_to_txt.py gt_img_folder gt_txt_output_folder seqmap")
    sys.exit(1)

  gt_img_folder = sys.argv[1]
  gt_txt_output_folder = sys.argv[2]
  seqmap_filename = sys.argv[3]

  seqmap, _ = load_seqmap(seqmap_filename)
  print("Loading ground truth images...")
  gt = load_sequences(gt_img_folder, seqmap)
  print("Writing ground truth txts...")
  write_sequences(gt, gt_txt_output_folder)
