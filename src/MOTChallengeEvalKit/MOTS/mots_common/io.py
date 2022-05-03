import PIL.Image as Image
import numpy as np
import pycocotools.mask as rletools
import glob
import os


class SegmentedObject:
  def __init__(self, mask, class_id, track_id):
    self.mask = mask
    self.class_id = class_id
    self.track_id = track_id


def load_sequences(path, seqmap):
  objects_per_frame_per_sequence = {}
  for seq in seqmap:
    print("Loading sequence", seq)
    seq_path_folder = os.path.join(path, seq)
    seq_path_txt = os.path.join(path, seq + ".txt")
    if os.path.isdir(seq_path_folder):
      objects_per_frame_per_sequence[seq] = load_images_for_folder(seq_path_folder)
    elif os.path.exists(seq_path_txt):
      objects_per_frame_per_sequence[seq] = load_txt(seq_path_txt)
    else:
      raise Exception( "<exc>Can't find data in directory " + path + "<!exc>")

  return objects_per_frame_per_sequence


def load_txt(path):
  objects_per_frame = {}
  track_ids_per_frame = {}  # To check that no frame contains two objects with same id
  combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
  with open(path, "r") as f:
    for line in f:
      line = line.strip()
      fields = line.split(" ")
      try:
        frame = int(fields[0])
      except:
        raise Exception("<exc>Error in {} in line: {}<!exc>".format(path.split("/")[-1], line))
      if frame not in objects_per_frame:
        objects_per_frame[frame] = []
      if frame not in track_ids_per_frame:
        track_ids_per_frame[frame] = set()
      if int(fields[1]) in track_ids_per_frame[frame]:
        raise Exception("<exc>Multiple objects with track id " + fields[1] + " in frame " + fields[0] + "<!exc>")
      else:
        track_ids_per_frame[frame].add(int(fields[1]))

      class_id = int(fields[2])
      if not(class_id == 1 or class_id == 2 or class_id == 10):
        raise Exception( "<exc>Unknown object class " + fields[2] + "<!exc>")

      mask = {'size': [int(fields[3]), int(fields[4])], 'counts': fields[5].encode(encoding='UTF-8')}
      if frame not in combined_mask_per_frame:
        combined_mask_per_frame[frame] = mask
      elif rletools.area(rletools.merge([combined_mask_per_frame[frame], mask], intersect=True)) > 0.0:
        raise Exception( "<exc>Objects with overlapping masks in frame " + fields[0] + "<!exc>")
      else:
        combined_mask_per_frame[frame] = rletools.merge([combined_mask_per_frame[frame], mask], intersect=False)
      objects_per_frame[frame].append(SegmentedObject(
        mask,
        class_id,
        int(fields[1])
      ))

  return objects_per_frame


def load_images_for_folder(path):
  files = sorted(glob.glob(os.path.join(path, "*.png")))

  objects_per_frame = {}
  for file in files:
    objects = load_image(file)
    frame = filename_to_frame_nr(os.path.basename(file))
    objects_per_frame[frame] = objects

  return objects_per_frame


def filename_to_frame_nr(filename):
  assert len(filename) == 10, "Expect filenames to have format 000000.png, 000001.png, ..."
  return int(filename.split('.')[0])


def load_image(filename, id_divisor=1000):
  img = np.array(Image.open(filename))
  obj_ids = np.unique(img)

  objects = []
  mask = np.zeros(img.shape, dtype=np.uint8, order="F")  # Fortran order needed for pycocos RLE tools
  for idx, obj_id in enumerate(obj_ids):
    if obj_id == 0:  # background
      continue
    mask.fill(0)
    pixels_of_elem = np.where(img == obj_id)
    mask[pixels_of_elem] = 1
    objects.append(SegmentedObject(
      rletools.encode(mask),
      obj_id // id_divisor,
      obj_id
    ))

  return objects


def load_seqmap(seqmap_filename):
  print("Loading seqmap...")
  seqmap = []
  max_frames = {}
  with open(seqmap_filename, "r") as fh:
    for i, l in enumerate(fh):
      fields = l.split(" ")
      seq = "%04d" % int(fields[0])
      seqmap.append(seq)
      max_frames[seq] = int(fields[3])
  return seqmap, max_frames


def write_sequences(gt, output_folder):
  os.makedirs(output_folder, exist_ok=True)
  for seq, seq_frames in gt.items():
    write_sequence(seq_frames, os.path.join(output_folder, seq + ".txt"))
  return


def write_sequence(frames, path):
  with open(path, "w") as f:
    for t, objects in frames.items():
      for obj in objects:
        print(t, obj.track_id, obj.class_id, obj.mask["size"][0], obj.mask["size"][1],
              obj.mask["counts"].decode(encoding='UTF-8'), file=f)
