import os
from PIL import Image
import numpy as np
import torch
import cv2

class BackgroundRemoval:
    def __init__(self, device='cuda'):
        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)
        image = self.interface([image])[0]
        image = np.array(image)
        return image
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, required=True)
    parser.add_argument('-n', '--save_path', type=str, required=True)
    args = parser.parse_args()
    
    # remove background from images
    image_dir = args.image_path
    print("Input path:", image_dir)

    split = {'chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship'}
    object_list = os.listdir(image_dir)
    print(f"{len(object_list)} objects are found")
    if os.path.exists(args.save_path) == False:
        os.makedirs(args.save_path)

    for object in object_list:
        print(f"Object: {object}")
        save_path = os.path.join(args.save_path, object)
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)

        image_list = os.listdir(os.path.join(image_dir, object))
        for image_path in image_list:
            if image_path.endswith('.png') == False:
                print(f"{image_path} is passed")
                continue
            background_removal = BackgroundRemoval()
            image = np.array(Image.open(os.path.join(image_dir, object, image_path)))
            image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_CUBIC)
            image = background_removal(image)
            Image.fromarray(image).save(os.path.join(save_path, image_path))
            print(f"Image is saved in {os.path.join(save_path, image_path)}")
    # background_removal = BackgroundRemoval()
    # image = np.array(Image.open(args.image_path))
    # image = background_removal(image)
    # Image.fromarray(image).save(args.name)
    
