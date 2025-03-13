import os
import shutil
import json
import numpy as np
import cv2


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ref_image_path', type=str, required=True)  # demo/genAI_selected_poses/
    parser.add_argument('-g', '--gen_image_path', type=str, required=True)  # removal/NeRF_ours/removed
    parser.add_argument('-n', '--save_path', type=str, required=True)       # sampled
    args = parser.parse_args()

    # objects = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
    objects = ["chair"]
    T_out = 100

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for object in objects:
        object_path = os.path.join(args.ref_image_path, object)   # demo/genAI_selected_poses/chair
        print(f"Object: {object_path}")


        # read the gen cam pose from transforms_train.json
        with open(os.path.join("demo/nerf_synthetic", object, "transforms_train.json"), "r") as file:
            gen_cam_pose_json = json.load(file)

        input_json = gen_cam_pose_json
        input_json_frames = input_json["frames"]
        # change file name from ./train/r_0 to image/r_0.png
        for frame in input_json_frames:
            frame["file_path"] = frame["file_path"].replace("./train", "images").replace(".png", ".png")
        input_json["frames"] = input_json_frames
        # save the gen cam pose to transforms_train.json
        if not os.path.exists(os.path.join(args.save_path, f"N1M{T_out}", object)):
            os.makedirs(os.path.join(args.save_path, f"N1M{T_out}", object))
        with open(os.path.join(args.save_path, f"N1M{T_out}", object, "transforms_sel_train.json"), "w") as file:
            json.dump(input_json, file, indent=4)
        

        # read the out cam pose from transforms_test.json
        with open(os.path.join("demo/nerf_synthetic", object, "transforms_test.json"), "r") as out_file:
            out_cam_pose_json = json.load(out_file)

        input_json = out_cam_pose_json
        input_json_frames = input_json["frames"]
        # change file name from ./test/r_0 to image/r_0.png
        for frame in input_json_frames:
            frame["file_path"] = frame["file_path"].replace("./test", "images").replace(".png", ".png")
        input_json["frames"] = input_json_frames
        # save the out cam pose to transforms_test.json
        with open(os.path.join(args.save_path, f"N1M{T_out}", object, "transforms_sel_test.json"), "w") as out_file:
            json.dump(input_json, out_file, indent=4)
        
        
        for number in range(1, 2):
            ref_path = os.path.join(object_path, str(number))    # demo/genAI_selected_poses/chair/1
            ref_cam_pose = os.path.join(ref_path, "transforms_selected.json")         # demo/genAI_selected_poses/chair/1/transforms_selected.json
            save_path = os.path.join(args.save_path, f"N{number}M{T_out}", object)   # sampled/N1M100/chair
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(os.path.join(save_path, "images")):
                os.makedirs(os.path.join(save_path, "images"))

            # read the ref cam pose from transforms_selected.json
            with open(ref_cam_pose, "r") as ref_file:
                ref_cam_pose_json = json.load(ref_file)
            
            # copy ref images into save_path/images
            ref_frame_list = ref_cam_pose_json["frames"]
            ref_img_list = [frame["file_path"].replace("\\", "/") for frame in ref_frame_list]
            ref_img_idx = [int(img.split("/")[-1].split(".")[0].split("_")[-1]) for img in ref_img_list]

            for ref_img in ref_img_list:
                shutil.copy(os.path.join(ref_path, ref_img), os.path.join(save_path, ref_img))

            # copy gen images into save_path/images
            gen_frame_list = gen_cam_pose_json["frames"]        
            gen_img_list = [frame["file_path"].split(".")[-1] for frame in gen_frame_list]  # ./train/r_0 -> train/r_0
            gen_img_idx = [int(img.split("/")[-1].split(".")[0].split("_")[-1]) for img in gen_img_list]
            # delete idx from gen_img_list if it is in ref_img_idx
            for idx in ref_img_idx:
                if idx in gen_img_idx:
                    gen_img_list.pop(gen_img_idx.index(idx))
                    gen_img_idx.remove(idx)

            for gen_img in gen_img_idx:
                # read image first
                # img = cv2.imread(os.path.join(args.gen_image_path, f"N{number}M{T_out}", object, f"r_{gen_img}.png"))
                # # upsample the image to same size as ref image
                # # img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_CUBIC)
                # # save the image
                # cv2.imwrite(os.path.join(save_path, f"images/r_{gen_img}.png"), img)
                
                shutil.copy(os.path.join(args.gen_image_path, f"N{number}M{T_out}", object, f"r_{gen_img}.png"), 
                            os.path.join(save_path, f"images/r_{gen_img}.png"))
                