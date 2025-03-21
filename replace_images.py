# python replace_images.py -i demo/nerf_synthetic -g removal/NeRF_ours/upsampled -n demo/combined_dataset
import os
import shutil
import json
import itertools

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--cap_image_path', type=str, required=True)  # 캡처된 이미지 경로
    parser.add_argument('-g', '--gen_image_path', type=str, required=True)  # 생성된 이미지 경로 (removal/NeRF_ours/upsampled/)
    parser.add_argument('-n', '--save_path', type=str, required=True)       # 저장할 경로 (demo/combined_dataset/)
    args = parser.parse_args()

    objects = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
    numbers = list(range(1, 101))  # 1~100
    # objects = ["chair"]
    # numbers = [1, 2, 3, 4, 5, 10, 50, 100]
    all_combinations = list(itertools.product(objects, numbers))

    T_out = 100

    for obj, num in all_combinations:
        save_obj_path = os.path.join(args.save_path, obj, f"N{num}M{T_out}")
        save_train_path = os.path.join(save_obj_path, 'train')

        # 저장 폴더 생성 (한 번만 실행)
        os.makedirs(save_train_path, exist_ok=True)

        gen_path = os.path.join(args.gen_image_path, f'N{num}M{T_out}', obj)  # 생성된 이미지 경로
        cap_obj_path = os.path.join(args.cap_image_path, obj)
        cap_train_path = os.path.join(cap_obj_path, 'train')
        cap_json_path = os.path.join(cap_obj_path, "transforms_train.json")

        # 📌 생성된 이미지 확인 (100개 존재하는지)
        if not os.path.exists(gen_path):
            print(f"Warning: Missing generated data for {gen_path}")
            continue
        
        gen_img_path = f'demo/genAI_selected_poses/{obj}/{int(num)}/images'
        gen_img_list = sorted([f for f in os.listdir(gen_img_path) if f.endswith(".png")])
        gen_img_set = set(gen_img_list)  # `set` 변환을 한 번만 수행

        # 📌 캡처된 이미지 및 카메라 포즈 로드
        if not os.path.exists(cap_train_path) or not os.path.exists(cap_json_path):
            print(f"Warning: Missing captured data for {obj}")
            continue

        # ✅ `os.listdir()`을 한 번만 실행 (성능 최적화)
        cap_train_img_list = os.listdir(cap_train_path)
        cap_train_img_set = set(cap_train_img_list)  # 빠른 검색을 위해 `set` 사용

        with open(cap_json_path, "r") as file:
            cap_train_pose = json.load(file)["frames"]

        updated_cam_pose = []
        copy_tasks = []  # ✅ 한 번에 복사할 파일 리스트 저장

        for cap_frame in cap_train_pose:
            img_idx = int(cap_frame["file_path"].split('_')[-1].split('.')[0])
            cap_img_name = f"r_{img_idx}.png"

            if not cap_img_name in gen_img_set:
                # ✅ 생성된 이미지에서 교체할 이미지 선택
                gen_img_src = os.path.join(gen_path, f'{img_idx}.png')
                gen_img_dest = os.path.join(save_train_path, cap_img_name)

                copy_tasks.append((gen_img_src, gen_img_dest))  # 복사할 파일을 리스트에 추가
                print(f"Replaced: {gen_img_src} -> {gen_img_dest}")

            else:
                # ✅ 기존 캡처된 이미지 유지
                cap_img_src = os.path.join(cap_train_path, cap_img_name)
                cap_img_dest = os.path.join(save_train_path, cap_img_name)

                copy_tasks.append((cap_img_src, cap_img_dest))  # 복사할 파일 추가

            # # 카메라 포즈 업데이트
            # updated_cam_pose.append(cap_frame)

        # ✅ 최적화: 여러 개의 파일을 한꺼번에 복사
        for src, dest in copy_tasks:
            shutil.copy2(src, dest)  # `copy2()`를 사용하여 복사 속도 최적화

        # 📌 새로운 transforms_train.json 저장
        save_json_path = os.path.join(save_obj_path, "transforms_train.json")
        # with open(save_json_path, "w") as file:
        #     json.dump({"frames": updated_cam_pose}, file, indent=4)
        shutil.copy2(cap_json_path, save_json_path)

    print("Processing complete!")
