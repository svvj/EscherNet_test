#!/usr/bin/env python3
import os
import json
import shutil
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
import random

def create_combined_dataset():
    """
    evenly_selected_poses의 전체 pose와 index를 기준으로,
    reference 이미지는 evenly_selected_poses에서,
    추가 이미지는 combined_dataset에서 가져와서 저장
    """
    print("Creating combined dataset with additional generated images...")
    
    # 기본 경로 설정
    base_dir = Path("/data/gpfs/projects/punim2482/workspace/EscherNet_test/demo")
    evenly_selected_dir = base_dir / "evenly_selected_poses"
    combined_dataset_dir = base_dir / "combined_dataset"
    output_base_dir = base_dir / "combined_less_num_dataset"
    
    # 출력 디렉토리 생성
    output_base_dir.mkdir(exist_ok=True)
    
    # 객체 리스트
    objects = ["drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
    # objects = ["chair"]

    # 기준 이미지(Reference) 총 개수 리스트
    reference_image_counts = [5, 10, 20, 30, 40, 50, 1, 2]
    
    # 생성 이미지 추가 비율 (0%, 25%, 50%, 75%, 100%)
    scale = 100
    additional_ratios = [0.0, 0.25, 0.50, 0.75, 1.0]
    less_num_ratios = [0.0, 1.0, 2.0, 3.0, 4.0]
    
    # 모든 객체에 대해 처리
    for obj in objects:
        print(f"\nProcessing object: {obj}")
        
        # evenly_selected 디렉토리 확인
        evenly_obj_dir = evenly_selected_dir / obj
        if not evenly_obj_dir.exists():
            print(f"Directory not found: {evenly_obj_dir}")
            continue
        
        # combined_dataset 디렉토리 확인
        combined_obj_dir = combined_dataset_dir / obj
        if not combined_obj_dir.exists():
            print(f"Combined dataset directory not found: {combined_obj_dir}")
            continue

        
        for ref_num in reference_image_counts:

            for idx, add_ratio in enumerate(additional_ratios):
                if ref_num <= 2:
                    add_ratio = less_num_ratios[idx]
                    scale = 1
                else:
                    scale = 100

                # 출력 디렉토리 생성
                output_dir = output_base_dir / obj / f"N{ref_num}R{int(add_ratio * scale)}"
                output_dir.mkdir(exist_ok=True, parents=True)
                
                # 추가 이미지 개수
                add_num = int(ref_num * add_ratio)

                # 총 이미지 개수
                total_num = ref_num + add_num

                # pose index 파일 경로
                total_pose_index_path = evenly_obj_dir / f"{total_num}/transforms_train.json"
                if not total_pose_index_path.exists():
                    print(f"Pose index file not found: {total_pose_index_path}")
                    continue
                # pose index 파일 로드
                with open(total_pose_index_path, "r") as f:
                    total_pose_index = json.load(f)
                total_poses = total_pose_index["frames"]
                total_images = [p["file_path"] for p in total_poses]
                total_indices = [int(t.split("_")[-1].split(".")[0]) for t in total_images]

                # reference 이미지 선택
                ref_indexes = evenly_obj_dir / f"{ref_num}/transforms_train.json"
                if not ref_indexes.exists():
                    print(f"Reference pose index file not found: {ref_indexes}")
                    continue
                with open(ref_indexes, "r") as f:
                    ref_pose_index = json.load(f)
                ref_poses = ref_pose_index["frames"]
                ref_images = [p["file_path"] for p in ref_poses]
                ref_indices = [r.split("_")[-1].split(".")[0] for r in ref_images]
                ref_indices = [int(i) for i in ref_indices]
                
                # 추가 이미지 선택 (total - reference)
                add_indexes = list(set(total_indices) - set(ref_indices))
                if len(add_indexes) != add_num:
                    print(f"Invalid number of additional images: {len(add_indexes)} != {add_num}")
                    continue

                # 이미지 복사
                for i in ref_images:
                    img = i.split("/")[-1]+".png"
                    folder = f"{obj}/train"
                    
                    ref_img_path = os.path.join("/data/gpfs/projects/punim2482/workspace/EscherNet_test/demo/nerf_synthetic", folder, img)
                    output_img_path = output_dir / "images"/ img
                    os.makedirs(output_dir/'images', exist_ok=True)
                    shutil.copy(ref_img_path, output_img_path)
                    
                for i in add_indexes:
                    # remaining images
                    img = f"r_{i}.png"
                    folder = f"N{ref_num}M100/train"

                    add_img_path = combined_obj_dir / folder / img
                    output_img_path = output_dir / "images"/ img
                    os.makedirs(output_dir/'images', exist_ok=True)
                    shutil.copy(add_img_path, output_img_path)
                    
                # copy json
                out_json = output_dir / "transforms_train.json"
                shutil.copy(total_pose_index_path, out_json)


                print(f"{ref_num}, {add_num}: {output_dir}")
    print("Done")
    

if __name__ == "__main__":
    create_combined_dataset()