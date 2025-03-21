import os
import shutil
import json
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors

def get_camera_position(transform_matrix):
    """변환 행렬에서 카메라 위치 추출"""
    return np.array(transform_matrix)[:3, 3]

def select_reference_views(frames, directions=None):
    """
    지정된 방향(front, side, back, top)에서 참조 이미지를 선택
    
    Args:
        frames: 카메라 프레임 목록
        directions: 선택할 방향 목록 (기본값: front, side, back, top)
    
    Returns:
        선택된 프레임 인덱스 목록
    """
    if directions is None:
        directions = ['front', 'side', 'back', 'top']
    
    positions = []
    for i, frame in enumerate(frames):
        position = get_camera_position(frame['transform_matrix'])
        positions.append((i, position))
    
    # 방향별 점수 계산
    direction_scores = {
        'front': [],  # +z 방향
        'side': [],   # +x 방향
        'back': [],   # -z 방향
        'top': []     # +y 방향
    }
    
    for idx, pos in positions:
        # 정규화된 방향 점수
        pos_norm = np.linalg.norm(pos)
        if pos_norm == 0:
            continue
            
        normalized_pos = pos / pos_norm
        
        # 각 방향에 대한 점수 계산
        direction_scores['front'].append((idx, normalized_pos[2]))       # z 양수 방향이 앞쪽
        direction_scores['side'].append((idx, normalized_pos[0]))        # x 양수 방향이 측면
        direction_scores['back'].append((idx, -normalized_pos[2]))       # z 음수 방향이 뒤쪽
        direction_scores['top'].append((idx, normalized_pos[1]))         # y 양수 방향이 위쪽
    
    # 각 방향에서 가장 높은 점수를 가진 프레임 선택
    selected_indices = []
    for direction in directions:
        if direction in direction_scores and direction_scores[direction]:
            sorted_scores = sorted(direction_scores[direction], key=lambda x: x[1], reverse=True)
            selected_idx = sorted_scores[0][0]
            selected_indices.append(selected_idx)
    
    # 중복 제거
    selected_indices = list(set(selected_indices))
    
    return selected_indices

def find_single_nearby_frames(frames, ref_indices, max_total=None):
    """
    각 참조 이미지 주변에서 가장 가까운 이미지 한 장씩만 선택
    
    Args:
        frames: 카메라 프레임 목록
        ref_indices: 참조 프레임 인덱스 목록
        max_total: 최대 총 프레임 수 (기본값: None = 제한 없음)
    
    Returns:
        모든 선택된 프레임 인덱스 목록 (참조 + 주변 이미지)
    """
    # 모든 카메라 위치 추출
    positions = np.array([get_camera_position(frame['transform_matrix']) for frame in frames])
    
    # 모든 인덱스 목록 (참조 인덱스 제외)
    all_indices = set(range(len(frames)))
    ref_indices_set = set(ref_indices)
    available_indices = list(all_indices - ref_indices_set)
    
    # 이웃 탐색을 위한 모델 구축
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(positions)
    
    # 각 참조 프레임 주변의 가장 가까운 이웃 한 개씩 찾기
    nearby_indices = set()
    for ref_idx in ref_indices:
        distances, indices = nbrs.kneighbors([positions[ref_idx]])
        # 인덱스 0은 자기 자신, 인덱스 1이 가장 가까운 이웃
        neighbor_idx = indices[0][1]
        if neighbor_idx in available_indices:  # 이미 선택된 참조 이미지 제외
            nearby_indices.add(neighbor_idx)
    
    # 참조 인덱스와 주변 인덱스 결합
    selected_indices = list(ref_indices) + list(nearby_indices)
    
    # 최대 개수 제한이 있는 경우
    if max_total is not None and len(selected_indices) > max_total:
        selected_indices = selected_indices[:max_total]
        
    return selected_indices

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--cap_image_path', type=str, required=True)  # demo/nerf_synthetic
    parser.add_argument('-g', '--gen_image_path', type=str, required=True)  # removal/NeRF_ours/removed
    parser.add_argument('-n', '--save_path', type=str, required=True)       # sampled
    parser.add_argument('--max_total', type=int, default=None, help='Maximum total number of images')
    args = parser.parse_args()

    # objects = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
    objects = ["chair"]
    T_out = 100

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for object in objects:
        object_path = os.path.join(args.cap_image_path, object)   # demo/genAI_selected_poses/chair
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
        # save the gen cam pose to transforms_sel_train.json
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
        
        # 모든 프레임 리스트
        gen_frames = gen_cam_pose_json["frames"]
        
        # 참조 뷰 선택 (front, side, back, top)
        ref_indices = select_reference_views(gen_frames, directions=['front', 'side', 'back', 'top'])
        print(f"Selected reference views: {ref_indices}, total: {len(ref_indices)}")
        
        # 각 참조 이미지 주변의 가장 가까운 이미지 한 장씩 선택
        selected_indices = find_single_nearby_frames(gen_frames, ref_indices, max_total=args.max_total)
        print(f"Selected total {len(selected_indices)} frames")
        print(f"  Reference images: {len(ref_indices)}")
        print(f"  Additional images: {len(selected_indices) - len(ref_indices)}")
        
        for number in range(1, 2):
            save_path = os.path.join(args.save_path, f"N{number}M{T_out}", object)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(os.path.join(save_path, "images")):
                os.makedirs(os.path.join(save_path, "images"))
            
            # 선택된 인덱스로 참조 프레임 목록 생성
            ref_frames = []
            for i, idx in enumerate(ref_indices):
                ref_frames.append(gen_frames[idx])
                # 파일 경로 변경
                ref_frames[-1]["file_path"] = f"images/ref_{i:03d}"
            
            # 참조 프레임 저장
            ref_cam_pose_json = {"camera_angle_x": gen_cam_pose_json["camera_angle_x"], "frames": ref_frames}
            with open(os.path.join(save_path, "transforms_reference.json"), "w") as ref_file:
                json.dump(ref_cam_pose_json, ref_file, indent=4)
            
            # 모든 선택된 프레임
            selected_frames = []
            for i, idx in enumerate(selected_indices):
                selected_frames.append(gen_frames[idx])
                # 파일 경로 변경
                selected_frames[-1]["file_path"] = f"images/selected_{i:03d}"
            
            # 선택된 프레임 저장
            selected_cam_pose_json = {"camera_angle_x": gen_cam_pose_json["camera_angle_x"], "frames": selected_frames}
            with open(os.path.join(save_path, "transforms_selected.json"), "w") as selected_file:
                json.dump(selected_cam_pose_json, selected_file, indent=4)
            
            # 이미지 복사
            for frame in selected_frames:
                src_img_path = os.path.join(args.gen_image_path, frame["file_path"].replace("images", f"N{number}M{T_out}/{object}"))
                dst_img_path = os.path.join(save_path, frame["file_path"])
                shutil.copy(src_img_path, dst_img_path)
