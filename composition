import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 图像预处理
def preprocess_image(image_path):
    print(f"加载图像: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("转换为灰度图像完成")
    
    # 展示原始图像和灰度图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("原始图像")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title("灰度图像")
    plt.imshow(gray_image, cmap='gray')
    plt.show()
    
    return image, gray_image

# 使用 SIFT 提取特征点，设置 nfeatures 参数来控制特征点数量
def extract_and_draw_keypoints(image, gray_image, max_features=3000):
    print("开始提取特征点...")
    sift = cv2.SIFT_create(nfeatures=max_features)  # 使用 SIFT 算法，并限制特征点数量
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    print(f"提取到 {len(keypoints)} 个特征点")
    
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    
    # 展示特征点图像
    plt.figure(figsize=(10, 5))
    plt.title("特征点提取结果")
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.show()
    
    return keypoints, descriptors, image_with_keypoints

# 使用 Lowe's ratio test 进行匹配
def match_keypoints_with_ratio_test(desc1, desc2, ratio_thresh=0.75):
    print("开始匹配特征点...")
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # 不使用 crossCheck
    matches = bf.knnMatch(desc1, desc2, k=2)  # 找到两个最好的匹配

    # 使用 Lowe's ratio test 来过滤匹配
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    print(f"匹配到 {len(good_matches)} 对特征点")
    return good_matches


# 绘制匹配结果
def draw_matches(image1, image2, kp1, kp2, matches):
    print("绘制特征点匹配结果...")
    matched_image = cv2.drawMatches(image1, kp1, image2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 7))
    plt.title("特征点匹配结果")
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.show()
    return matched_image

# 图像拼接
def stitch_images(image1, image2, kp1, kp2, matches):
    print("开始拼接图像...")
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        raise ValueError("无法计算单应性矩阵")
    
    height, width, _ = image2.shape
    result = cv2.warpPerspective(image1, H, (width * 2, height))
    result[0:height, 0:width] = image2
    return result

def process_and_stitch(img1_path, img2_path):
    start_time = datetime.now()

    # 预处理
    image1, gray1 = preprocess_image(img1_path)
    image2, gray2 = preprocess_image(img2_path)

    # 提取特征点
    kp1, desc1, img1_kp = extract_and_draw_keypoints(image1, gray1, max_features=1000)
    kp2, desc2, img2_kp = extract_and_draw_keypoints(image2, gray2, max_features=1000)

    # 匹配特征点（使用 Lowe's ratio test 进行过滤）
    matches = match_keypoints_with_ratio_test(desc1, desc2)
    draw_matches(image1, image2, kp1, kp2, matches)

    # 拼接图片
    result = stitch_images(image1, image2, kp1, kp2, matches)

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    fps = 1 / processing_time

    print(f"处理时间: {processing_time:.2f} 秒")
    print(f"帧率: {fps:.2f} FPS")

    # 显示拼接结果
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.title("图像 1 特征点")
    plt.imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 2)
    plt.title("图像 2 特征点")
    plt.imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 1, 2)
    plt.title("拼接后的图像")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


# 示例调用
# 替换为实际图片路径
process_and_stitch("3.jpg", "4.jpg")
process_and_stitch("8.jpg", "9.jpg")
