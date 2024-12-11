import cv2
import numpy as np
from skimage.filters import threshold_otsu


def resize_to_even(image):
    """调整图像尺寸为偶数"""
    h, w = image.shape[:2]
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    return cv2.resize(image, (new_w, new_h))


def manual_pyr_up(image, target_size):
    """手动调整 pyrUp 输出尺寸"""
    upscaled = cv2.pyrUp(image)
    return cv2.resize(upscaled, target_size)


def itti_saliency_map(image):
    """计算 Itti-Koch 显著性图"""
    # 转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯金字塔
    pyr = [resize_to_even(gray)]  # 确保起始图像为偶数
    for i in range(1, 6):  # 创建 6 层金字塔
        down = cv2.pyrDown(pyr[i - 1])
        pyr.append(resize_to_even(down))  # 确保每层尺寸为偶数
    
    # 中心环对比 (Center-Surround Differences)
    cs_maps = []
    for i in range(2, 5):  # 高金字塔层
        center = pyr[i]
        for delta in range(1, 3):  # 环形差异
            if i + delta < len(pyr):  # 防止越界
                surround = pyr[i + delta]
                target_size = (center.shape[1], center.shape[0])  # 恢复到中心尺寸
                surround_resized = manual_pyr_up(surround, target_size)  # 使用手动调整
                cs_maps.append(cv2.absdiff(center, surround_resized))
    
    # 合并显著性图
    saliency_map = np.zeros_like(gray, dtype=np.float32)
    for m in cs_maps:
        resized_m = cv2.resize(m, gray.shape[::-1])  # 恢复到原始输入图像尺寸
        saliency_map += resized_m.astype(np.float32)
    
    # 归一化到 0-255
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
    return saliency_map.astype(np.uint8)


def generate_aoi_overlay(image, saliency_map, min_area=1000, top_n=10, blur_radius=15):
    """从显著性图生成前 N 个面积最大的 AOI，柔化显著性图，并直接绘制到原图上"""
    # 使用 Otsu 阈值分割显著性区域
    threshold = threshold_otsu(saliency_map)
    _, binary_map = cv2.threshold(saliency_map, threshold, 255, cv2.THRESH_BINARY)
    
    # 提取轮廓
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 筛选轮廓：剔除小面积区域，并按面积排序
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:top_n]  # 取前 N 个
    
    # 柔化显著性图
    smoothed_saliency = cv2.GaussianBlur(saliency_map, (blur_radius, blur_radius), 0)
    
    # 创建彩色填充图
    overlay = image.copy()
    
    # 随机生成颜色并填充区域
    for contour in sorted_contours:
        color = tuple(np.random.randint(0, 255, size=3).tolist())  # 随机颜色
        cv2.drawContours(overlay, [contour], -1, color, thickness=cv2.FILLED)
    
    # 叠加原始图像和填充图，添加透明度
    alpha = 0.6  # 设置透明度
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    return blended


# 主程序
stimuli_path = "dataset/pics/TrainingData/Images/28.png"  # 替换为你的图像路径
output_image_path = "output_image_with_transparency.png"  # 输出文件路径

# 加载图像
image = cv2.imread(stimuli_path)
if image is None:
    raise ValueError("Failed to load the stimuli image.")

# 调整输入图像尺寸为偶数
image = resize_to_even(image)

# 计算 Itti Saliency Map
itti_saliency = itti_saliency_map(image)

# 从显著性图生成前 10 个面积最大的 AOI，柔化显著性图
aoi_img = generate_aoi_overlay(image, itti_saliency, min_area=1000, top_n=10)

# 保存结果为 PNG 文件
cv2.imwrite(output_image_path, aoi_img)
