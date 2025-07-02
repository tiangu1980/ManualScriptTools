
from PIL import Image
import numpy as np
import argparse
import hashlib
import time

# python .\images_differ.py 16_1.bmp 16_2.bmp --separator 0 0 0

# -*- coding: utf-8 -*-

def load_image_as_bitmap(image_path, mode='RGB'):
    img = Image.open(image_path)
    return img.convert(mode)

def split_image_with_separators(bitmap, separator_color):
    pixels = np.array(bitmap)
    height = pixels.shape[0]
    blocks = []
    current_block = {'start': 0, 'type': 'content'}
    in_separator = False

    for y in range(height):
        if np.all(pixels[y] == separator_color):
            if not in_separator:
                if current_block['start'] < y:  # 保存前一个内容块
                    current_block['end'] = y - 1
                    blocks.append(current_block)
                blocks.append({'start': y, 'type': 'separator'})
                in_separator = True
        else:
            if in_separator:
                blocks[-1]['end'] = y - 1
                in_separator = False
                current_block = {'start': y, 'type': 'content'}

    # 处理最后未闭合的块
    if in_separator:
        blocks[-1]['end'] = height - 1
    elif current_block['start'] < height:
        current_block['end'] = height - 1
        blocks.append(current_block)

    return blocks

def compire_images(bitmap1, bitmap2):
    if bitmap1.size != bitmap2.size:
        raise ValueError("两张图片的尺寸不一致")
    
    pixels1 = np.array(bitmap1)
    pixels2 = np.array(bitmap2)
    
    if not np.array_equal(pixels1, pixels2):
        print("图片内容不同")
        find_differences(pixels1, pixels2)
    else:
        print("图片内容相同")

# 未排序的listMatch存储匹配行信息。
listMatch = []

# ListDiffL 和 ListDiffR 分别存储左侧和右侧不匹配的行信息并且长度相等。
# ListDiffL 和 ListDiffR 的数据格式为 [[所有match和不match的块的全局排序索引值 globalBlockIndex, 起始行号, 长度], ...]
ListDiffL = []
ListDiffR = []
# ListSame 处理全局块排序后的listMatch的全局排序索引值和匹配信息。
# ListSame 的数据格式为 [[所有match的块的全局排序索引值 globalBlockIndex, 左侧起始行号, 匹配长度], ...]
listSame = []
        
# pixels1: expected, pixels2: actual
# 比较两张图片的像素差异
def find_differences(pixelsL, pixelsR):    
    # hashes: [[行号, 哈希值], ...]
    hashes_L = generate_line_hashes(pixelsL)
    hashes_R = generate_line_hashes(pixelsR)
    
    # 获取匹配行信息
    GetMatchingLines(hashes_L, hashes_R)    
      # 获取匹配与不匹配关系的块信息
    GetMatchingInfoBlocks(hashes_L, hashes_R, listMatch)
    sorted_data_DiffL = sorted(ListDiffL, key=lambda x: x[0])
    sorted_data_DiffR = sorted(ListDiffR, key=lambda x: x[0])
    
    print("不匹配行信息L:", ListDiffL)
    print("不匹配行信息R:", ListDiffR)
    print("匹配行信息:", listSame)
    
    # 输出差异PNG图片
    OutputMergedDiffPng(listSame, ListDiffL, ListDiffR, pixelsL, pixelsR)
    
# OutputMergedDiffPng : 输出合并后的png图片
# 从 pixelsL 或者 pixelsR 得到图片的宽度 width。
# 首先，新建一个空的像素矩阵 pixesMatrix，高度为 0，宽度为 width。
# 计算 listSame 与 ListDiffL（或者 ListDiffR）长度之和(总块数)为 total_BlockCount。
# 循环处理 total_BlockCount 次：
#     当前循环时，索引值为i。
#     从 listSame 与 ListDiffL（或者 ListDiffR）中寻找是否有 item[0] 值为i的元素item。
#     由前述逻辑，可知：当在 listSame 中找到时，表示当前块是匹配的块；当在 ListDiffL（或者 ListDiffR）中找到时，表示当前块是不匹配的块。
#     如果在 listSame 中找到，则在 pixelsMatrix 添加 item[2]（匹配长度）行的宽度为 width 的透明元素。
#     如果在 ListDiffL（或者 ListDiffR）中找到, 则得到 item_L 和 item_R 。
#         在 pixelsL 中取出从第 item_L[1] 行开始一共 item_L[2] 个行，为新矩阵blocksL 。
#         在 pixelsR 中取出从第 item_R[1] 行开始一共 item_R[2] 个行，为新矩阵blocksR 。
#         计算 blocksL 和blocksR的能够一一对应的行的数量 LR_diff_count，LR_diff_count 是 item_L[2] 与 item_R[2] 的最小值。
#         比较 item_L[2] 与 item_R[2] 的大小:
#             当item_L[2] 比较大时， l_longer_count = item_L[2] - LR_diff_count；
#             当item_R[2] 比较大时， r_longer_count = item_R[2] - LR_diff_count; 
#             如果item_L[2] 与 item_R[2] 一样大时， l_longer_count = R_longer_count = 0 。
#         循环 LR_diff_count 次：
#             将blocksL和blocksR的对应行(第 0 行到第 LR_diff_count - 1 行)一一进行像素对比：像素值一样的输出透明像素，不一样的输出红色像素，得到一个新的像素行。
#             将新的像素行添加到 pixelsMatrix 中。
#         如果 l_longer_count > 0, 则将 blocksL 最后的一共 l_longer_count 行的像素行执行 色彩值减半然后R值为255 操作后添加到 pixelsMatrix 中。
#         如果 r_longer_count > 0, 则将 blocksR 最后的一共 r_longer_count 行的像素行执行 色彩值减半然后B值为255 操作后添加到 pixelsMatrix 中。
# 最后，将 pixelsMatrix 转换为PNG图片并保存， 文件名为 diff_ouput_{tick时间戳}.png。
def OutputMergedDiffPng(listSame, ListDiffL, ListDiffR, pixelsL, pixelsR):
    """
    输出合并后的差异PNG图片
    :param listSame: 匹配块信息列表 [[globalBlockIndex, 起始行号, 匹配长度], ...]
    :param ListDiffL: 左侧不匹配块信息列表 [[globalBlockIndex, 起始行号, 长度], ...]
    :param ListDiffR: 右侧不匹配块信息列表 [[globalBlockIndex, 起始行号, 长度], ...]
    :param pixelsL: 左侧图片像素矩阵
    :param pixelsR: 右侧图片像素矩阵
    """
    # 从 pixelsL 或者 pixelsR 得到图片的宽度 width
    width = pixelsL.shape[1]
    channels = pixelsL.shape[2] if len(pixelsL.shape) == 3 else 1
    
    # 首先，新建一个空的像素矩阵 pixelsMatrix，高度为 0，宽度为 width
    pixelsMatrix = np.empty((0, width, channels), dtype=np.uint8)
    
    # 计算 listSame 与 ListDiffL（或者 ListDiffR）长度之和(总块数)为 total_BlockCount
    total_BlockCount = len(listSame) + len(ListDiffL)
    print(f"总块数: {total_BlockCount}")
    
    # 循环处理 total_BlockCount 次
    for i in range(total_BlockCount):
        print(f"处理第 {i} 个块")
        
        # 从 listSame 与 ListDiffL（或者 ListDiffR）中寻找是否有 item[0] 值为i的元素item
        same_item = None
        diff_item_L = None
        diff_item_R = None
        
        # 在 listSame 中查找
        for item in listSame:
            if item[0] == i:
                same_item = item
                break
        
        # 在 ListDiffL 和 ListDiffR 中查找
        for j, item_L in enumerate(ListDiffL):
            if item_L[0] == i:
                diff_item_L = item_L
                diff_item_R = ListDiffR[j]  # ListDiffL 和 ListDiffR 对应
                break
        
        # 如果在 listSame 中找到，表示当前块是匹配的块
        if same_item is not None:
            print(f"  匹配块: {same_item}")
            # 在 pixelsMatrix 添加 item[2]（匹配长度）行的宽度为 width 的透明元素
            match_length = same_item[2]
            if match_length > 0:
                # 创建透明像素行 (RGBA格式，Alpha=0表示透明)
                if channels == 3:
                    # RGB格式，添加alpha通道
                    transparent_block = np.zeros((match_length, width, 4), dtype=np.uint8)
                    transparent_block[:, :, 3] = 0  # Alpha通道设为0（透明）
                else:
                    # 已经是RGBA格式
                    transparent_block = np.zeros((match_length, width, channels), dtype=np.uint8)
                    if channels == 4:
                        transparent_block[:, :, 3] = 0  # Alpha通道设为0（透明）
                
                # 确保pixelsMatrix有正确的通道数
                if pixelsMatrix.shape[0] == 0:
                    pixelsMatrix = transparent_block
                else:
                    pixelsMatrix = np.vstack([pixelsMatrix, transparent_block])
        
        # 如果在 ListDiffL（或者 ListDiffR）中找到，表示当前块是不匹配的块
        elif diff_item_L is not None and diff_item_R is not None:
            print(f"  不匹配块L: {diff_item_L}, R: {diff_item_R}")
            
            # 在 pixelsL 中取出从第 item_L[1] 行开始一共 item_L[2] 个行，为新矩阵blocksL
            start_L = diff_item_L[1]
            length_L = diff_item_L[2]
            if length_L > 0:
                blocksL = pixelsL[start_L:start_L + length_L]
            else:
                blocksL = np.empty((0, width, channels), dtype=np.uint8)
            
            # 在 pixelsR 中取出从第 item_R[1] 行开始一共 item_R[2] 个行，为新矩阵blocksR
            start_R = diff_item_R[1]
            length_R = diff_item_R[2]
            if length_R > 0:
                blocksR = pixelsR[start_R:start_R + length_R]
            else:
                blocksR = np.empty((0, width, channels), dtype=np.uint8)
            
            # 计算 blocksL 和blocksR的能够一一对应的行的数量 LR_diff_count
            LR_diff_count = min(length_L, length_R)
            
            # 比较 item_L[2] 与 item_R[2] 的大小
            l_longer_count = 0
            r_longer_count = 0
            
            if length_L > length_R:
                l_longer_count = length_L - LR_diff_count
            elif length_R > length_L:
                r_longer_count = length_R - LR_diff_count
            
            print(f"    LR_diff_count: {LR_diff_count}, l_longer_count: {l_longer_count}, r_longer_count: {r_longer_count}")
            
            # 循环 LR_diff_count 次：对比像素
            for row_idx in range(LR_diff_count):
                # 将blocksL和blocksR的对应行一一进行像素对比
                row_L = blocksL[row_idx]
                row_R = blocksR[row_idx]
                
                # 像素值一样的输出透明像素，不一样的输出红色像素
                diff_row = np.zeros((width, 4), dtype=np.uint8)  # RGBA格式
                
                # 比较每个像素
                pixels_equal = np.all(row_L == row_R, axis=-1)  # 比较所有通道
                
                # 相同像素设为透明
                diff_row[pixels_equal] = [0, 0, 0, 0]  # 透明
                
                # 不同像素设为红色
                diff_row[~pixels_equal] = [255, 0, 0, 255]  # 红色
                
                # 将新的像素行添加到 pixelsMatrix 中
                if pixelsMatrix.shape[0] == 0:
                    pixelsMatrix = diff_row.reshape(1, width, 4)
                else:
                    # 确保通道数匹配
                    if pixelsMatrix.shape[2] != 4:
                        # 转换现有矩阵为RGBA
                        if pixelsMatrix.shape[2] == 3:
                            alpha_channel = np.ones((pixelsMatrix.shape[0], pixelsMatrix.shape[1], 1), dtype=np.uint8) * 255
                            pixelsMatrix = np.concatenate([pixelsMatrix, alpha_channel], axis=2)
                    
                    pixelsMatrix = np.vstack([pixelsMatrix, diff_row.reshape(1, width, 4)])
              # 如果 l_longer_count > 0, 则将 blocksL 最后的一共 l_longer_count 行的像素行执行 色彩值减半然后R值为255 操作后添加到 pixelsMatrix 中
            if l_longer_count > 0:
                # 取出 blocksL 最后的 l_longer_count 行
                l_longer_rows = blocksL[LR_diff_count:LR_diff_count + l_longer_count]
                
                # 执行 色彩值减半然后R值为255 操作
                processed_l_rows = np.zeros((l_longer_count, width, 4), dtype=np.uint8)
                
                # 色彩值减半
                processed_l_rows[:, :, :3] = (l_longer_rows[:, :, :3].astype(np.float32) / 2).astype(np.uint8)
                
                # R值设为255
                processed_l_rows[:, :, 0] = 255
                
                # Alpha通道设为255（不透明）
                processed_l_rows[:, :, 3] = 255
                
                if pixelsMatrix.shape[0] == 0:
                    pixelsMatrix = processed_l_rows
                else:
                    # 确保通道数匹配
                    if pixelsMatrix.shape[2] != 4:
                        if pixelsMatrix.shape[2] == 3:
                            alpha_channel = np.ones((pixelsMatrix.shape[0], pixelsMatrix.shape[1], 1), dtype=np.uint8) * 255
                            pixelsMatrix = np.concatenate([pixelsMatrix, alpha_channel], axis=2)
                    
                    pixelsMatrix = np.vstack([pixelsMatrix, processed_l_rows])
            
            # 如果 r_longer_count > 0, 则将 blocksR 最后的一共 r_longer_count 行的像素行执行 色彩值减半然后B值为255 操作后添加到 pixelsMatrix 中
            if r_longer_count > 0:
                # 取出 blocksR 最后的 r_longer_count 行
                r_longer_rows = blocksR[LR_diff_count:LR_diff_count + r_longer_count]
                
                # 执行 色彩值减半然后B值为255 操作
                processed_r_rows = np.zeros((r_longer_count, width, 4), dtype=np.uint8)
                
                # 色彩值减半
                processed_r_rows[:, :, :3] = (r_longer_rows[:, :, :3].astype(np.float32) / 2).astype(np.uint8)
                
                # B值设为255
                processed_r_rows[:, :, 2] = 255
                
                # Alpha通道设为255（不透明）
                processed_r_rows[:, :, 3] = 255
                
                if pixelsMatrix.shape[0] == 0:
                    pixelsMatrix = processed_r_rows
                else:
                    # 确保通道数匹配
                    if pixelsMatrix.shape[2] != 4:
                        if pixelsMatrix.shape[2] == 3:
                            alpha_channel = np.ones((pixelsMatrix.shape[0], pixelsMatrix.shape[1], 1), dtype=np.uint8) * 255
                            pixelsMatrix = np.concatenate([pixelsMatrix, alpha_channel], axis=2)
                    
                    pixelsMatrix = np.vstack([pixelsMatrix, processed_r_rows])
    
    # 最后，将 pixelsMatrix 转换为PNG图片并保存，文件名为 diff_ouput_{tick时间戳}.png
    if pixelsMatrix.shape[0] > 0:
        # 生成时间戳
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        filename = f"diff_output_{timestamp}.png"
        
        # 转换为PIL图像并保存
        if pixelsMatrix.shape[2] == 4:
            # RGBA模式
            image = Image.fromarray(pixelsMatrix, 'RGBA')
        else:
            # RGB模式
            image = Image.fromarray(pixelsMatrix, 'RGB')
        
        image.save(filename)
        print(f"差异图片已保存为: {filename}")
    else:
        print("没有生成差异图片数据")
    

# 获取不匹配的行信息并给出全局块排序索引值 globalBlockIndex。
# ListDiffL 与 ListDiffR 分别存储左侧和右侧不匹配的行信息并且长度相等。
# ListDiffL 与 ListDiffR 的数据格式为 [[起始行号, 长度], ...]
def GetMatchingInfoBlocks(hashes1, hashes2, listMatch):
    """
    获取不匹配的行信息
    :param hashes1: 第一张图片的行哈希列表
    :param hashes2: 第二张图片的行哈希列表
    :param listMatch: 用于存储匹配行信息的列表
    """
    
    sorted_matching_data = sorted(listMatch, key=lambda x: x[1])
    matchesCount = len(sorted_matching_data)
    
    globalBlockIndex = 0
    
    # 如果没有匹配行，直接将所有行都视为不匹配
    if matchesCount == 0:
        # 如果没有匹配行，直接将所有行都视为不匹配
        print("GetNoMatchingLines: listMatch没有元素，所有行都视为不匹配")
        ListDiffL.append([globalBlockIndex, 0, len(hashes1)])
        ListDiffR.append([globalBlockIndex, 0, len(hashes2)])
        listSame.append([globalBlockIndex + 1, 0, 0])  # 添加一个占位的匹配信息
        return
    
    # 处理头部不匹配部分
    has_head_diff_l = sorted_matching_data[0][1] > 0
    has_head_diff_r = sorted_matching_data[0][2] > 0
    
    if has_head_diff_l or has_head_diff_r:
        print("GetNoMatchingLines: listMatch有头部不匹配部分")
        # 左侧头部不匹配处理
        if has_head_diff_l:
            ListDiffL.append([globalBlockIndex, 0, sorted_matching_data[0][1]])
        else:
            ListDiffL.append([globalBlockIndex, 0, 0])  # 左侧没有不匹配，添加长度为0的占位
        
        # 右侧头部不匹配处理
        if has_head_diff_r:
            ListDiffR.append([globalBlockIndex, 0, sorted_matching_data[0][2]])
        else:
            ListDiffR.append([globalBlockIndex, 0, 0])  # 右侧没有不匹配，添加长度为0的占位
            
        globalBlockIndex += 1
        
    # 处理中间不匹配部分
    if matchesCount == 1:
        # 如果listMatch只有1个元素
        listSame.append([globalBlockIndex, sorted_matching_data[0][1], sorted_matching_data[0][0]])        
        globalBlockIndex += 1
    else:
        # 如果listMatch中有大于1个元素，处理每相邻两个元素之间的部分
        for i in range(1, matchesCount):        
            # 获取当前匹配行和前一个匹配行的信息
            current_match = sorted_matching_data[i]
            previous_match = sorted_matching_data[i - 1]
            
            if i == 1:
                # 如果是第一个匹配行，直接添加到listSame
                listSame.append([globalBlockIndex, previous_match[1], previous_match[0]])
                globalBlockIndex += 1
            
            # 计算当前匹配行的起始行号和前一个匹配行的结束行号
            current_start_l = current_match[1]
            previous_end_l = previous_match[1] + previous_match[0]
            
            current_start_r = current_match[2]
            previous_end_r = previous_match[2] + previous_match[0]
            
            # 左侧不匹配部分
            if current_start_l > previous_end_l:
                ListDiffL.append([globalBlockIndex, previous_end_l, current_start_l - previous_end_l])
            else:
                ListDiffL.append([globalBlockIndex, previous_end_l, 0])
            # 右侧不匹配部分
            if current_start_r > previous_end_r:
                ListDiffR.append([globalBlockIndex, previous_end_r, current_start_r - previous_end_r])
            else:
                ListDiffR.append([globalBlockIndex, previous_end_r, 0])
            
            globalBlockIndex += 1            
            
            listSame.append([globalBlockIndex, current_match[1], current_match[0]])
            globalBlockIndex += 1
            
    # 处理尾部不匹配部分
    has_tail_diff_l = hashes1[-1][0] > sorted_matching_data[-1][1] + sorted_matching_data[-1][0]
    has_tail_diff_r = hashes2[-1][0] > sorted_matching_data[-1][2] + sorted_matching_data[-1][0]
    if has_tail_diff_l or has_tail_diff_r:
        print("GetNoMatchingLines: listMatch有尾部不匹配部分")
        # 左侧尾部不匹配处理
        if has_tail_diff_l:
            ListDiffL.append([globalBlockIndex, sorted_matching_data[-1][1] + sorted_matching_data[-1][0], len(hashes1) - (sorted_matching_data[-1][1] + sorted_matching_data[-1][0])])
        else:
            ListDiffL.append([globalBlockIndex, sorted_matching_data[-1][1] + sorted_matching_data[-1][0], 0])
        # 右侧尾部不匹配处理
        if has_tail_diff_r:
            ListDiffR.append([globalBlockIndex, sorted_matching_data[-1][2] + sorted_matching_data[-1][0], len(hashes2) - (sorted_matching_data[-1][2] + sorted_matching_data[-1][0])])
        else:
            ListDiffR.append([globalBlockIndex, sorted_matching_data[-1][2] + sorted_matching_data[-1][0], 0])
    
    
        

def GetMatchingLines(hashes1, hashes2):
    max_len, l_startIndex, r_startIndex = find_longest_hash_match(hashes1, hashes2)
    l_startLine = hashes1[l_startIndex][0] if l_startIndex < len(hashes1) else -1
    r_startLine = hashes2[r_startIndex][0] if r_startIndex < len(hashes2) else -1
    
    if max_len > 0:
        # 将hashes1和hashes2中匹配的部分添加到listMatch
        listMatch.append([max_len, l_startLine, r_startLine])
        # hashes1和hashes2在match部分的前面不match部分
        if (l_startIndex > 0 and r_startIndex > 0):
            # 如果l_start和r_start都大于0，说明有前面的不匹配部分
            hashes1PreRows = hashes1[0:l_startIndex]
            hashes2PreRows = hashes2[0:r_startIndex]
            GetMatchingLines(hashes1PreRows, hashes2PreRows)
        if (l_startIndex + max_len < len(hashes1) and r_startIndex + max_len < len(hashes2)):
            # hashes1和hashes2在match部分的后面不match部分
            hashes1PostRows = hashes1[l_startIndex + max_len:]
            hashes2PostRows = hashes2[r_startIndex + max_len:]
            GetMatchingLines(hashes1PostRows, hashes2PostRows)
    
def generate_line_hashes(pixel_array):
    """
    生成包含行号和行哈希值的二维列表
    :param pixel_array: 三维NumPy数组 (height, width, channels)
    :return: [[行号, 哈希值], ...]
    """
    result = []
    for y in range(pixel_array.shape[0]):
        row_hash = hashlib.sha256(pixel_array[y].tobytes()).hexdigest()
        result.append([y, row_hash])
    return result

def find_longest_hash_match(lst_l, lst_r):
    """
    找出两个列表中连续哈希值相等的最大连续区间
    参数:
        lst_l: [[行号, 哈希值], ...] 
        lst_r: [[行号, 哈希值], ...]
    返回:
        (最大长度, 左侧匹配起始索引, 右侧匹配起始索引)
    """
    max_len = 0
    l_start = r_start = 0
    i = j = 0
    
    while i < len(lst_l) and j < len(lst_r):
        if lst_l[i][1] == lst_r[j][1]:
            current_len = 1
            # 向后探测匹配长度
            while (i + current_len < len(lst_l) and 
                   j + current_len < len(lst_r) and 
                   lst_l[i + current_len][1] == lst_r[j + current_len][1]):
                current_len += 1
            
            if current_len > max_len:
                max_len = current_len
                l_start = i
                r_start = j
            
            i += current_len
            j += current_len
        elif lst_l[i][1] < lst_r[j][1]:
            i += 1
        else:
            j += 1
    
    return (max_len, l_start, r_start)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image1', help='第一张图片路径')
    parser.add_argument('image2', help='第二张图片路径')
    parser.add_argument('--mode', default='RGB', help='色彩模式')
    #parser.add_argument('--separator', required=True, nargs=3, type=int,
    #                   help='分隔行颜色(R G B)')
    
    args = parser.parse_args()
    #separator_color = tuple(args.separator)
    
    bitmap1 = load_image_as_bitmap(args.image1, args.mode)
    #blocks1 = split_image_with_separators(bitmap1, separator_color)
    
    bitmap2 = load_image_as_bitmap(args.image2, args.mode)
    #blocks2 = split_image_with_separators(bitmap2, separator_color)
    
    #print("图像1块信息:", blocks1)
    #print("图像2块信息:", blocks2)
    
    compire_images(bitmap1, bitmap2)

if __name__ == '__main__':
    main()
