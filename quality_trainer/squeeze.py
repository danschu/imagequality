import cv2
import numpy as np

def get_default_squeeze_size():
    return (480, 270)

def squeeze(img, expected_width=None, expected_height=None, block_cnt_x = 7, block_cnt_y = 7):
    if expected_width is None:
        expected_width, expected_height = get_default_squeeze_size()
    
    img_h, img_w, c = img.shape
    factor = img_h/img_w
    
    cnt_x = block_cnt_x
    cnt_y = block_cnt_y

    in_scale_x = 28//cnt_x
    in_scale_y = 28//cnt_y
    
    if expected_height is None:
        expected_height = int(expected_width*factor)

    block_size_w = int(expected_width//cnt_x)
    block_size_h = int(block_size_w*factor)
    
    new_img_width = cnt_x*block_size_w 
    new_img_height = cnt_y*block_size_h 

    img_w_new = ((cnt_x+1)//2+(cnt_x//2)*in_scale_x)*block_size_w
    img_h_new = ((cnt_y+1)//2+(cnt_y//2)*in_scale_y)*block_size_h

    orig_block_cnt_x = (cnt_x+1)//2
    orig_block_cnt_y = (cnt_y+1)//2

    orig_block_cnt_space_w = orig_block_cnt_x*block_size_w
    orig_block_cnt_space_h = orig_block_cnt_y*block_size_h

    res_space_w = (img_w_new-orig_block_cnt_space_w)
    res_space_h = (img_h_new-orig_block_cnt_space_h)

    scale_step_x = res_space_w//(cnt_x//2)
    scale_step_y = res_space_h//(cnt_y//2)
    
    small_img = np.zeros((new_img_height, new_img_width, 3), np.uint8)
    
    img = cv2.resize(img, (img_w_new, img_h_new))
    
    
    for y in range(0, cnt_y):
        for x in range(0, cnt_x):
            left_small = x*block_size_w
            top_small = y*block_size_h
            right_small = left_small+block_size_w
            bottom_small = top_small+block_size_h
         
            if x%2==0:
                left_big = (x//2)*(block_size_w+scale_step_x)
                right_big = left_big+block_size_w
            else:
                left_big = (x//2)*(block_size_w+scale_step_x)+block_size_w
                right_big = left_big+scale_step_x
                
            if y%2==0:
                top_big = (y//2)*(block_size_h+scale_step_y)     
                bottom_big = top_big+block_size_h
            else:
                top_big = (y//2)*(block_size_h+scale_step_y)+block_size_h  
                bottom_big = top_big+scale_step_y
                
            small_img[top_small:bottom_small, left_small:right_small, :] = cv2.resize(img[top_big:bottom_big, left_big:right_big,:], (block_size_w, block_size_h))
            
    small_img = cv2.resize(small_img, (expected_width, expected_height))
    return small_img