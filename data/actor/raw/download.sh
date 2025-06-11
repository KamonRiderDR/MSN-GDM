#!/bin/bash  
###
 # @Description: 
 # @Author: Rui Dong
 # @Date: 2024-04-18 20:42:45
 # @LastEditors: Rui Dong
 # @LastEditTime: 2024-04-18 20:44:09
### 

# 设定文件的基本URL和路径  
BASE_URL="https://github.com/bingzhewei/geom-gcn/blob/master/splits/film_split_0.6_0.2_"  
FILE_SUFFIX=".npz"  

# 循环下载文件  
for i in {0..9}  
do  
    # 构造完整的文件URL  
    FILE_URL="${BASE_URL}${i}${FILE_SUFFIX}"  
    
    # 使用wget下载文件  
    wget "$FILE_URL"  
      
    # # 检查wget的退出状态，如果非0则打印错误信息并退出脚本  
    # if [ $? -ne 0 ]; then  
    #     echo "Error downloading $FILE_URL"  
    #     exit 1  
    # fi  
done