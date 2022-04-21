# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:31:20 2020

@author: ycp99
"""
import os
import PSNR
import argparse
import graphing

parser = argparse.ArgumentParser(description='test PSNR AND SSIM')

parser.add_argument('--path', default="./eval/epochweights/backupx2/",  help="put results here")
parser.add_argument('--HR', default="./Input/validation/", help="put HR images here")

opt = parser.parse_args()

def search(txt):
    typeoffile = ['.txt','.jpg','.png','.bmp']
    print(txt)
    
    for str in typeoffile:
        if txt.find(str)!=-1: 
            return False
            
    return True
    
def calALL():
    
    path = opt.path
    
    
    ps = []
    ss = []
    
    last_PSNR =[]
    last_SSIM =[]
    
    list_PSNR =[]
    list_SSIM =[]
    
    file_list = os.listdir(path)
 
    f = open(path+"average.txt", 'w')
    
    f.write('average (PSNR,SSIM)\n')
    f.close()
    
    file_list = sorted(file_list, key = len)
    
    for Npath in file_list:
        
        newPath = path+Npath+'/'
        averPSNR = 0
        averSSIM = 0
        count = 0

        if not (search(newPath)):
            continue 
        else:            
            f = open(newPath+"newPSNR.txt", 'w')
            f.close()
    
            f = open(newPath+"newSSIM.txt", 'w')
            f.close()
            
            print("*folder* = "+newPath)
            
            newfile_list = os.listdir(newPath)
            
            if(count ==0): last_PSNR = list_PSNR
            
            if(count ==0): last_SSIM = list_SSIM
            
            list_PSNR =[]
            list_SSIM =[]
            
            for i in newfile_list:
             
                if(i[-9:]=='epoch.jpg'):
                    os.remove(newPath+'epoch.jpg')
                    continue
                
                if(i[-3:]=='txt' or i[-8:]=='SSIM.jpg' or i[-8:]=='PSNR.jpg'):
                    continue
                
                img = i
                LR = newPath + img
                print('open '+ LR)
                
                HR = opt.HR
                HR = HR + img
                print('open '+HR)
                one, two = PSNR.cal_PSNRandSSIM(HR, LR)
                print(f"{one} is PSNR, {two} is SSIM")
                
                f = open(newPath+"newPSNR.txt", 'a')
                f.write(img+f" PSNR is {one}\n")
                f.close()
    
                f = open(newPath+"newSSIM.txt", 'a')
                f.write(img+f" SSIM is {two}\n")
                f.close()
         
                list_PSNR.append(one)
                list_SSIM.append(two) 
                
                #well.. im not good at python..
                averPSNR += one
                averSSIM += two
                
                count +=1 
                print('=============================')        
            
            graphing.graph(list_PSNR, last_PSNR, newPath,'PSNR','Vali_pics')
            graphing.graph(list_SSIM, last_SSIM, newPath,'SSIM','Vali_pics')                
            
            
            
            if(count==0):
                averPSNR = 0
                averSSIM = 0
            else:
                averPSNR /= count
                averSSIM /= count
            
            print("{} is average PSNR, {} is average SSIM".format(averPSNR,averSSIM))
            
            ps.append(averPSNR)
            ss.append(averSSIM)
            
            graphing.graph(ps, [], path,'PSNR','epoches')
            graphing.graph(ss, [], path,'SSIM','epoches')
            
            f = open(path+"average.txt", 'a')
            f.write("{} : {} is average PSNR, {} is average SSIM\n".format(Npath,averPSNR,averSSIM))
            f.close()
       
       
       
        

calALL()
