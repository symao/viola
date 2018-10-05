import os
import cv2
import numpy as np
import shutil

def create_euroc_stereo_video():
  data_dir = '/home/symao/data/euroc/MH_02_easy/mav0'
  cam0_dir = os.path.join(data_dir, 'cam0/data')
  cam1_dir = os.path.join(data_dir, 'cam1/data')

  writer = None
  for f in sorted(os.listdir(cam0_dir))[800:1300:2]:
    img0 = cv2.imread(os.path.join(cam0_dir, f))
    img1 = cv2.imread(os.path.join(cam1_dir, f))
    if img0 is not None and img1 is not None:
      img_save = cv2.resize(np.hstack((img0, img1)), None, fx=0.5, fy=0.5)
      if writer is None:
        writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_save.shape[1],img_save.shape[0]))
      writer.write(img_save)

  if writer is not None:
    writer.release()

def img2gif(img_list, save_gif, frame_rate = 1):
  # resize imgs to same size, hold aspect ratio
  def imresize(img):
    # return cv2.resize(img, None, fx=0.5, fy=0.5)
    tar_w,tar_h = 400,400
    h,w = img.shape[:2]
    resize_rate = min(tar_w / w, tar_h / h)
    res = cv2.resize(img, None, fx=resize_rate, fy=resize_rate)
    h,w = res.shape[:2]
    pad_h = (tar_h-h)//2
    pad_w = (tar_w-w)//2
    res = cv2.copyMakeBorder(res, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255,255,255))
    print(res.shape)
    return res

  tmp_dir = './temp'
  shutil.rmtree(tmp_dir, ignore_errors=True)
  os.makedirs(tmp_dir)
  save_str = os.path.join(tmp_dir, '%03d.jpg')
  for i,img in enumerate(img_list):
    cv2.imwrite(save_str%i, imresize(img))
  os.system('ffmpeg -f image2 -framerate %d -i %s -loop 100 %s'%(frame_rate, save_str, save_gif))

if __name__ == '__main__':
  datadir = '../doc/img/tag'
  flist = [os.path.join(datadir, f) for f in sorted(os.listdir(datadir))]
  imgs = [cv2.imread(f) for f in flist]
  img2gif(imgs, datadir+'.gif', 2)

  # datadir = '../data/'
  # for i in range(1,5):
  #   img = cv2.imread(os.path.join(datadir, 'bg%d.png'%i))
  #   cv2.imwrite(os.path.join(datadir, 'bg%d.jpg'%i), cv2.resize(img, (400,400)))
