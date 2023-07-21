#To visualize ground truth

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



def plot_results(pil_img, labels, boxes,img_name):
    title1 = 'ground truth'
    fs = 37 #font size
    colr='r' 
    #colr='#FF7FFF' #F691FF #EC6FFF


    h,w,_ = pil_img.shape    
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for label, (x1, y1, x2, y2) in zip(labels, boxes.tolist()):
        ax.add_patch(plt.Rectangle((x1, y1), x2, y2,
                                   fill=False, color=colr, linewidth=4))
        text = f'{label}'
        #ax.text(x1+5, y1+5, text, fontsize=28,
        #        bbox=dict(facecolor='k', alpha=1),color='w')
    ax.text(w*0.993,h*0.01, title1,fontsize=fs,bbox=dict(facecolor='k',alpha=0.85),horizontalalignment='right',verticalalignment='top',color='w')
    #ax.set_title(title2,fontsize=16,y=-0.05)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.axis('off')
    plt.savefig(img_name,bbox_inches = 'tight', pad_inches = 0)
    #plt.show()
    plt.close()


img_sample='/home/yazdi/NEMO_DENSE/val_sets/common_val_set/1_smoke/Axis_Aeneas_08-31-2021_12-47PM.jpeg'
#img_sample='/home/yazdi/NEMO_DENSE/DATSETS/All_images/Triple_Fire_spotted_from_the_Jacks_Peak_fire_camera_at_2_PM_FR-259.jpg'
figsave_path='/home/yazdi/TEMP/gt_sc.png'

boxes= [[1194,538,79,228],[916,515,59,92],[817,540,65,75],[770,510,79,37],[402,495,90,90],[1147,735,14,13]]
boxes = np.array(boxes)
labels = ['low','low','low','low','low','low']
labels=np.array(labels)
orig_image = Image.open(img_sample)
img = np.array(orig_image)


plot_results(img, labels, boxes,figsave_path)
