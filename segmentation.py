import os
import glob
import random
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.structures.instances import  Instances
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
from skimage import img_as_ubyte
from Lama_mask_gen import MixedMaskGenerator

def build_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    panoptic_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    predictor = DefaultPredictor(cfg)
    return predictor,panoptic_metadata


def get_seg_mask(im,predictor,category_dict):
    outputs = predictor(im)
    scores = outputs['instances'].scores
    class_idx = list(outputs["instances"].pred_classes.cpu().numpy())
    if len(class_idx) >0:
        pred_classes = [category_dict.get(idx) for idx in class_idx]
    else:
        pred_classes = []
    mask_tensor = outputs["instances"].pred_masks
    mask_tensor = mask_tensor.to('cpu')
    score_threhold = 0.8
    score_mask = scores > score_threhold
    selected_scores = torch.masked_select(scores,score_mask)
    obj_num = selected_scores.size()[0]
    if obj_num >= 1:
        selected_indices = get_qualify_obj(scores, mask_tensor,pred_classes, score_threshold=0.5, mask_threshold=0.005)
        print('qualified obj_num: ', len(selected_indices))
        if len(selected_indices) >= 1:
            _,mask,_ = generate_seg_mask_cv(mask_tensor,im,selected_indices)

            return mask,outputs,len(selected_indices),True
        else:
            print('Found objects but not qualified.')
            return None,len(selected_indices),False
    else:
        print('No detectable foreground object!Save as background image!')
        return im,False

def generate_seg_mask_cv(mask_tensor,img,selected_indices):
    obj_num,h,w = mask_tensor.size()
    im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    im_t = torch.from_numpy(im.transpose(2,0,1))
    init_mask = torch.full_like(im_t,0)
    for i in range(obj_num):
        if i in selected_indices:
            init_mask.masked_fill_(mask_tensor[i:i+1],1)
            init_mask = torch.clamp(init_mask,0,1)

    threhold = 0.5
    one = init_mask >= threhold
    zero = init_mask < threhold
    init_mask.masked_fill_(one,1)
    mask_t = init_mask.masked_fill_(zero,0)
    mask_np = mask_t.numpy().transpose(1,2,0)
    mask_np = (mask_np * 255).astype(np.uint8)
    img = im_t.numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    masked_img = img * (255 -mask_np)

    return img,mask_np,masked_img

def checkDir(dir):
    if isinstance(dir,list):
        for d in dir:
            if not os.path.exists(d):
                os.makedirs(d)
    else:
        if not os.path.exists(dir):
            os.makedirs(dir)

def get_qualify_obj(scores,masks,pred_classes,score_threshold = 0.8,mask_threshold=0.1,exclude_type=[]):
    qualified_obj_idx = []
    for i in range(scores.shape[0]):
        score = scores[i].item()
        if score >= score_threshold:
            if pred_classes[i] not in exclude_type:
                mask_0 = torch.zeros(size=masks[i:i + 1].size())
                mask = mask_0.masked_fill_(masks[i:i + 1],1)
                mask_ratio = np.mean(mask)
                if mask_ratio >= mask_threshold:
                    qualified_obj_idx.append(i)

    return qualified_obj_idx


#input mask : 0 for background ,1 for missing area
def get_transparent_mask(im,mask,color=(255,0,0)):    #(220,20,60)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernel)
    mask = mask[:,:,0:1]
    threshold = 0.5
    mask = np.where(mask>=threshold,1,0)
    R = np.expand_dims(np.ones(mask.shape[:2]) * color[0],axis=2)
    G = np.expand_dims(np.ones(mask.shape[:2]) * color[1],axis=2)
    B = np.expand_dims(np.ones(mask.shape[:2])* color[2],axis=2)
    color_M = np.concatenate((R,G,B),axis=2)
    color_im = color_M.astype(np.uint8)
    masked_color_im = color_im * mask
    masked_color_im = masked_color_im.astype(np.uint8)

    alpha = 1     #img1 transparent ratio
    beta  = 0.55     #img2 transparent ratio
    gamma = 0       #adjust value add to img

    masked_img = cv2.addWeighted(im,alpha,masked_color_im,beta,gamma)
    masked_img = masked_img.astype(np.uint8)
    # masked_img =Image.fromarray(masked_img)
    # masked_img.show()

    mask = (mask * 255).astype(np.uint8)
    mask = np.concatenate((mask,mask,mask),axis=2)
    # mask = Image.fromarray(mask)

    return masked_img,mask

def dilated_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    iter = np.random.randint(5,10)
    mask = cv2.dilate(mask, kernel, iterations=iter)  # generate more aggressive mask
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = mask[:, :, 0:1]
    threshold = 0.5
    mask = np.where(mask >= threshold, 1, 0)
    mask = (mask * 255).astype(np.uint8)
    mask = np.concatenate((mask, mask, mask), axis=2)
    # mask = Image.fromarray(mask)
    # mask.show()
    return mask

def paste_mask2im(srcImg,objBackgroundImg,objSie=512):
    srcImg = cv2.imread(srcImg)
    objBackgroundImg = cv2.imread(objBackgroundImg)
    srcImg = cv2.resize(srcImg,(objSie,objSie))
    objImg = cv2.resize(objBackgroundImg,(objSie,objSie))
    mask,status = get_seg_mask(srcImg)
    if status:
        srcImg = cv2.cvtColor(srcImg,cv2.COLOR_BGR2RGB)
        objImg = cv2.cvtColor(objImg,cv2.COLOR_BGR2RGB)
        mask = (mask * 255).astype(np.uint8)
        comp_img =objImg * (1 - mask) + srcImg * mask
        masked_im = objImg * (1 - mask) + mask*255
        show_src = Image.fromarray(srcImg, mode="RGB")
        # show_src.show()
        show_obj = Image.fromarray(objImg, mode="RGB")
        # show_obj.show()
        show_mask = Image.fromarray(mask * 255, mode="RGB")
        # show_mask.show()
        show_comp = Image.fromarray(comp_img, mode="RGB")
        # show_comp.show()
        show_masked_im = Image.fromarray(masked_im, mode="RGB")
        # show_masked_im.show()

        return show_comp,masked_im

def get_file_info(path):
    dir_path, file_full_name = os.path.split(path)
    file_name, file_type = os.path.splitext(file_full_name)

    return {"dir_path": dir_path, "file_name": file_name, "file_type": file_type}


def visualize_anno_img(img,predict_output,metadata):
    v = Visualizer(img[:, :, ::-1], metadata, scale=1.2)
    out = v.draw_instance_predictions(predict_output["instances"].to("cpu"))
    return out.get_image()

def find_mask_contour(mask):
    mask = (mask > 0).astype(np.uint8)
    # calculate center from mask
    cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(cnts,key=cv2.contourArea)

    # compute the center of the contour
    #only return the center of maximum contour
    M = cv2.moments(max_cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cnt_center = (cX, cY)

    return cnts,cnt_center

def draw_text(img,text,start_point=None):
    if start_point == None:
        _,start_point = find_mask_contour(img)
    img_with_text = cv2.putText(img, text, start_point, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    return img_with_text

try:
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    DETECTRON_INSTALLED = True
except:
    print("Detectron v2 is not installed")
    DETECTRON_INSTALLED = False

class MySegMaskGen():
    def __init__(self,hole_range=[0.0,0.7],max_obj_area = 0.5,confidence_threshold=0.5):
        assert DETECTRON_INSTALLED, 'Cannot use SegmentationMask without detectron2'
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        self.predictor = DefaultPredictor(self.cfg)
        self.hole_range = hole_range
        self.max_obj_area = max_obj_area
        bg_cfg = {
            'irregular_proba': 1,
            'hole_range': [0.0, 1.0],
            'irregular_kwargs': {
                'max_angle': 4,
                'max_len': 250,
                'max_width': 150,
                'max_times': 3,
                'min_times': 1,
            },
            'box_proba': 0,
            'box_kwargs': {
                'margin': 10,
                'bbox_min_size': 30,
                'bbox_max_size': 150,
                'max_times': 4,
                'min_times': 1,
            }
        }
        self.bg_mask_gen = MixedMaskGenerator(**bg_cfg)

    def get_segmentation(self, img):
        im = img_as_ubyte(img)
        panoptic_seg, segment_info = self.predictor(im)["panoptic_seg"]
        return panoptic_seg, segment_info

    def identify_candidates(self, panoptic_seg, segments_info):
        potential_mask_ids = []
        for segment in segments_info:
            if not segment["isthing"]:
                continue
            mask = (panoptic_seg == segment["id"]).int().detach().cpu().numpy()
            area = mask.sum().item() / np.prod(panoptic_seg.shape)
            if area >= self.max_obj_area:
                continue
            potential_mask_ids.append(segment["id"])
        return potential_mask_ids

    def get_mask(self,im,single_obj=False,objRemoval=True):
        panoptic_seg, segments_info = self.get_segmentation(im)
        potential_mask_ids = self.identify_candidates(panoptic_seg, segments_info)
        mask_set = []
        for mask_id in potential_mask_ids:
            mask = (panoptic_seg == mask_id).int().detach().cpu().numpy()
            if not np.any(mask):
                continue

            mask = mask.astype(np.uint8)
            if objRemoval:
                mask = self.dilated_mask(mask)  #make sure the mask cover the whole object
            mask_set.append(mask)

        if single_obj:
            fg_mask = np.zeros(shape=im.shape[:2])
            max_try = 10
            i = 0
            while np.mean(fg_mask) >= self.hole_range[1]:
                idx = np.random.randint(low=0,high=len(mask_set))
                fg_mask = mask_set[idx]
                i += 1
                if i >max_try:
                    return mask_set[0]
        else:
            np.random.shuffle(mask_set)
            mask = np.zeros(shape=im.shape[:2])
            for m in mask_set:
                area = np.mean(mask + m)
                if area <= self.hole_range[1]:
                    mask += m
                else:
                    break

            fg_mask = mask

        fg_mask = fg_mask > 0
        if objRemoval:
            return fg_mask.astype(np.uint8)
        else:
            mask = self.get_bg_mask(fg_mask)
            while np.mean(mask) < self.hole_range[0] or np.mean(mask) > self.hole_range[1]:
                mask = self.get_bg_mask(fg_mask)

            return mask

    def get_bg_mask(self,fg_mask):
        bg_mask = self.bg_mask_gen(shape=fg_mask.shape)
        bg_mask = (bg_mask > 0).astype(np.uint8)
        fg_mask = (fg_mask >0).astype(np.uint8)
        mask = bg_mask + fg_mask
        mask = (mask > 0).astype(np.uint8)
        mask = mask - fg_mask
        mask = (mask > 0).astype(np.uint8)
        return mask

    def dilated_mask(self,mask):
        kernel = np.ones((3, 3), np.uint8)
        iter = np.random.randint(5, 10)
        mask = cv2.dilate(mask, kernel, iterations=iter)  # generate more aggressive mask
        return mask

    def __call__(self, im):
        if np.random.binomial(1, 0.5) > 0:
            objRemoval = True
            if np.random.binomial(1, 0.5) > 0:
                single_obj = True
            else:
                single_obj = False
        else:
            objRemoval = False
            single_obj = False

        mask = self.get_mask(im,single_obj=single_obj,objRemoval=objRemoval)

        return mask

class MySegMaskGen_():
    def __init__(self,hole_range=[0.0,0.7],max_obj_area = 0.3,max_obj_num= 3,confidence_threshold=0.8,exclude_types=[],obj_threshold=0.8):
        assert DETECTRON_INSTALLED, 'Cannot use SegmentationMask without detectron2'
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        self.obj_threshold = obj_threshold
        self.meta = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.class_names = self.meta.thing_classes
        self.exclude_ids = [self.class_names.index(t) for t in exclude_types]
        self.predictor = DefaultPredictor(self.cfg)
        self.hole_range = hole_range
        self.max_obj_area = max_obj_area
        self.max_obj_num = max_obj_num
        bg_cfg = {
            'irregular_proba': 1,
            'hole_range': [self.hole_range[0], self.hole_range[1]],
            'irregular_kwargs': {
                'max_angle': 4,
                'max_len': 100,
                'max_width': 50,
                'max_times': 5,
                'min_times': 4,
            },
            'box_proba': 0.3,
            'box_kwargs': {
                'margin': 0,
                'bbox_min_size': 10,
                'bbox_max_size': 50,
                'max_times': 5,
                'min_times': 1,
            }
        }
        self.bg_mask_gen = MixedMaskGenerator(**bg_cfg)
        self.seg_status = False

    def get_segmentation(self, img,dilated_mask=True):
        im = img_as_ubyte(img)
        pred_results = self.predictor(im)
        instances = pred_results["instances"]
        masks_set, new_instances = self.identify_candidates(instances)
        for i, mask in enumerate(masks_set):
            mask = mask.astype(np.uint8)
            if dilated_mask:
                mask = self.dilated_mask(mask)  # make sure the mask cover the whole object
            masks_set[i] = mask
        return masks_set, new_instances

    def identify_candidates(self, instances):
        instances = instances.to('cpu')
        new_instances = None
        scores = instances.scores
        pred_classes = instances.pred_classes
        pred_masks = instances.pred_masks
        potential_mask_ids = []
        masks_set = []

        for i in range(len(instances)):
            if not scores[i] > self.obj_threshold:
                continue

            if pred_classes[i] in self.exclude_ids:
                continue

            mask = (pred_masks[i]).int().detach().cpu().numpy()
            if np.mean(mask) >= self.max_obj_area:
                continue

            potential_mask_ids.append(i)
            masks_set.append(mask)

        if len(potential_mask_ids) > 0 :
            new_instances = self.get_instances(potential_mask_ids,instances)

        return masks_set,new_instances

    def get_instances(self,obj_ids,instances):
        new_instances = Instances(image_size=instances.image_size)
        fields = instances.get_fields()
        for k, v in fields.items():
            if not isinstance(v,torch.Tensor):
                v = v.tensor
            new_instances.set(k, torch.cat([v[i:i + 1] for i in obj_ids]))

        return new_instances

    def visualize_anno_img(self,img, instances, metadata):
        v = Visualizer(img[:, :, ::-1], metadata, scale=1.2)
        out = v.draw_instance_predictions(instances)
        return out.get_image()

    def get_fg_mask(self,im):
        masks_set, instances = self.get_segmentation(im)
        if masks_set == []:
            return None, None

        target_idxs = []
        mask = np.zeros(shape=im.shape[:2])
        max_objs = min(self.max_obj_num,len(masks_set))
        objs_num = np.random.randint(1,max_objs) if max_objs > 1 else 1
        idxs = random.sample(range(len(masks_set)),objs_num)
        mask_dict = {i: masks_set[i] for i in idxs}
        for idx, m in mask_dict.items():
            area = np.mean(mask + m)
            if area > self.hole_range[1]:
                break
            mask += m
            target_idxs.append(idx)

        # visualize the selected masks
        vis_instances = self.get_instances(target_idxs, instances)
        vis_img = self.visualize_anno_img(img, instances=vis_instances, metadata=self.meta)
        fg_mask = mask > 0
        return fg_mask.astype(np.uint8),vis_img

    def get_bg_mask(self,shape,fg_mask=None):
        bg_mask = self.bg_mask_gen(shape=shape)
        bg_mask = (bg_mask > 0).astype(np.uint8)
        if isinstance(fg_mask,np.ndarray):
            fg_mask = (fg_mask >0).astype(np.uint8)
            mask = bg_mask + fg_mask
            mask = (mask > 0).astype(np.uint8)
            mask = mask - fg_mask
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = bg_mask

        return mask

    def dilated_mask(self,mask):
        kernel = np.ones((3, 3), np.uint8)
        iter = np.random.randint(5, 10)
        mask = cv2.dilate(mask, kernel, iterations=iter)  # generate more aggressive mask
        return mask

    def __call__(self, im):
        if np.random.binomial(1, 0.5) > 0:
            mask = self.get_fg_mask(im)
        else:
            fg_mask,_ = self.get_fg_mask(im)
            mask = self.get_bg_mask(shape=im.shape,fg_mask=fg_mask)

        return mask



if __name__ == '__main__':
    import os
    all_instance_type = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                          'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                          'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                          'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                          'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                          'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                          'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                          'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                          'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                          'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    dataName = 'place_512_test_background'
    target_size = 512
    img_dir = '/home/codeoops/CV/data/test_large'
    save_dir = f'/home/codeoops/CV/data/{dataName}'
    vis_mask_save_dir = save_dir + '/visual_masked_im'
    mask_save_dir = save_dir + '/object_masks'
    background_save_dir = save_dir + '/background'
    img_with_obj_save_dir = save_dir +'/objects'
    checkDir([mask_save_dir,vis_mask_save_dir,background_save_dir,img_with_obj_save_dir])
    max_im_count = 20000
    imgs_path = glob.glob(img_dir+'/*.jpg') + glob.glob(img_dir+'/*.png')
    confidence_threshold = 0.6
    # random.shuffle(imgs_path)
    obj_imgs = []
    backgrounds = []
    exclude_types = ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                          'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                          'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                          'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                          'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                          'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                          'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                          'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                          'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                          'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    seg_mask_gen = MySegMaskGen_(max_obj_area = 0.99, confidence_threshold = 0.6, exclude_types=exclude_types)
    im_idx = 0
    no_idx = 0
    for fpathe, dirs, fs in os.walk(img_dir):

        for f in fs:
            print(f"process the NO {no_idx} img")
            im_path = os.path.join(fpathe, f)
            img = cv2.imread(im_path)
            # tmp = im_path.split('/')
            # folder = tmp[-4] + '_' + tmp[-3] + '_' + tmp[-2]
            # save_path = os.path.join(background_save_dir, folder)

            if not os.path.exists(background_save_dir):
                os.makedirs(background_save_dir)

            if (img.shape[-1] != 3):
                continue
            img = np.array(Image.fromarray(img).resize(size=(target_size, target_size)))
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_name = get_file_info(im_path)['file_name']
            try:
                mask,vis_img = seg_mask_gen.get_fg_mask(np.array(img))
            except Exception as e:
                pass
            else:
                if isinstance(mask,np.ndarray):
                    print('img_name:',img_name)
                    mask = np.expand_dims(mask, axis=-1)
                    mask = np.concatenate([mask, mask, mask], axis=-1)
                    mask = (mask * 255).astype(np.uint8)
                    obj_imgs.append(im_path)
                    Image.fromarray(mask).save(mask_save_dir+f'/{img_name}.png')
                    Image.fromarray(imgRGB).save(img_with_obj_save_dir+f'/{img_name}.jpg')
                    Image.fromarray(vis_img).save(vis_mask_save_dir + f'/{img_name}_vis.jpg')
                    im_idx += 1
                    if im_idx >= max_im_count:
                        break
                else:

                    if (imgRGB.shape[-1] == 3):
                        Image.fromarray(imgRGB).save(background_save_dir+ f'/{im_idx:0>8d}.jpg')

                    im_idx += 1

            no_idx += 1

