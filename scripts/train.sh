task="${task:-osod}" # ovd, fsod, osod
vit="${vit:-l}" # s, b, l
dataset="${dataset:-coco}" # coco, lvis
shot="${shot:-30}"
split="${split:-1}"
num_gpus="${num_gpus:-`nvidia-smi -L | wc -l`}"

echo "task=$task, vit=$vit, dataset=$dataset, shot=$shot, split=$split, num_gpus=$num_gpus"

case $task in

    ovd)
    if [[ "$dataset" == "coco" ]]
    then
        python3 tools/train_net.py    --num-gpus $num_gpus  \
            --config-file configs/open-vocabulary/coco/vit${vit}.yaml \
            MODEL.WEIGHTS  weights/initial/open-vocabulary/vit${vit}+rpn.pth \
            DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
            OUTPUT_DIR output/train/open-vocabulary/coco/vit${vit}/ $@
    else
        python3 tools/train_net.py    --num-gpus $num_gpus  --eval-only \
            --config-file  configs/open-vocabulary/lvis/vit${vit}.yaml \
            MODEL.WEIGHTS  weights/initial/open-vocabulary/vit${vit}+rpn_lvis.pth \
            DE.OFFLINE_RPN_CONFIG  configs/RPN/mask_rcnn_R_50_FPN_1x.yaml \
            OUTPUT_DIR output/train/open-vocabulary/lvis/vit${vit}/ $@
    fi
    ;;

    fsod)
    if [[ "$dataset" == "coco" ]]
    then
        #--config-file configs/few-shot/vit${vit}_shot${shot}.yaml  \
        # MODEL.WEIGHTS  weights/initial/few-shot/vit${vit}+rpn.pth \
        #OUTPUT_DIR output/train/few-shot/shot-${shot}/vit${vit}/  $@
        #----------------------------------------------------------------
        #MODEL.WEIGHTS  /root/data1/xfr/CDFSOD/weights/trained/few-shot/vitl_0089999.pth \
        #OUTPUT_DIR output/train/few-shot/shot-${shot}/ft-vit${vit}/  $@

        python3 tools/train_net.py --num-gpus $num_gpus  \
            --config-file configs/few-shot/ft_vit${vit}_shot${shot}.yaml  \
            MODEL.WEIGHTS  /root/data1/xfr/CDFSOD/weights/trained/few-shot/vitl_0089999.pth \
            DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
            OUTPUT_DIR output/train/few-shot/shot-${shot}/ft-vit${vit}/  $@
    else
        python3 tools/train_net.py --num-gpus $num_gpus  \
            --config-file configs/few-shot-voc/${shot}shot/vit${vit}_${split}s.yaml  \
            MODEL.WEIGHTS  weights/initial/few-shot-voc/voc${split}/${vit}+rpn.pth \
            DE.OFFLINE_RPN_CONFIG configs/VOC_RPN/faster_rcnn_R_50_C4.few_shot_s1.yaml \
            OUTPUT_DIR output/train/few-shot-voc/${shot}shot/${split}/vit${vit}/  $@
    fi
    ;;

    osod)
        python3 tools/train_net.py \
            --num-gpus $num_gpus \
            --config-file configs/one-shot/split${split}_vit${vit}.yaml \
            MODEL.WEIGHTS weights/initial/oneshot/vit${vit}+rpn.split${split}.pth \
            DE.OFFLINE_RPN_CONFIG  configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
            OUTPUT_DIR  output/train/one-shot/split${split}/vit${vit}/ \
            DE.ONE_SHOT_MODE  True $@
            ;;
    *)
        echo "skip"
        ;;
esac


source /opt/miniconda3/etc/profile.d/conda.sh
conda activate cdfsod-base
cd /root/data1/xfr/CDFSOD
python train_net.py --config-file configs/artaxor/vitl_shot10_artaxor_finetune-run.yaml MODEL.WEIGHTS /root/data1/xfr/CDFSOD-XFEW/weights/trained/few-shot/vitl_0089999.pth DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml OUTPUT_DIR output/other/


#--num-gpus 2
#--config-file configs/few-shot/ft_vitl_shot10.yaml
#MODEL.WEIGHTS  /root/data1/xfr/CDFSOD/weights/trained/few-shot/vitl_0089999.pth
#DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml
#OUTPUT_DIR output/train/few-shot/tiaoshi/
#--num-gpus 1
#--config-file configs/few-shot/vitl_shot10.yaml
#MODEL.WEIGHTS  weights/initial/few-shot/vitl+rpn.pth
#DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml
#OUTPUT_DIR output/train/few-shot/shot-10/vitl/


--num-gpus 1  \
    --config-file configs/few-shot-voc/${shot}shot/vit${vit}_${split}s.yaml  \
    MODEL.WEIGHTS  weights/initial/few-shot-voc/voc${split}/${vit}+rpn.pth \
    DE.OFFLINE_RPN_CONFIG configs/VOC_RPN/faster_rcnn_R_50_C4.few_shot_s1.yaml \
    OUTPUT_DIR output/train/few-shot-voc/${shot}shot/${split}/vit${vit}/  $@

--num-gpus
1
--config-file
configs/few-shot-voc/10shot/vitl_3s.yaml
MODEL.WEIGHTS
/root/data1/xfr/CDFSOD/weights/trained/few-shot/vitl_0089999.pth
DE.OFFLINE_RPN_CONFIG
configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml
OUTPUT_DIR
output/train/few-shot/tiaoshi/











