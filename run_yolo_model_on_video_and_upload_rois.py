#! /usr/bin/env python

import argparse
import os
from pprint import pprint
from tqdm import tqdm
from pathlib import Path

import tator


class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('MEDIA', help='the data/video you want to process')
    parser.add_argument('MODEL', help='the yolo model (weights) to apply to MEDIA')

    parser.add_argument('--cache_offline', action='store_true', help='if results should be cached locally')
    parser.add_argument('--local_project', required=False, type=str, help='local project directory to use if caching offline')

    tator_args = parser.add_argument_group(title='Tator Parameters', description=None)
    tator_args.add_argument('--host', default='https://tator.whoi.edu', help='Default is "https://tator.whoi.edu"')
    tator_args.add_argument('--token', required=True, help='Tator user-access token (required)')
    tator_args.add_argument('--project', default='2', help='Default is "2" for the WARP fish detection project')
    tator_args.add_argument('--media_type', type=int, required=True, help='2 = CUREE videos, 3 = Diver Survey, 4 = Diver Images')
    tator_args.add_argument('--version_id', type=int, default=2, help='')  # which layer.
    tator_args.add_argument('--localization_type', default='2', help='Default is "2" for Animal BBox localization_type id')  # ROI

    tator_args.add_argument('--class-attribute', default='Class', help='Default is the localization_type "Class" attribute)')
    tator_args.add_argument('--media_id', type=int, help='[optional] tator media id reference')

    tator_args.add_argument('--frame-offset', type=int, default=0, help='Frame number offset. Eg: -1. Default is 0')
    tator_args.add_argument('--skip-title-frame', action='store_true', help='If invoked, the first frame of the video will be skipped')

    # from yolov5/detect.py
    model_args = parser.add_argument_group(title='Model Parameters', description=None)
    model_args.add_argument('--classlist', dest='model_classlist', required=True, help='A file with the ordered list of class names, newlines deliminated')
    model_args.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    model_args.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    model_args.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    model_args.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    model_args.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    model_args.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    model_args.add_argument('--half', action='store_true', help='use FP16 half-precision inference')

    args = parser.parse_args()
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand

    model_classlist = []
    with open(args.model_classlist) as f:
        for l in f:
            l = l.rstrip()
            if l:
                model_classlist.append(l)
    args.model_classlist = model_classlist
    return args


def tator_args_str2id(args, api):
    # 1) convert any strings to IDs by looking them up on tator for project, media_type, and localization_type

    # PROJECT
    projects_avail = {d.name:d.id for d in api.get_project_list()}
    pprint(projects_avail)
    if args.project.isnumeric():
        args.project = int(args.project)
    else:
        assert args.project in projects_avail, f'Project Type "{args.project}" not in: {", ".join(projects_avail)}'
        args.project = projects_avail[args.project]
    project_name = {v:k for k,v in projects_avail.items()}[args.project]

    # MEDIA TYPE
    # mediatypes_avail = {d.name:d.id for d in api.get_media_type_list(args.project)}
    # pprint(mediatypes_avail)
    # if args.media_type.isnumeric():
    #    args.media_type = int(args.media_type)
    # else:
    #    assert args.media_type in mediatypes_avail, f'Media Type "{args.media_type}" not in: {", ".join(mediatypes_avail)}'
    #    args.media_type = mediatypes_avail[args.media_type]

    # LOCALIZATION TYPE
    localizationtypes_avail = {d.name:d.id for d in api.get_localization_type_list(args.project)}
    pprint(localizationtypes_avail)
    if args.localization_type.isnumeric():
        args.localization_type = int(args.localization_type)
    else:
        assert args.localization_type in localizationtypes_avail, f'Localization Type "{args.localization_type}" not in: {", ".join(localizationtypes_avail)}'
        args.localization_type = localizationtypes_avail[args.localization_type]

    # 2) check that all do infact exist
    project = api.get_project(args.project)
    #media_type = api.get_media_type(args.media_type)
    localization_type = api.get_localization_type(args.localization_type)
    args.roi_classes = [attrib.choices for attrib in localization_type.attribute_types if attrib.name==args.class_attribute][0]
    args.roi_classes_type = [attrib.dtype for attrib in localization_type.attribute_types if attrib.name==args.class_attribute][0]

    # 3) if MEDIA exists on tator, note media ID
    if args.media_id:
        media_obj = api.get_media(id=args.media_id)
        MEDIA_name = os.path.splitext(media_obj.name)[0]
    else:
        MEDIA_name = os.path.splitext(os.path.basename(args.MEDIA))[0]
        media_obj = api.get_media_list(args.project, name=Path(args.MEDIA).name)
        assert len(media_obj) != 0, f'MEDIA name "{MEDIA_name}" was not found in project {project_name} ({args.project}). Has it been uploaded yet?'
        assert len(media_obj) == 1, f'MEDIA name "{MEDIA_name}" has duplicates in project {project_name} ({args.project}). Select one from these IDs using --media_id: {",".join([str(d.id) for d in media_obj])}'
        media_obj = media_obj[0]
        #pprint(media_obj)
    args.media_id = media_obj.id

    # 4) check that MEDIA is a real/accessible path. if not, check tator media for src file path.
    assert os.path.exists(args.MEDIA), f'MEDIA "{args.MEDIA}" is not a valid file.'
    print('MEDIA_ID:',args.media_id)
    # TODO check media_obj for a source attribute to go looking up the correct piece of backend/original media

    return args


def run_model(args):
    from yolov5 import detect
    #preds = detect.run('path/to/model_weights.pt', 'path/to/source.video', device='cuda', return_preds_only=True)

    if args.cache_offline:
        return_preds_only = False
        nosave = False
        save_txt = True
        save_conf = True
        project = args.local_project
    else:
        return_preds_only = True
        nosave = True
        save_txt = False
        save_conf = False
        project = "yolov5/runs/detect" # relative path from this script to yolo directory + runs/detect is the default

    preds = detect.run(
        args.MODEL, args.MEDIA, return_preds_only=return_preds_only,
        device=args.device, imgsz=args.imgsz, half=args.half,
        conf_thres=args.conf_thres, iou_thres=args.iou_thres,
        max_det=args.max_det, agnostic_nms=args.agnostic_nms,
        project=project, nosave=nosave, save_txt=save_txt, save_conf=save_conf
    )

    return preds


def format_preds(preds, args):

    spec_list = []
    classes = set()
    spec_invalid = []
    for pred in preds:
        frame,c,x,y,w,h,score = pred
        frame = int(frame)+args.frame_offset
        x,y,w,h = float(x),float(y),float(w),float(h)
        x = x-w/2
        y = y-h/2
        model_class = args.model_classlist[c]

        classes.add(model_class)
        d = hashabledict({
                'media_id': args.media_id,
                'type':     args.localization_type,
                'version':  args.version_id,
                'frame':    frame,
                'x':        x,
                'y':        y,
                'width':    w,
                'height':   h,
                'Verified': False,
                'ModelScore': float(score),
                'ModelName':  args.MODEL,
                args.class_attribute: model_class,  # 'Class'
                })
        if frame <= 0 and args.skip_title_frame: continue
        elif not args.roi_classes_type == "string" and model_class not in args.roi_classes:
            spec_invalid.append(d)
            continue

        spec_list.append(d)

    classes = sorted(classes)
    print('Model Classes Found')
    for c in classes:
        print(f'  {c}')

    print()
    if spec_invalid:
        print('ERROR: the following model_classes are not valid ROI classes. {len(spec_invalid)} ROIs were skipped')
        for c in sorted({d['Class'] for d in spec_invalid}):
            print(f'  {c}')
        print('\nValid ROI classes are:')
        for c in args.roi_classes:
            print(f'  {c}')
        print()

    # check / eliminate duplicates
    print(f'{len(spec_list)} vs. set({len(set(spec_list))})')

    spec_list = sorted(set(spec_list), key=lambda d:(d['media_id'],d['frame'],d['x'],d['y']))
    return spec_list


def upload_ROIs(spec_list,api,args):
    created_ids = []
    print('Uploading!')
    uploader = tator.util.chunked_create(api.create_localization_list, args.project, localization_spec=spec_list)
    for response in tqdm(uploader):
        created_ids += response.id
    print()
    print(f"Created {len(created_ids)} localizations!")
    return created_ids



if __name__ == '__main__':

    # 0) inputs: model, video, tator_configs
    args = get_args()
    api = tator.get_api(args.host, args.token)

    # 1) check video exists on tator. If not, then upload?
    args = tator_args_str2id(args, api)
    if not args.media_id:
        raise NotImplemented('video not found on tator. upload it first')

    # 2) run model on video, output bboxes
    preds = run_model(args)

    # 3) format preds for tator
    roi_specs = format_preds(preds, args)
    #pprint(roi_specs)

    # 4) bulk upload detections to tator, w/ new version number
    roi_ids = upload_ROIs(roi_specs, api, args)


    print('Done')






