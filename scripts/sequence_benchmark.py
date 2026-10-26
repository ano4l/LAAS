"""Prepare independent review footage or score labelled events/possession."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def prepare(sources,directory):
    from workers.video.real_analysis import video_metadata
    import cv2
    clips=[]
    (directory/'videos').mkdir(parents=True,exist_ok=True)
    for source in sources:
        meta=video_metadata(source);digest=hashlib.sha256(source.read_bytes()).hexdigest()
        target=directory/'videos'/source.name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise ValueError('Existing benchmark video does not match source; use a new output directory')
        if not target.exists():
            try:os.link(source,target)
            except OSError:shutil.copy2(source,target)
        labels=dict(schema_version='sequence-benchmark-1',source_sha256=digest,duration_seconds=meta['duration_seconds'],
            annotation_source='unreviewed',reviewer='',split='development',match_group='unverified-match',reviewed_windows=[],
            kit_mappings={},events=[],possession_intervals=[])
        label_path=directory/f'{source.stem}-labels.json'
        if label_path.exists():
            labels=json.loads(label_path.read_text(encoding='utf-8'))
            if labels.get('source_sha256')!=digest:raise ValueError('Existing labels do not match source')
        else:label_path.write_text(json.dumps(labels,indent=2),encoding='utf-8')
        clips.append(dict(id=source.stem,filename=source.name,video='videos/'+source.name,labels=labels))
        # Uniform frames chosen without looking at any prediction output.
        frames=[];frame_dir=directory/'ball'/source.stem;frame_dir.mkdir(parents=True,exist_ok=True)
        cap=cv2.VideoCapture(str(source))
        try:
            for i in sorted({round(n*(meta['frame_count']-1)/23) for n in range(24)}):
                cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read()
                if not ok:raise ValueError('Cannot decode benchmark frame')
                name=f'frame-{i:08d}.png';cv2.imwrite(str(frame_dir/name),frame)
                frames.append(dict(frame_idx=i,image=name,reviewed=False,boxes=[]))
        finally:cap.release()
        ball_labels=dict(schema_version=1,video_sha256=digest,width=meta['width'],height=meta['height'],annotation_source='unreviewed',frames=frames)
        if not (frame_dir/'labels.json').exists():(frame_dir/'labels.json').write_text(json.dumps(ball_labels,indent=2),encoding='utf-8')
        else:ball_labels=json.loads((frame_dir/'labels.json').read_text(encoding='utf-8'))
        template=Path(__file__).with_name('ball_review.html').read_text(encoding='utf-8')
        (frame_dir/'review.html').write_text(template.replace('__LABELS_JSON__',json.dumps(ball_labels).replace('<','\\u003c')),encoding='utf-8')
    collection=dict(schema_version='sequence-benchmark-collection-1',clips=clips)
    (directory/'collection.json').write_text(json.dumps(collection,indent=2),encoding='utf-8')
    template=Path(__file__).with_name('sequence_review.html')
    if template.exists():(directory/'review.html').write_text(template.read_text(encoding='utf-8').replace('__BENCHMARK_JSON__',json.dumps(collection).replace('<','\\u003c')),encoding='utf-8')
    return collection


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);cmd=p.add_subparsers(dest='command',required=True)
    prep=cmd.add_parser('prepare');prep.add_argument('directory',type=Path);prep.add_argument('videos',type=Path,nargs='+')
    score=cmd.add_parser('score');score.add_argument('labels',type=Path);score.add_argument('run',type=Path);score.add_argument('--allow-visual-pilot',action='store_true');score.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.command=='prepare':
        result=prepare(args.videos,args.directory);print(f"Prepared {len(result['clips'])} UNREVIEWED clips and uniformly selected ball frames. No predictions copied into labels.")
    else:
        from workers.video.benchmark import score_sequences
        manifest=json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
        result=score_sequences(json.loads(args.labels.read_text(encoding='utf-8')),manifest['metrics']['video_insights'],manifest['metadata'],allow_visual_pilot=args.allow_visual_pilot)
        if args.output:args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
