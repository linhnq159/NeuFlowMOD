"""Detection behavior, segmentation compatibility and short real-model integration checks."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch

from NeuFlow.model_builder import ModelBuilder
from NeuFlow.config import DEFAULT_TRAIN_AUGMENTATION
from data_utils.frame_box_dataset import FrameBoxDataset, FrameBoxAugmentor, read_yolo, collate_frame_boxes
from tool.checkpoints import load_checkpoint, resolve_task, transfer_model_weights, load_model_weights
from tool.detection import DetectionLoss, DetectionStats, decode_detections, make_targets
from tool.train import Trainer, parse_args, run_training
from tool.infer import parse_args as pair_args, run_inference as infer_pair
from tool.infer_video import parse_args as video_args, run_inference as infer_video
from tool.export_seg import ExportModel, export_model, compare_outputs
from tool.training_log import select_preview_samples
from checks.check_frame_training import QuietProgress
from checks.check_training_logging import RecordingWriter
from data_utils.frame_pair import init_frame_worker


def numpy_box_collate(batch):
    # NumPy transport avoids file-descriptor sharing in restricted test sandboxes.
    return [(a.numpy(), b.numpy(), boxes.numpy()) for a, b, boxes in batch]


def create_dataset(root, count=8, size=(47, 35), sequence='seq'):
    images, labels = root / 'images' / sequence, root / 'labels' / sequence
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        rng = np.random.default_rng(i)
        Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(images / f'frame_{i}.png')
        (labels / f'frame_{i}.txt').write_text('0 0.5 0.5 0.2 0.4\n' if i % 2 else '')
    return images, labels


def outputs(h=4, w=5, batch=1):
    return dict(center_logits=torch.full((batch, 1, h, w), -20.), size=torch.ones(batch, 2, h, w),
                offset=torch.full((batch, 2, h, w), .5))


class DetectionDatasetChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images, self.labels = create_dataset(self.root)

    def test_labels_resize_empty_missing_and_past(self):
        (self.labels / 'frame_2.txt').unlink()
        dataset = FrameBoxDataset(self.root/'images', self.root/'labels', img_size=(94, 70), frame_skip=2)
        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset.samples[0][0].stem, 'frame_3')
        self.assertEqual(dataset.samples[1][1].stem, 'frame_2')
        current, past, boxes = dataset[0]
        self.assertEqual(current.shape, (3, 70, 94))
        self.assertEqual(past.shape, current.shape)
        torch.testing.assert_close(boxes, torch.tensor([[37.6, 21., 56.4, 49.]]))
        self.assertEqual(dataset[1][2].shape, (0, 4))
        batch = collate_frame_boxes([dataset[0], dataset[1]])
        self.assertEqual([len(b) for b in batch[2]], [1, 0])

    def test_temporal_split_no_shared_frames(self):
        train = FrameBoxDataset(self.root/'images', self.root/'labels', split='train', val_split=.5)
        val = FrameBoxDataset(self.root/'images', self.root/'labels', split='val', val_split=.5)
        self.assertFalse({p for s in train.samples for p in s[:2]} & {p for s in val.samples for p in s[:2]})
        self.assertEqual(val.samples[0][0].stem, 'frame_5')

    def test_invalid_labels(self):
        for text in ('1 .5 .5 .1 .1', '0 nan .5 .1 .1', '0 .5 .5 0 .1', '0 .9 .5 .4 .1',
                     '0 .5 .5 .1', '0 .5 .5 .1 .1 3', 'x .5 .5 .1 .1'):
            with self.subTest(text=text):
                path = self.labels/'invalid.txt'
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, 'Invalid YOLO'):
                    read_yolo(path)

    def test_preview_selection_persists_and_val_not_augmented(self):
        dataset = FrameBoxDataset(self.root/'images', self.root/'labels', split='val', val_split=.75)
        selected, identities = select_preview_samples(dataset, 4)
        self.assertEqual(len(selected), 4)
        self.assertTrue(any(not len(dataset.annotations[i]) for i in selected))
        self.assertEqual(select_preview_samples(dataset, 4, identities)[0], selected)
        self.assertIsNone(dataset.augmentor)

    def test_native_batch_sizes_rejected(self):
        image, past, labels = FrameBoxDataset(self.root/'images', self.root/'labels')[0]
        with self.assertRaisesRegex(ValueError, '--resize'):
            collate_frame_boxes([(image, past, labels), (image[:, :-1], past[:, :-1], labels)])

    def test_worker_seed_reproducibility(self):
        cfg=dict(DEFAULT_TRAIN_AUGMENTATION, enabled=['scale','hflip','brightness_contrast'])
        dataset=FrameBoxDataset(self.root/'images',self.root/'labels',augmentation=cfg)
        for workers in (0,2):
            runs=[]
            for repeat in range(2):
                torch.manual_seed(1211)
                loader=torch.utils.data.DataLoader(dataset,batch_size=2,num_workers=workers,
                    worker_init_fn=init_frame_worker,collate_fn=numpy_box_collate,
                    generator=torch.Generator().manual_seed(1211),shuffle=True)
                runs.append([sample for batch in loader for sample in batch])
            for a,b in zip(*runs):
                for x,y in zip(a,b): np.testing.assert_array_equal(x,y)


class BoxAugmentationChecks(unittest.TestCase):
    def setUp(self):
        self.current = torch.arange(3*16*16).reshape(3,16,16).float() % 200
        self.past = self.current + 30
        self.boxes = torch.tensor([[2., 4., 10., 12.]])

    def aug(self, **options):
        return FrameBoxAugmentor(dict(DEFAULT_TRAIN_AUGMENTATION, **options))

    def test_shrink_shared_geometry_padding_and_no_mutation(self):
        current_before, boxes_before = self.current.clone(), self.boxes.clone()
        with patch('data_utils.frame_box_dataset._offset', return_value=4):
            current, past, boxes = self.aug()._scale(self.current, self.past, self.boxes, .5)
        torch.testing.assert_close(boxes, torch.tensor([[5.,6.,9.,10.]]))
        torch.testing.assert_close(past[:,4:12,4:12]-current[:,4:12,4:12], torch.full((3,8,8),30.))
        torch.testing.assert_close(current[:,0,0], past[:,0,0])
        torch.testing.assert_close(current_before, self.current)
        torch.testing.assert_close(boxes_before, self.boxes)

    def test_missing_box_rolls_back_only_scale(self):
        boxes = torch.tensor([[0.,0.,2.,2.], [12.,12.,16.,16.]])
        aug = self.aug(enabled=['scale','hflip','brightness_contrast'], zoom_out_p=.6, zoom_in_p=.3,
                       hflip_p=1., brightness_contrast_p=1.)
        with patch('torch.rand', return_value=torch.tensor(.7)), \
                patch('data_utils.frame_box_dataset._uniform', side_effect=[2.,1.1,1.]), \
                patch('data_utils.frame_box_dataset._offset', return_value=16):
            current, past, target = aug(self.current,self.past,boxes)
        torch.testing.assert_close(current, (self.current.flip(-1)*1.1).clamp(0,255))
        torch.testing.assert_close(past, (self.past.flip(-1)*1.1).clamp(0,255))
        torch.testing.assert_close(target, torch.tensor([[14.,0.,16.,2.], [0.,12.,4.,16.]]))

    def test_partial_crop_and_empty(self):
        with patch('data_utils.frame_box_dataset._offset', return_value=8):
            _,_,boxes = self.aug()._scale(self.current,self.past,torch.tensor([[6.,6.,14.,14.]]),2.)
            current,past,empty = self.aug()._scale(self.current,self.past,torch.empty(0,4),2.)
        torch.testing.assert_close(boxes, torch.tensor([[4.,4.,16.,16.]]))
        self.assertEqual(empty.shape,(0,4))
        self.assertEqual(current.shape,self.current.shape)
        self.assertFalse(torch.equal(current,self.current))

    def test_repeatable_seed(self):
        aug=self.aug(enabled=['scale','hflip','brightness_contrast'])
        torch.manual_seed(42); a=aug(self.current,self.past,self.boxes)
        torch.manual_seed(42); b=aug(self.current,self.past,self.boxes)
        for x,y in zip(a,b): torch.testing.assert_close(x,y)


class DetectionLossChecks(unittest.TestCase):
    def test_positive_gradient_and_empty_regression(self):
        out={k:v.requires_grad_() for k,v in outputs(batch=2).items()}
        boxes=[torch.tensor([[8.,4.,12.,8.]]),torch.empty(0,4)]
        loss=DetectionLoss()(out,boxes)
        self.assertTrue(torch.isfinite(loss['total']))
        loss['total'].backward()
        self.assertLess(out['center_logits'].grad[0,0,1,2].item(),0)
        self.assertGreater(out['center_logits'].grad[1].sum().item(),0)
        self.assertEqual(out['size'].grad[1].abs().sum().item(),0)
        self.assertEqual(out['offset'].grad[1].abs().sum().item(),0)

    def test_collision_keeps_small_box(self):
        heatmap,objects,count=make_targets(torch.tensor([[4.,4.,12.,12.],[7.,7.,9.,9.]]),8,8,'cpu')
        self.assertEqual(count,1)
        self.assertEqual(objects[(2,2)][:2],(.5,.5))
        self.assertEqual(int((heatmap==1).sum()),1)

    def test_real_model_gradient_padding_and_unused_layers(self):
        torch.manual_seed(7)
        model=ModelBuilder('flow_rgb',task='detection').train()
        a,b=torch.rand(2,3,35,47)*255,torch.rand(2,3,35,47)*255
        with patch.object(model.seg_head.transposed5_conv,'forward',side_effect=AssertionError('mask decoder ran')):
            out=model(a,b,iters_s8=1)
            loss=DetectionLoss()(out,[torch.tensor([[5.,5.,12.,12.]]),torch.empty(0,4)])
            loss['total'].backward()
        self.assertEqual(out['center_logits'].shape,(2,1,9,12))
        self.assertGreater(model.detection_head.center.weight.grad.abs().sum().item(),0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone_op.parameters()))
        self.assertTrue(all(p.grad is None for p in model.seg_head.transposed5_conv.parameters()))
        model.eval()
        with torch.no_grad(): result=model(a,b,iters_s8=1)
        self.assertEqual(result['offset'].shape,(2,2,9,12))


class DetectionMetricsChecks(unittest.TestCase):
    def test_decode_original_coordinates(self):
        out=outputs();out['center_logits'][0,0,1,2]=10;out['size'][0,:,1,2]=torch.tensor([2.,1.])
        result=decode_detections(out,(20,16),(40,48))[0]
        self.assertEqual(len(result),1)
        torch.testing.assert_close(result[0,:4],torch.tensor([12.,12.,28.,24.]))
        self.assertEqual(result[0,5],0)

    def test_padding_nms_and_limits(self):
        out=outputs(h=9,w=12)
        out['center_logits'][0,0,8,11]=20;out['offset'][0,:,8,11]=.99
        out['center_logits'][0,0,1,1]=10;out['center_logits'][0,0,1,4]=9
        out['size'][:]=20
        self.assertEqual(len(decode_detections(out,(47,35))[0]),1)
        self.assertEqual(len(decode_detections(out,(47,35),max_detections=1)[0]),1)
        out['center_logits'][:]=-20
        self.assertEqual(len(decode_detections(out,(47,35))[0]),0)
        out['size'][0,0,0,0]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'non-finite'):
            decode_detections(out,(47,35))

    def test_ap_perfect_duplicate_false_positive_and_empty(self):
        gt=torch.tensor([[0.,0.,4.,4.]])
        pred=torch.tensor([[0.,0.,4.,4.,.9,0.]])
        stats=DetectionStats();stats.update([pred],[gt]);self.assertEqual(stats.values()['map'],1.)
        stats.update([torch.tensor([[2.,2.,5.,5.,.95,0.]])],[torch.empty(0,4)])
        values=stats.values()
        self.assertEqual(values['map'],.5)
        self.assertEqual(values['precision'],.5)
        self.assertEqual(values['recall'],1.)
        self.assertEqual(values['empty_false_positives'],1)
        stats=DetectionStats();stats.update([torch.cat((pred,pred))],[gt])
        self.assertEqual(stats.values()['precision'],.5)
        self.assertEqual(stats.values()['map50'],1.)
        stats=DetectionStats();stats.update([pred],[torch.empty(0,4)])
        self.assertIsNone(stats.values()['map'])

    def test_ap_localizes_and_counts_misses(self):
        gt=torch.tensor([[0.,0.,4.,4.],[10.,10.,14.,14.]])
        stats=DetectionStats();stats.update([torch.tensor([[0.,0.,4.,4.,.9,0.]])],[gt])
        self.assertAlmostEqual(stats.values()['map'],51/101)
        stats=DetectionStats();stats.update([torch.tensor([[0.,0.,4.,3.,.9,0.]])],[gt[:1]])
        self.assertEqual(stats.values()['map50'],1.)
        self.assertAlmostEqual(stats.values()['map'],.6)


class DetectionIntegrationChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.images,self.labels=create_dataset(self.root)
        self.base=['--data_path',str(self.root),'--task','detection','--work_dir',str(self.root/'runs'),
                   '--device','cpu','--resize','47','35','--batch_size','2','--num_workers','0',
                   '--epochs','2','--warm_up_epochs','0','--no-amp','--no-tensorboard',
                   '--backbone_op_pretrained','','--stabilization','none','--iters_s8','1','--val_split','.5']

    def test_task_resolution_and_invalid_options(self):
        self.assertEqual(resolve_task(checkpoint={}), 'segmentation')
        with self.assertRaisesRegex(ValueError,'conflicts'):
            resolve_task('detection',{})
        for extra in (['--loss','CUSTOM'],['--teacher','teacher.pth'],['--conf_threshold','nan'],['--max_detections','0']):
            with self.subTest(extra=extra),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parse_args(self.base+extra)

    def test_transfer_strict_shared_weights_and_no_decoder_optimizer(self):
        old=ModelBuilder('flow');new=ModelBuilder('flow_rgb','detection')
        transfer_model_weights(new,dict(model=old.state_dict(),args={'seg_input':'flow'}))
        torch.testing.assert_close(new.seg_head.regular4_2.ext_conv1[0].weight,old.seg_head.regular4_2.ext_conv1[0].weight)
        with self.assertRaisesRegex(ValueError,'conflicts'):
            load_model_weights(new,dict(model=old.state_dict(),args={'seg_input':'flow'}))
        trainer=Trainer(parse_args(self.base));self.addCleanup(trainer.close)
        optimized={id(p) for group in trainer.optimizer.param_groups for p in group['params']}
        self.assertFalse(any(id(p) in optimized for p in trainer.model.seg_head.transposed5_conv.parameters()))

    def test_training_resume_history_inference_and_video(self):
        with patch('tool.detection_training.tqdm',QuietProgress):
            trainer=Trainer(parse_args(self.base))
            trainer.training(1);improved=trainer.validation(1);trainer.save_checkpoint(1,improved)
            root=trainer.save_folder3;checkpoint=trainer.save_pth3/'last.pth'
            preview=list(trainer.tb_image_samples)
            trainer.close()
            saved=load_checkpoint(checkpoint)
            self.assertEqual(saved['loss_version'],'centernet_v1')
            self.assertEqual(saved['args']['task'],'detection')
            self.assertTrue((root/'checkpoint'/'best_map.pth').is_file())
            row=json.loads((root/'history.jsonl').read_text().splitlines()[0])
            self.assertNotIn('train_iou',row)
            self.assertEqual(row['train_boxes'],1)
            self.assertEqual(row['train_samples'],2)
            self.assertEqual(row['val_samples'],3)
            self.assertAlmostEqual(row['train_loss_total'],row['train_loss_center']+.1*row['train_loss_size']+row['train_loss_offset'],places=5)
            resumed=Trainer(parse_args(self.base+['--resume',str(checkpoint),'--save_freq','1']))
            self.assertEqual(resumed.args.start_epoch,2)
            self.assertEqual(resumed.tb_image_samples,preview)
            resumed.training(2);resumed.save_checkpoint(2)
            self.assertTrue((resumed.history.session_dir/'checkpoint'/'epoch_000002.pth').is_file())
            resumed.close()
        rows=[json.loads(line) for line in (root/'history.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),2)
        self.assertNotEqual(rows[0]['session_id'],rows[1]['session_id'])
        self.assertIsNone(rows[1]['val_map'])
        args=pair_args(['--checkpoint',str(checkpoint),'--current',str(self.images/'frame_3.png'),
                       '--past',str(self.images/'frame_2.png'),'--output',str(self.root/'boxes.png'),
                       '--device','cpu','--resize','32','32','--conf_threshold','0','--max_detections','5'])
        detections,_=infer_pair(args)
        report=json.loads((self.root/'boxes.json').read_text())
        self.assertEqual(report['image_size'],[47,35]);self.assertEqual(len(report['detections']),len(detections))
        self.assertEqual(Image.open(self.root/'boxes.png').size,(47,35))
        video=video_args(['--checkpoint',str(checkpoint),'--input',str(self.images),'--output',str(self.root/'out.avi'),
                          '--device','cpu','--max_frames','3','--conf_threshold','0','--max_detections','5'])
        report=infer_video(video)
        self.assertEqual(report['frames_written'],3);self.assertEqual(report['predicted_frames'],2)
        self.assertIn('box_scores',(self.root/'out.timings.csv').read_text())
        with self.assertRaisesRegex(ValueError,'mask_dir'):
            infer_video(video_args(['--checkpoint',str(checkpoint),'--input',str(self.images),
                                   '--output',str(self.root/'bad.avi'),'--mask_dir',str(self.root/'masks')]))

    def test_tb_frequency_no_extra_forward_and_scalar_fields(self):
        with patch('tool.detection_training.tqdm',QuietProgress):
            trainer=Trainer(parse_args(self.base));self.addCleanup(trainer.close)
            trainer.writer=writer=RecordingWriter()
            trainer.training(1)
            with patch.object(trainer,'_forward',wraps=trainer._forward) as forward:
                improved=trainer.validation(1)
                self.assertEqual(forward.call_count,2)
                self.assertEqual(len(writer.images),3)
                trainer.save_checkpoint(1,improved)
                trainer.best_map=trainer.best_map50=2.
                trainer.validation(2)
                self.assertEqual(len(writer.images),3)
                trainer.validation(10);trainer.validation(10)
                self.assertEqual(len(writer.images),6)
                trainer.best_map=-1.
                trainer.validation(11)
                self.assertEqual(len(writer.images),9)
                self.assertEqual(forward.call_count,10)
            self.assertIn('Epoch/train_loss_center',[tag for tag,_,_ in writer.scalars])
            self.assertNotIn('Epoch/train_iou',[tag for tag,_,_ in writer.scalars])
            self.assertFalse(list(trainer.save_folder3.rglob('*.png')))

    def test_amp_skip_bookkeeping_and_backbone_lr(self):
        class SkipScaler:
            value=16.
            def is_enabled(self): return False
            def scale(self,loss): return loss
            def get_scale(self): return self.value
            def step(self,optimizer): pass
            def update(self): self.value/=2
        with patch('tool.detection_training.tqdm',QuietProgress):
            trainer=Trainer(parse_args(self.base+['--epoch_training_backbone','1','--backbone_layer_lr','.01']))
            self.addCleanup(trainer.close)
            trainer.training(1)
            self.assertAlmostEqual(trainer.train_epoch_stats['lr_backbone']/trainer.train_epoch_stats['lr_head'],.01)
            trainer.scaler=SkipScaler()
            step=trainer.lr_scheduler.last_epoch
            trainer.training(2)
            self.assertEqual(trainer.lr_scheduler.last_epoch,step)
            self.assertEqual(trainer.train_epoch_stats['optimizer_steps'],0)
            self.assertEqual(trainer.train_epoch_stats['optimizer_skipped_steps'],1)
            self.assertEqual(trainer.train_epoch_stats['lr_head'],0)

    @unittest.skipUnless(importlib.util.find_spec('tensorboard'),'optional TensorBoard unavailable')
    def test_real_tensorboard_events(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with patch('tool.detection_training.tqdm',QuietProgress):
            trainer=Trainer(parse_args(self.base+['--tensorboard']))
            trainer.training(1);improved=trainer.validation(1);trainer.save_checkpoint(1,improved)
            folder=trainer.save_folder3/'tensorboard'/trainer.session_id
            trainer.close()
        events=EventAccumulator(str(folder)).Reload()
        self.assertIn('Epoch/train_loss_center',events.Tags()['scalars'])
        self.assertTrue(events.Tags()['images'])

    def test_video_invalid_contour_option_and_json_protection(self):
        model=ModelBuilder(task='detection')
        checkpoint=self.root/'model.pth'
        torch.save(dict(model=model.state_dict(),args={'task':'detection','img_size':[32,32],'stabilization':'none'}),checkpoint)
        with self.assertRaisesRegex(ValueError,'contour'):
            infer_video(video_args(['--checkpoint',str(checkpoint),'--input',str(self.images),
                                   '--output',str(self.root/'out.avi'),'--draw_boxes']))
        json_checkpoint=self.root/'annotated.json'
        json_checkpoint.write_bytes(checkpoint.read_bytes())
        before=json_checkpoint.read_bytes()
        with self.assertRaisesRegex(ValueError,'JSON'):
            infer_pair(pair_args(['--checkpoint',str(json_checkpoint),'--current',str(self.images/'frame_3.png'),
                                  '--past',str(self.images/'frame_2.png'),'--output',str(self.root/'annotated.png')]))
        self.assertEqual(before,json_checkpoint.read_bytes())

    def test_disabled_validation_and_interrupt_preserve_completed_history(self):
        original=Trainer.training
        def train_then_interrupt(trainer,epoch):
            if epoch==2: raise KeyboardInterrupt()
            return original(trainer,epoch)
        with patch('tool.detection_training.tqdm',QuietProgress),patch.object(Trainer,'training',train_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_training(parse_args(self.base+['--val_split','0']))
        root=next((self.root/'runs').iterdir())
        summary=json.loads((root/'summary.json').read_text())
        self.assertEqual(summary['status'],'interrupted');self.assertEqual(summary['last_completed_epoch'],1)
        row=json.loads((root/'history.jsonl').read_text())
        self.assertIsNone(row['val_map']);self.assertIsNone(row['val_loss_center'])

    def test_export_wrapper_three_outputs_and_numerical_comparison(self):
        model=ModelBuilder('flow_rgb','detection').eval()
        wrapper=ExportModel(model,32,32,iters_s8=1)
        current,past=torch.rand(1,3,32,32)*255,torch.rand(1,3,32,32)*255
        with torch.no_grad(): raw=model(current,past,iters_s8=1);exported=wrapper(current,past)
        values=dict(zip(('center_logits','size','offset'),[x.numpy() for x in exported]))
        check=compare_outputs({k:v.numpy() for k,v in raw.items()},values,(32,32))
        self.assertTrue(check['passed'])

    def test_detection_trace_on_two_frame_pairs(self):
        model=ModelBuilder('flow_rgb','detection').eval()
        wrapper=ExportModel(model,32,32,iters_s8=1)
        a,b=torch.rand(1,3,32,32)*255,torch.rand(1,3,32,32)*255
        with torch.no_grad():
            traced=torch.jit.trace(wrapper,(a,b))
            for current,past in ((a,b),(a,torch.roll(a,1,-1))):
                for expected,actual in zip(wrapper(current,past),traced(current,past)):
                    torch.testing.assert_close(expected,actual)

    @unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'),'optional ONNX packages unavailable')
    def test_real_onnx_export(self):
        model=ModelBuilder('flow_rgb','detection').eval()
        metadata=export_model(model,self.root/'detection.onnx',(32,32),iters_s8=1)
        self.assertEqual(metadata['output_names'],['center_logits','size','offset'])
        self.assertTrue(all(check['passed'] for check in metadata['validation']))


if __name__=='__main__':
    torch.set_num_threads(1);cv2.setNumThreads(1)
    unittest.main()
