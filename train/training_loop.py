import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import re
from os.path import join as pjoin
from typing import Optional

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from eval import eval_humanml, eval_humanact12_uestc
from sample.generate import main as generate
from data_loaders.get_data import get_dataset_loader
from utils.model_util import load_model_wo_clip
from data_loaders.humanml.scripts.motion_process import get_target_location, sample_goal, get_allowed_joint_options
from utils.sampler_util import ClassifierFreeSampleModel


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.model_avg = None
        if self.args.use_ema:
            self.model_avg = copy.deepcopy(self.model)
        self.model_for_eval = self.model_avg if self.args.use_ema else self.model
        if args.gen_guidance_param != 1:
            self.model_for_eval = ClassifierFreeSampleModel(self.model_for_eval)   # wrapping model with the classifier-free sampler
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        if self.args.use_ema:
            self.opt = AdamW(
                # with amp, we don't need to use the mp_trainer's master_params
                (self.model.parameters()
                 if self.use_fp16 else self.mp_trainer.master_params),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, self.args.adam_beta2),
            )
        else:
            self.opt = AdamW(
                self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
            )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                            split=args.eval_split,
                                            hml_mode='eval',
                                            autoregressive=args.autoregressive,
                                            fixed_len=args.context_len+args.pred_len, pred_len=args.pred_len, device=dist_util.dev())

            self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                                   split=args.eval_split,
                                                   hml_mode='gt', device=dist_util.dev())
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            self.eval_data = {
                'test': lambda: eval_humanml.get_mdm_loader(self.args,
                    self.model_for_eval, diffusion, args.eval_batch_size,
                    gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples, scale=args.gen_guidance_param,
                )
            }
        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            # we add 1 because self.resume_step has already been done and we don't want to run it again
            # in particular we don't want to run the evaluation and generation again
            self.step += 1  
            
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint) 
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                resume_checkpoint, map_location=dist_util.dev())

            if 'model_avg' in state_dict:
                print('loading both model and model_avg')
                state_dict, state_dict_avg = state_dict['model'], state_dict[
                    'model_avg']
                load_model_wo_clip(self.model, state_dict)
                load_model_wo_clip(self.model_avg, state_dict_avg)
            else:
                load_model_wo_clip(self.model, state_dict)
                if self.args.use_ema:
                    # in case we load from a legacy checkpoint, just copy the model
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict(), strict=False)

            # self.model.load_state_dict(
            #     dist_util.load_state_dict(
            #         resume_checkpoint, map_location=dist_util.dev()
            #     ), strict=False
            # )

    def _load_optimizer_state(self):
        main_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )

            if self.use_fp16:
                if 'scaler' not in state_dict:
                    print("scaler state not found ... not loading it.")
                else:
                    # load grad scaler state
                    self.scaler.load_state_dict(state_dict['scaler'])
                    # for the rest
                    state_dict = state_dict['opt']

            tgt_wd = self.opt.param_groups[0]['weight_decay']
            print('target weight decay:', tgt_wd)
            self.opt.load_state_dict(state_dict)
            print('loaded weight decay (will be replaced):',
                  self.opt.param_groups[0]['weight_decay'])
            # preserve the weight decay parameter
            for group in self.opt.param_groups:
                group['weight_decay'] = tgt_wd
            self.opt.param_groups[0]['capturable'] = True

    def cond_modifiers(self, cond, motion):
        # All modifiers must be in-place
        self.target_cond_modifier(cond, motion)
    
    def target_cond_modifier(self, cond, motion):
        if self.args.multi_target_cond:
            batch_size = motion.shape[0]
            cond['target_joint_names'], cond['is_heading'] = sample_goal(batch_size, motion.device, self.args.target_joint_names)

            cond['target_cond'] = get_target_location(motion, 
                                                      self.data.dataset.mean[None, :, None, None], 
                                                      self.data.dataset.std[None, :, None, None], 
                                                      cond['lengths'], 
                                                      self.data.dataset.t2m_dataset.opt.joints_num, self.model.all_goal_joint_names, cond['target_joint_names'], cond['is_heading']).detach()

    def run_loop(self):
        print('train steps:', self.num_steps)
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                    break
                
                self.cond_modifiers(cond['y'], motion) # Modify in-place for efficiency
                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                self.run_step(motion, cond)
                if self.total_step() % self.log_interval == 0:
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.total_step(), v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.total_step(), group_name='Loss')

                if self.total_step() % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    if self.args.use_ema:
                        self.model_avg.eval()
                    self.evaluate()
                    self.generate_during_training()
                    self.model.train()
                    if self.args.use_ema:
                        self.model_avg.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.total_step() > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.total_step() - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.eval_wrapper is not None:
            print('Running evaluation loop: [Should take about 90 min]')
            log_file = os.path.join(self.save_dir, f'eval_humanml_{(self.total_step()):09d}.log')
            diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = eval_humanml.evaluation(
                self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
                replication_times=self.args.eval_rep_times, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)
            print(eval_dict)
            for k, v in eval_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                          iteration=self.total_step(),
                                                          group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=v, iteration=self.total_step(),
                                                      group_name='Eval')

        elif self.dataset in ['humanact12', 'uestc']:
            eval_args = SimpleNamespace(num_seeds=self.args.eval_rep_times, num_samples=self.args.eval_num_samples,
                                        batch_size=self.args.eval_batch_size, device=self.device, guidance_param = 1,
                                        dataset=self.dataset, unconstrained=self.args.unconstrained,
                                        model_path=os.path.join(self.save_dir, self.ckpt_file_name()))
            eval_dict = eval_humanact12_uestc.evaluate(eval_args, model=self.model, diffusion=self.diffusion, data=self.data.dataset)
            print(f'Evaluation results on {self.dataset}: {sorted(eval_dict["feats"].items())}')
            for k, v in eval_dict["feats"].items():
                if 'unconstrained' not in k:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval Unconstrained')

        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval-start_eval)/60}min')


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)

        if self.step % 200 == 0 and hasattr(self.model, 'contact_posterior'):
            print(f"[GRAD CHECK] Step {self.step}")
            found_grad = False
            for name, param in self.model.contact_posterior.named_parameters():
                if param.grad is not None:
                    grad_mean = param.grad.abs().mean().item()
                    grad_max = param.grad.abs().max().item()
                    print(f"  > ContactPosterior ({name}): Mean={grad_mean:.6f} | Max={grad_max:.6f}")
                    found_grad = True
                    break # Just check the first valid parameter to verify flow
            
            if not found_grad:
                print("  > WARNING: No gradients found in ContactPosterior! Check your computation graph.")

        self.update_average_model()
        self._anneal_lr()
        self.log_step()

    def update_average_model(self):
        # update the average model using exponential moving average
        if self.args.use_ema:
            # master params are FP32
            params = self.model.parameters(
            ) if self.use_fp16 else self.mp_trainer.master_params
            for param, avg_param in zip(params, self.model_avg.parameters()):
                # avg = avg + (param - avg) * (1 - alpha)
                # avg = avg + param * (1 - alpha) - (avg - alpha * avg)
                # avg = alpha * avg + param * (1 - alpha)
                avg_param.data.mul_(self.args.avg_model_beta).add_(
                    param.data, alpha=1 - self.args.avg_model_beta)
                
    def calc_virtual_observation_loss(self, x0, contact_logits):
        """ Calculates Expected VO Loss: Sum[ P(c|x0) * Loss(Geometry) ] """
        
        # Get Posterior Probabilities
        probs = torch.softmax(contact_logits, dim=-1)
        
        # L_Heel active if Left(0) is Heel(1) or Both(3)
        w_L_heel = probs[..., 0, 1] + probs[..., 0, 3]
        w_L_toe  = probs[..., 0, 2] + probs[..., 0, 3]
        w_R_heel = probs[..., 1, 1] + probs[..., 1, 3]
        w_R_toe  = probs[..., 1, 2] + probs[..., 1, 3]

        # --- BRANCH A: HumanML3D (hml_vec) ---
        if self.model.data_rep == 'hml_vec':
            from data_loaders.humanml.scripts.motion_process import recover_from_ric
            
            # 1. Recover XYZ of Skeleton
            B, J, F, T = x0.shape
            x0_perm = x0.permute(0, 3, 1, 2).reshape(B, T, J*F)
            
            # --- FIX: Convert NumPy -> Tensor ---
            mean = torch.from_numpy(self.data.dataset.t2m_dataset.mean).float().to(x0.device)
            std = torch.from_numpy(self.data.dataset.t2m_dataset.std).float().to(x0.device)
            # ------------------------------------
            
            x0_unnorm = x0_perm * std + mean
            
            # Recover XYZ joints [B, T, 22, 3]
            x_xyz = recover_from_ric(x0_unnorm.float(), 22)
            
            # 2. Apply Static Offsets (Approximation)
            geo = self.model.geometry_wrapper
            
            # Left Ankle is Index 10, Right is 11
            pos_l_ankle = x_xyz[..., 10, :] 
            pos_l_toe   = pos_l_ankle + geo.offset_l_toe
            pos_l_heel  = pos_l_ankle + geo.offset_l_heel
            
            pos_r_ankle = x_xyz[..., 11, :]
            pos_r_toe   = pos_r_ankle + geo.offset_r_toe
            pos_r_heel  = pos_r_ankle + geo.offset_r_heel

            # 3. Compute Height Loss
            loss_eq = (
                w_L_heel * pos_l_heel[..., 1]**2 + 
                w_L_toe  * pos_l_toe[..., 1]**2 +
                w_R_heel * pos_r_heel[..., 1]**2 + 
                w_R_toe  * pos_r_toe[..., 1]**2
            ).mean()
            
            # 4. Compute Slip Loss
            vel_l = (pos_l_ankle[:, 1:] - pos_l_ankle[:, :-1]).pow(2).sum(dim=-1)
            vel_r = (pos_r_ankle[:, 1:] - pos_r_ankle[:, :-1]).pow(2).sum(dim=-1)
            
            loss_slip = (
                (w_L_heel[:, :-1] + w_L_toe[:, :-1]) * vel_l +
                (w_R_heel[:, :-1] + w_R_toe[:, :-1]) * vel_r
            ).mean()

            is_near_floor_L = (pos_l_heel[..., 1] < 0.03) | (pos_l_toe[..., 1] < 0.03)
            is_near_floor_R = (pos_r_heel[..., 1] < 0.03) | (pos_r_toe[..., 1] < 0.03)
            
            target_L = is_near_floor_L.float().detach()
            target_R = is_near_floor_R.float().detach()

            # Binary Cross Entropy to force probabilities up when close to ground
            # Clamp probabilities to avoid log(0) errors
            w_L_clamped = torch.clamp(w_L_heel + w_L_toe, min=1e-4, max=1-1e-4)
            w_R_clamped = torch.clamp(w_R_heel + w_R_toe, min=1e-4, max=1-1e-4)
            
            loss_heuristic = (
                torch.nn.functional.binary_cross_entropy(w_L_clamped, target_L) +
                torch.nn.functional.binary_cross_entropy(w_R_clamped, target_R)
            )

            if self.step % 100 == 0:
                print(f"\n[PHYSICS CHECK] Step {self.step}")
                # Check Y-axis (index 1) of the first batch, first frame
                h_val_L = pos_l_heel[0, 0, 1].item()
                p_L_contact = (w_L_heel + w_L_toe)[0, 0].item() # Sum heel+toe prob
                
                print(f"  > Left Heel Height: {h_val_L:.4f} m")
                print(f"  > Contact Prob (L): {p_L_contact:.4f} (Target: {target_L[0,0].item()})")
                print(f"  > VO Loss: {loss_eq.item():.6f}")
                print(f"  > Heuristic Loss: {loss_heuristic.item():.6f}")

            return loss_eq + loss_slip + (loss_heuristic * 1.0)
        # --- BRANCH B: SMPL Mesh (Rotations) ---
        else:
            # ... (Previous SMPL logic remains the same) ...
            # Ensure device
            try:
                smpl_device = next(self.model.rot2xyz.smpl_model.parameters()).device
            except StopIteration:
                smpl_device = next(self.model.rot2xyz.smpl_model.buffers()).device
            if smpl_device != x0.device:
                self.model.rot2xyz.smpl_model.to(x0.device)

            vertices = self.model.rot2xyz(
                x0, mask=None, pose_rep=self.model.data_rep, translation=True, glob=True,
                jointstype='vertices', vertstrans=True
            )

            geo = self.model.geometry_wrapper
            landmarks = geo.get_landmarks(vertices)
            heights = geo.compute_heights(landmarks)
            vel_sq = geo.compute_tangential_velocities(landmarks)

            loss_eq = (
                w_L_heel * heights['L_Heel']**2 + w_L_toe * heights['L_Toe']**2 +
                w_R_heel * heights['R_Heel']**2 + w_R_toe * heights['R_Toe']**2
            ).mean()

            loss_slip = (
                w_L_heel[:, :-1] * vel_sq['L_Heel'] + w_L_toe[:, :-1] * vel_sq['L_Toe'] +
                w_R_heel[:, :-1] * vel_sq['R_Heel'] + w_R_toe[:, :-1] * vel_sq['R_Toe']
            ).mean()

        if self.step % 100 == 0:
            # Get probabilities for the first batch item
            probs = torch.softmax(contact_logits, dim=-1)
            p_L_contact = probs[0, 0, 1] + probs[0, 0, 3] # Heel + Both
            
            # Get physical values (Branch A vs B handling)
            if self.model.data_rep == 'hml_vec':
                 # Re-extract for logging if needed, or use variables from Branch A above
                 # Assuming 'pos_l_heel' and 'pos_r_toe' are available from the Branch A block
                 h_val_L = pos_l_heel[0, 0, 1].item() 
                 h_val_R = pos_r_toe[0, 0, 1].item()
            else:
                 # Assuming 'heights' dict is available from Branch B
                 h_val_L = heights['L_Heel'][0, 0].item()
                 h_val_R = heights['R_Toe'][0, 0].item()

            print(f"\n[PHYSICS CHECK] Step {self.step}")
            print(f"  > Left Heel Height: {h_val_L:.4f} (Goal: near 0.0 when contact=1)")
            print(f"  > Right Toe Height: {h_val_R:.4f}")
            print(f"  > Contact Prob (L): {p_L_contact.item():.2f}")
            print(f"  > VO Loss Components: EQ={loss_eq.item():.6f} | SLIP={loss_slip.item():.6f}")

            return loss_eq + loss_slip

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            
            # 1. Sample Timesteps
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # [ELBO MODIFICATION]
            # If using AR Decoder, the Diffusion model only handles t >= 2 (Paper notation).
            # In MDM code (0-indexed), this means t >= 1.
            # We force t to be at least 1, so the diffusion UNet ignores the t=0 step.
            if hasattr(self.model, 'use_ar_decoder') and self.model.use_ar_decoder:
                t = torch.clamp(t, min=1)

            # 2. Compute Standard Diffusion Loss
            # (This is your original logic, preserved)
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            # Base Diffusion Loss
            loss_diffusion = (losses["loss"] * weights).mean()
            
            # 3. [ELBO MODIFICATION] Add Hybrid Terms
            if hasattr(self.model, 'use_ar_decoder') and self.model.use_ar_decoder:
                
                # A. Generate x1 (Noisy state at t=1 / code-index 0)
                t_one = torch.zeros(micro.shape[0], device=dist_util.dev(), dtype=torch.long)
                x1 = self.diffusion.q_sample(micro, t_one)
                
                # Reshape for Transformer [B, S, D]
                # MDM internal: [B, Joints, Feats, Frames] -> [B, Frames, Joints*Feats]
                B, J, F, T = micro.shape
                x0_seq = micro.permute(0, 3, 1, 2).reshape(B, T, J*F)
                x1_seq = x1.permute(0, 3, 1, 2).reshape(B, T, J*F)
                
                # Get Text Embedding
                y_emb = micro_cond['y'].get('text_embed')
                if y_emb is None:
                     y_emb = self.model.encode_text(micro_cond['y']['text'])

                # Check for [1, Batch, Dim] shape (Common in MDM/Transformers)
                if y_emb.ndim == 3 and y_emb.shape[0] == 1:
                    y_emb = y_emb.permute(1, 0, 2)  # [1, B, D] -> [B, 1, D]
                
                # Check for [Batch, Dim] shape
                elif y_emb.ndim == 2:
                    y_emb = y_emb.unsqueeze(1)      # [B, D] -> [B, 1, D]
                
                # B. Posterior Pass (Condition on Clean x0)
                logits_q = self.model.contact_posterior(x0_seq)
                c_dist_q = torch.distributions.Categorical(logits=logits_q)
                c_sample = c_dist_q.sample() # Sample for AR input
                
                # C. Prior Pass (Condition on Noisy x1 + Text)
                logits_p = self.model.contact_prior(x1_seq, y_emb)
                
                # D. KL Divergence Loss: Sum( q * (log q - log p) )
                log_q = torch.log_softmax(logits_q, dim=-1)
                log_p = torch.log_softmax(logits_p, dim=-1)
                loss_kl = (torch.exp(log_q) * (log_q - log_p)).sum(dim=-1).mean()
                
                # E. AR Decoder Loss (NLL)
                # Shift x0 right for teacher forcing
                x0_shifted = torch.cat([torch.zeros_like(x0_seq[:, :1, :]), x0_seq[:, :-1, :]], dim=1)
                pi, mu, sigma = self.model.terminal_decoder(x0_shifted, x1_seq, c_sample, y_emb)
                
                # Calculate NLL (Target is x0_seq)
                loss_term = -self.model.terminal_decoder.mog_head.log_prob(x0_seq, pi, mu, sigma).mean()
                
                # F. Virtual Observation Loss (Geometric Constraints)
                loss_vo = self.calc_virtual_observation_loss(micro, logits_q)

                # Total Loss Summation
                # Note: You may need to tune '0.001' for KL stability
                total_loss = loss_diffusion + loss_term + loss_vo + (loss_kl * 0.001)
                
                # Add to logging dict
                losses['ar_nll'] = loss_term.detach()
                losses['vo'] = loss_vo.detach()
                losses['kl'] = loss_kl.detach()
                
            else:
                # If AR is disabled, just use standard diffusion loss
                total_loss = loss_diffusion

            # 4. Backward Pass & Logging
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(total_loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.total_step() / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.total_step())
        logger.logkv("samples", (self.total_step() + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.total_step()):09d}.pt"

  
    def generate_during_training(self):
        if not self.args.gen_during_training:
            return
        gen_args = copy.deepcopy(self.args)
        gen_args.model_path = os.path.join(self.save_dir, self.ckpt_file_name())
        gen_args.output_dir = os.path.join(self.save_dir, f'{self.ckpt_file_name()}.samples')
        gen_args.num_samples = self.args.gen_num_samples
        gen_args.num_repetitions = self.args.gen_num_repetitions
        gen_args.guidance_param = self.args.gen_guidance_param
        gen_args.motion_length = 6  # fixed length
        gen_args.input_text = gen_args.text_prompt = gen_args.action_file = gen_args.action_name = gen_args.dynamic_text_path = ''
        if gen_args.multi_target_cond:
            gen_args.sampling_mode = 'goal'
            gen_args.target_joint_source = 'data'
        all_sample_save_path = generate(gen_args)
        self.train_platform.report_media(title='Motion', series='Predicted Motion', iteration=self.total_step(),
                                         local_path=all_sample_save_path)        

    
    def find_resume_checkpoint(self) -> Optional[str]:
        '''look for all file in save directory in the pattent of model{number}.pt
            and return the one with the highest step number.

        TODO: Implement this function (alredy existing in MDM), so that find model will call it in case a ckpt exist.
        TODO: Change call for find_resume_checkpoint and send save_dir as arg.
        TODO: This means ignoring the flag of resume_checkpoint in case some other ckpts exists in that dir!
        '''

        matches = {file: re.match(r'model(\d+).pt$', file) for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}

        return pjoin(self.args.save_dir, models[max(models)]) if models else None
    
    def total_step(self):
        return self.step + self.resume_step
    
    def save(self):
        def save_checkpoint():
            def del_clip(state_dict):
                # Do not save CLIP weights
                clip_weights = [
                    e for e in state_dict.keys() if e.startswith('clip_model.')
                ]
                for e in clip_weights:
                    del state_dict[e]

            if self.use_fp16:
                state_dict = self.model.state_dict()
            else:
                state_dict = self.mp_trainer.master_params_to_state_dict(
                    self.mp_trainer.master_params)
            del_clip(state_dict)

            if self.args.use_ema:
                # save both the model and the average model
                state_dict_avg = self.model_avg.state_dict()
                del_clip(state_dict_avg)
                state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint()

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.total_step()):09d}.pt"),
            "wb",
        ) as f:
            opt_state = self.opt.state_dict()
            if self.use_fp16:
                # with fp16 we also save the state dict
                opt_state = {
                    'opt': opt_state,
                    'scaler': self.scaler.state_dict(),
                }

            torch.save(opt_state, f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()



def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
