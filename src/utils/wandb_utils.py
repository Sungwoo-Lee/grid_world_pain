import wandb
import os

def upload_video(video_path, run_path=None, step=None, episode=None, caption="Evaluation Video", fps=4):
    """
    Uploads a video to WandB.
    
    Args:
        video_path (str): Path to the video file.
        run_path (str, optional): If provided, connects to this specific run (entity/project/id) to upload.
                                  If None, assumes a WandB run is already active.
        step (int, optional): The global step to associate with the video. 
                              Note: If uploading to an existing run, logging to a past step is ignored by WandB.
                              It's recommended to leave this None for existing runs (appends to end) and rely on 'episode'.
        episode (int, optional): The episode number to log as a metric (eval/checkpoint_episode).
        caption (str): Caption for the video.
        fps (int): Frames per second.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Case 1: Connect to specific external run (e.g. from evaluation.py trying to update training run)
    if run_path:
        try:
            print(f"Uploading video to specific WandB run: {run_path}...")
            
            # Allow active run to handle it if it matches? 
            # No, if run_path is provided, we assume we want to target THAT run specifically.
            # But we must check if we are already in that run? 
            # Simpler: If run_path provided, we prioritize it (resume='must'). 
            # Note: wandb.init will close current run if we don't manage it carefully. 
            # My evaluation.py logic handled this by checking wandb.run. 
            # But usually upload_video is called ONCE per checkpoint.
            
            # Logic:
            # If we are already logging to the CORRECT run, just log.
            # If not, we might need to init. 
            
            # For simplicity matching the successful fix:
            # If run_path is passed, we check if we need to init.
            
            # Parse path
            path_parts = run_path.strip().split('/')
            entity = None
            project = None
            run_id = None
            
            if len(path_parts) == 3:
                entity, project, run_id = path_parts
            elif len(path_parts) == 2:
                project, run_id = path_parts
            else:
                run_id = path_parts[0]
            
            # Check if current run matches?
            need_init = True
            if wandb.run is not None:
                if wandb.run.id == run_id:
                    need_init = False
                else:
                    # Active run is different. Close it? Or error?
                    # If called from evaluation.py with --all, we might keep it open.
                    # We assume caller manages the run lifecycle if they want persistence.
                    # But if this is a "one-off" upload helper...
                    pass
            
            if need_init:
                # Ensure previous is finished? 
                if wandb.run is not None:
                     wandb.finish()
                     
                wandb.init(
                    entity=entity,
                    project=project,
                    id=run_id,
                    resume="must",
                    job_type="evaluation"
                )
            
            # Log
            log_dict = {
                "eval/video": wandb.Video(video_path, caption=caption, fps=fps, format="mp4")
            }
            if episode is not None:
                log_dict["eval/checkpoint_episode"] = int(episode)
                
            wandb.log(log_dict, step=step) # step might be None, which is fine (appends)
            
            # If we initialized it here, should we close it?
            # If 'need_init' was True, we opened it. We should probably close it to be safe 
            # UNLESS the caller expects it to stay open (e.g. --all loop).
            # This utility function is ambiguous about ownership. 
            # Better pattern: Caller manages init, this just logs if active. 
            # OR this function is "Upload and Go".
            
            # Let's stick to "Upload and Go" style for atomic uploads, 
            # BUT for 'train.py', run_path is None (it uses active run).
            
            if need_init:
                wandb.finish()
                print("  Upload complete (run finished).")
            else:
                print("  Upload complete (run active).")

        except Exception as e:
            print(f"  Error uploading to WandB: {e}")
            
    # Case 2: Use currently active run (e.g. called from train.py)
    elif wandb.run is not None:
        try:
            log_dict = {
                "eval/video": wandb.Video(video_path, caption=caption, fps=fps, format="mp4")
            }
            if episode is not None:
                 log_dict["eval/checkpoint_episode"] = int(episode)
                 
            wandb.log(log_dict, step=step)
            print("  Video logged to active WandB run.")
        except Exception as e:
            print(f"  Error logging to active WandB run: {e}")
    else:
        print("Error: No WandB run active and no run_path provided. Cannot upload.")
