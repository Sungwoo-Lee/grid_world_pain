import wandb
import os
import contextlib
import sys

def wandb_login(quiet=False):
    """
    Attempts to login to WandB using a shared API key file if available.
    The file should be named '.wandb_api_key' in the project root.
    """
    # Find project root (where .wandb_api_key should be)
    # We assume this script is in src/utils/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    key_path = os.path.join(project_root, ".wandb_api_key")

    if os.path.exists(key_path):
        try:
            with open(key_path, 'r') as f:
                key = f.read().strip()
            
            if key:
                if not quiet: print(f"Logging into WandB using shared key from {key_path}...")
                wandb.login(key=key)
                return True
        except Exception as e:
            if not quiet: print(f"Warning: Failed to read WandB key from {key_path}: {e}")
    
    # Fallback to default login (uses environment or .netrc)
    if not quiet: print("No shared WandB key found or failed to read. Using default environment/netrc login.")
    return wandb.login()

# Context manager to suppress stdout/stderr

@contextlib.contextmanager
def suppress_output(suppress=False):
    if suppress:
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                yield
    else:
        yield

def upload_video(video_path, run_path=None, step=None, episode=None, caption="Evaluation Video", fps=4, quiet=False, extra_data=None):
    """
    Uploads a video to WandB.
    
    Args:
        video_path (str): Path to the video file.
        run_path (str, optional): If provided, connects to this specific run.
        step (int, optional): The global step to associate with the video. 
        episode (int, optional): The episode number to log as a metric (eval/checkpoint_episode).
        caption (str): Caption for the video.
        fps (int): Frames per second.
        quiet (bool): If True, suppresses print output.
        extra_data (dict, optional): Extra metrics to log along with the video.
    """
    if not os.path.exists(video_path):
        if not quiet: print(f"Error: Video file not found at {video_path}")
        return

    # Case 1: Connect to specific external run
    if run_path:
        try:
            if not quiet: print(f"Uploading video to specific WandB run: {run_path}...")
            
            path_parts = run_path.strip().split('/')
            entity, project, run_id = (None, None, None)
            if len(path_parts) == 3: entity, project, run_id = path_parts
            elif len(path_parts) == 2: project, run_id = path_parts
            else: run_id = path_parts[0]
            
            need_init = True
            if wandb.run is not None and wandb.run.id == run_id:
                need_init = False
            
            if need_init:
                if wandb.run is not None: wandb.finish()
                wandb.init(entity=entity, project=project, id=run_id, resume="must", job_type="evaluation")
            
            log_dict = {
                "eval/video": wandb.Video(video_path, caption=caption, fps=fps, format="mp4")
            }
            if episode is not None:
                log_dict["eval/checkpoint_episode"] = int(episode)
            
            if extra_data:
                log_dict.update(extra_data)
                
            wandb.log(log_dict, step=step)
            
            if need_init:
                wandb.finish()
                if not quiet: print("  Upload complete (run finished).")
            else:
                if not quiet: print("  Upload complete (run active).")
        except Exception as e:
            if not quiet: print(f"  Error uploading video to WandB: {e}")
            
    # Case 2: Use currently active run
    elif wandb.run is not None:
        try:
            with suppress_output(quiet):
                log_dict = {
                    "eval/video": wandb.Video(video_path, caption=caption, format="mp4")
                }
                if episode is not None:
                     log_dict["eval/checkpoint_episode"] = int(episode)
                
                if extra_data:
                    log_dict.update(extra_data)
                     
                wandb.log(log_dict, step=step)
            
            if not quiet: print("  Video logged to active WandB run.")
        except Exception as e:
            if not quiet: print(f"  Error logging video to active WandB run: {e}")
    else:
        if not quiet: print("Error: No WandB run active and no run_path provided. Cannot upload.")

def upload_image(image_path, run_path=None, step=None, episode=None, caption="Analysis Plot", quiet=False, extra_data=None):
    """
    Uploads an image (plot) to WandB.
    
    Args:
        image_path (str): Path to the image file.
        run_path (str, optional): If provided, connects to this specific run.
        step (int, optional): The global step to associate with the image.
        episode (int, optional): The episode number to log as a metric.
        caption (str): Caption for the image.
        quiet (bool): If True, suppresses print output.
        extra_data (dict, optional): Extra metrics to log along with the image.
    """
    if not os.path.exists(image_path):
        if not quiet: print(f"Error: Image file not found at {image_path}")
        return

    # Case 1: Connect to specific external run
    if run_path:
        try:
            if not quiet: print(f"Uploading image to specific WandB run: {run_path}...")
            
            path_parts = run_path.strip().split('/')
            entity, project, run_id = (None, None, None)
            if len(path_parts) == 3: entity, project, run_id = path_parts
            elif len(path_parts) == 2: project, run_id = path_parts
            else: run_id = path_parts[0]
            
            need_init = True
            if wandb.run is not None and wandb.run.id == run_id:
                need_init = False
            
            if need_init:
                if wandb.run is not None: wandb.finish()
                wandb.init(entity=entity, project=project, id=run_id, resume="must", job_type="analysis")
            
            log_dict = {
                "Analysis/ActionScatter": wandb.Image(image_path, caption=caption)
            }
            if episode is not None:
                log_dict["eval/checkpoint_episode"] = int(episode)
            
            if extra_data:
                log_dict.update(extra_data)
                
            wandb.log(log_dict, step=step)
            
            if need_init:
                wandb.finish()
                if not quiet: print("  Upload complete (run finished).")
            else:
                if not quiet: print("  Upload complete (run active).")
        except Exception as e:
            if not quiet: print(f"  Error uploading image to WandB: {e}")
            
    # Case 2: Use currently active run
    elif wandb.run is not None:
        try:
            with suppress_output(quiet):
                log_dict = {
                    "Analysis/ActionScatter": wandb.Image(image_path, caption=caption)
                }
                if episode is not None:
                     log_dict["eval/checkpoint_episode"] = int(episode)
                
                if extra_data:
                    log_dict.update(extra_data)
                     
                wandb.log(log_dict, step=step)
            
            if not quiet: print("  Image logged to active WandB run.")
        except Exception as e:
            if not quiet: print(f"  Error logging image to active WandB run: {e}")
    else:
        if not quiet: print("Error: No WandB run active and no run_path provided. Cannot upload.")
