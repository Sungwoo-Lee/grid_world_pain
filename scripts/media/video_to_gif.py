import imageio
import sys
import os

def convert_video_to_gif(input_path, output_path, fps=10, loop=0):
    """Converts a video file to an optimized GIF."""
    if not os.path.exists(input_path):
        print(f"Error: Input video not found at {input_path}")
        return

    print(f"Converting {input_path} to {output_path}...")
    reader = imageio.get_reader(input_path)
    frames = []
    for frame in reader:
        frames.append(frame)
    
    imageio.mimsave(output_path, frames, fps=fps, loop=loop)
    print(f"Optimization complete. Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python video_to_gif.py <input_mp4> <output_gif> [fps]")
        sys.exit(1)
        
    in_video = sys.argv[1]
    out_gif = sys.argv[2]
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    convert_video_to_gif(in_video, out_gif, fps=fps)
