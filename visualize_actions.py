import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation

def load_action_sequence(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data['ee_action_seq'])

def create_coordinate_arrows(ax, position, quaternion, scale=0.1):
    # Convert quaternion to rotation matrix
    r = Rotation.from_quat(quaternion)
    rotation_matrix = r.as_matrix()
    
    
    # Transform rotation matrix to match new coordinate system
    transform = np.array([[1, 0, 0],
                         [0, 0, -1], 
                         [0, 1, 0]])
    rotation_matrix = transform @ rotation_matrix @ transform.T
    
    # Create arrows for each axis (x=red, y=green, z=blue)
    colors = ['red', 'green', 'blue']
    for i in range(3):
        direction = rotation_matrix[:, i] * scale
        ax.quiver(position[0], position[1], position[2],
                 direction[0], direction[1], direction[2],
                 color=colors[i], alpha=0.8, linewidth=2)

def update(frame, actions, ax, azim=90):
    ax.cla()
    
    # Print progress
    print(f"Processing frame {frame}/{len(actions)} ({frame/len(actions)*100:.1f}%)")
    
    # Set axis labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Z')  # Changed from Y to Z
    ax.set_zlabel('Y')  # Changed from Z to Y
    
    # Set the viewing angle (azimuth and elevation)
    ax.view_init(elev=20, azim=azim)  # Rotated counterclockwise
    
    # Get min and max coordinates for setting bounds
    positions = actions[:, :3]  # [x, y, z]
    # Rearrange coordinates to [x, z, -y]
    positions_transformed = np.column_stack([
        positions[:, 0],    # x stays the same
        positions[:, 2],    # z goes to y-axis
        -positions[:, 1]    # negative y goes to z-axis
    ])
    
    min_coords = np.min(positions_transformed, axis=0) - 0.1
    max_coords = np.max(positions_transformed, axis=0) + 0.1
    
    ax.set_xlim([min_coords[0], max_coords[0]])
    ax.set_ylim([min_coords[1], max_coords[1]])
    ax.set_zlim([min_coords[2], max_coords[2]])
    
    # Plot all waypoints as larger dots with more visible color
    ax.scatter(positions_transformed[:, 0], 
              positions_transformed[:, 1], 
              positions_transformed[:, 2], 
              color='darkgray', s=50, alpha=0.8)
    
    # Plot trajectory up to current frame
    positions_current = positions_transformed[:frame+1]
    ax.plot3D(positions_current[:, 0], 
              positions_current[:, 1], 
              positions_current[:, 2], 'gray')
    
    # Plot current position
    current_pos = positions_transformed[frame]
    current_quat = actions[frame, 3:7]
    ax.scatter(*current_pos, color='blue', s=100)
    
    # Create coordinate arrows at current position
    create_coordinate_arrows(ax, current_pos, current_quat)
    
    ax.set_title(f'Frame {frame}')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    return ax,

def visualize_trajectory(actions, output_path='trajectory.mp4', azim=90):
    print(f"Starting visualization of {len(actions)} frames...")
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    anim = FuncAnimation(fig, update, frames=len(actions),
                        fargs=(actions, ax, azim),
                        interval=200, blit=False)
    
    print("Saving animation...")
    # writer = animation.FFMpegWriter(fps=5)
    # anim.save(output_path, writer=writer)
    # print(f"Animation saved to {output_path}")
    
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'  # 确保使用系统ffmpeg
    
    writer = animation.FFMpegWriter(
        fps=5, 
        metadata=dict(artist='Me'),
        bitrate=2000,
        codec='libx264',
        extra_args=[
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'slow',
            '-profile:v', 'baseline'
        ]
    )
    
    try:
        anim.save(output_path, writer=writer)
    except Exception as e:
        print(f"Error: {str(e)}")
        # 降级到GIF格式
        anim.save(output_path.replace('.mp4', '.gif'), writer='pillow')
    plt.close()

def main(task_name='DEBUG'):
    # Load action sequence
    json_path = f'/home/franka/R2D2_3dhat/ReKep/outputs/action.json'
    actions = load_action_sequence(json_path)
    
    # Visualize trajectory
    # Generate videos from different viewing angles
    azim_angles = [10, 100, 190, 280]
    for azim in azim_angles:
        print(f"Generating visualization from {azim}° viewing angle...")
        output_path = f'/home/franka/R2D2_3dhat/ReKep/outputs/action_azim{azim}.mp4'
        visualize_trajectory(actions, output_path=output_path, azim=azim)
    print(f"Visualization saved to {output_path}")
if __name__ == "__main__":
    import sys
    # if not given, use DEBUG
    task_name = sys.argv[1] if len(sys.argv) > 1 else 'DEBUG'
    print(f"Visualizing actions for task: {task_name}")
    main(task_name)